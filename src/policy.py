"""Qwen policy model with LoRA support for RL training."""

from dataclasses import dataclass, field
from typing import Any, Optional, Union
import torch
import torch.nn as nn

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


@dataclass
class PolicyConfig:
    """Configuration for the policy model."""
    model_name: str = "Qwen/Qwen3-1.7B"
    max_length: int = 2048
    dtype: str = "bfloat16"
    device_map: str = "auto"

    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Generation defaults
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 512


class QwenPolicy(nn.Module):
    """Qwen model wrapper with LoRA for RL fine-tuning."""

    def __init__(self, config: Optional[PolicyConfig] = None):
        super().__init__()
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers required: pip install transformers")
        self.config = config or PolicyConfig()
        self._model = None
        self._tokenizer = None

    def load(self) -> "QwenPolicy":
        """Load model and tokenizer, apply LoRA."""
        if self._model is not None:
            return self

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        torch_dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype,
            device_map=self.config.device_map,
            trust_remote_code=True,
        )

        if self.config.use_lora:
            if not HAS_PEFT:
                raise ImportError("peft required for LoRA: pip install peft")
            peft_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self._model = get_peft_model(self._model, peft_config)
            self._model.print_trainable_parameters()

        return self

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self.load()
        return self._tokenizer

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> Any:
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def generate(
        self,
        messages: Optional[list[dict]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Generate from messages or pre-tokenized input."""
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature

        if messages is not None and input_ids is None:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = self.tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=self.config.max_length - max_new_tokens,
            )
            input_ids = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs["attention_mask"].to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_p=self.config.top_p if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )
        return outputs

    def generate_turn(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate one assistant turn, stopping at <|im_end|>.

        Returns raw text INCLUDING <think>, <tool_call> tags.
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature

        template_kwargs = dict(
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        if tools:
            template_kwargs["tools"] = tools

        text = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        inputs = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=self.config.max_length - max_new_tokens,
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)
        prompt_len = input_ids.shape[1]

        # Stop at <|im_end|>
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        eos_ids = [self.tokenizer.eos_token_id]
        if isinstance(im_end_id, int) and im_end_id != self.tokenizer.unk_token_id:
            eos_ids.append(im_end_id)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_ids,
            )

        new_tokens = output_ids[0][prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=False)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]
