"""Public settings; task and control are selected on TrainerPlus."""
from dataclasses import dataclass
import math
from pathlib import Path


def positive(name, value, *, optional=False):
    if optional and value is None:
        return
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: str | Path = "runs"
    num_epochs: int = 1
    max_steps: int | None = None
    microbatch_size: int = 1
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "no"
    seed: int = 42
    shuffle: bool = True
    drop_last: bool = False
    lr_schedule: str = "constant"
    warmup_ratio: float = 0.0
    max_grad_norm: float | None = None
    native_compile: bool = False
    logging_steps: int = 50
    eval_steps: int | None = None
    save_steps: int | None = None
    save_model: bool = True

    def __post_init__(self):
        for name in ("num_epochs", "microbatch_size", "gradient_accumulation_steps", "logging_steps"):
            positive(name, getattr(self, name))
        for name in ("max_steps", "eval_steps", "save_steps"):
            positive(name, getattr(self, name), optional=True)
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for name in ("shuffle", "drop_last", "native_compile", "save_model"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be boolean")
        if self.mixed_precision not in {"no", "bf16"}:
            raise ValueError("mixed_precision must be no or bf16")
        if self.lr_schedule not in {"constant", "cosine"}:
            raise ValueError("lr_schedule must be constant or cosine")
        if not math.isfinite(self.warmup_ratio) or not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.lr_schedule == "constant" and self.warmup_ratio:
            raise ValueError("Use cosine for the qualified warmup schedule")
        if self.max_grad_norm is not None and (not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0):
            raise ValueError("max_grad_norm must be positive or None")

    @property
    def effective_batch_size(self):
        return self.microbatch_size * self.gradient_accumulation_steps


@dataclass(frozen=True)
class DataConfig:
    max_length: int = 512
    padding: str = "fixed"
    truncation: str = "error"
    packing: bool = False
    add_eos: bool = True
    sft_loss: str = "auto"
    text_field: str = "text"
    prompt_field: str = "prompt"
    completion_field: str = "completion"
    messages_field: str = "messages"

    def __post_init__(self):
        positive("max_length", self.max_length)
        if self.max_length < 2:
            raise ValueError("max_length must be at least two")
        if self.padding not in {"fixed", "dynamic"}:
            raise ValueError("padding must be fixed or dynamic")
        if self.truncation not in {"error", "truncate"}:
            raise ValueError("truncation must be error or truncate")
        if self.sft_loss not in {"auto", "all", "completion", "assistant"}:
            raise ValueError("sft_loss must be auto, all, completion or assistant")
        if type(self.packing) is not bool or type(self.add_eos) is not bool:
            raise TypeError("packing and add_eos must be boolean")
        for name in ("text_field", "prompt_field", "completion_field", "messages_field"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")
