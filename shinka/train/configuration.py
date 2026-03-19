from typing import List, Optional, Literal
import dataclasses
import peft

@dataclasses.dataclass
class HardwareConfig:
    fp16: bool = True
    bf16: bool = False

@dataclasses.dataclass
class LoRAConfig:
    enabled: bool = False
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = dataclasses.field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"

    def to_peft_config(self) -> Optional[peft.LoraConfig]:
        if not self.enabled:
            return None
        
        return peft.LoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type=self.task_type,
        )

@dataclasses.dataclass
class EvoTuneConfig:
    """Configuration specific to the EvoTune algorithm.

    Attributes:
        training_enabled:
            Indicates whether training is enabled or not. If False,
            evolution will not be interrupted until all generations are
            completed.
        num_generations_per_period:
            The number of generations of programs that must be produced before
            a training period starts.
    """
    training_enabled: bool = True
    num_generations_per_period: int = 3

"""
DPO-specific configuration
"""

@dataclasses.dataclass
class DPOHyperparameters:
    beta: float = 0.1
    learning_rate: float= 5e-6
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    epochs: int = 1
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.1
    max_length: int = 1024
    max_prompt_length: int = 512

@dataclasses.dataclass
class DPOTrainingConfig:
    model_id: str
    base_output_dir: str
    logging_steps: int = 10

    evotune: EvoTuneConfig = dataclasses.field(default_factory=EvoTuneConfig)
    hardware: HardwareConfig = dataclasses.field(default_factory=HardwareConfig)
    hyperparameters: DPOHyperparameters = dataclasses.field(default_factory=DPOHyperparameters)
    lora: LoRAConfig = dataclasses.field(default_factory=LoRAConfig)
