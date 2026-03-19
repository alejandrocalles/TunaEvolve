import logging
from typing import Union

from transformers import AutoProcessor
import datasets
from trl.trainer.dpo_config import DPOConfig
from trl.trainer.dpo_trainer import DPOTrainer
from .configuration import DPOHyperparameters, HardwareConfig, LoRAConfig

logger = logging.getLogger(__name__)



def launch_dpo(
        dataset: Union[datasets.Dataset, datasets.IterableDataset],
        model_dir: str,
        checkpoints_dir: str,
        save_dir: str,
        hyperparameters: DPOHyperparameters,
        hardware_config: HardwareConfig,
        lora_config: LoRAConfig,
        logging_steps: int
    ) -> None:
    """
    Trains a model using Direct Preference Optimization (DPO) based on the database history.
    
    Args:
        dataset:
            A preference dataset as described in https://huggingface.co/docs/trl/main/en/dataset_formats#preference
        model_dir:
            Path to the base model weights.
        checkpoints_dir:
            Path to save the model checkpoints to.
        save_dir:
            Path to save the final model to.
        hyperparameters:
            Hyperparameters specific to DPO training.
        hardware_config:
            Configuration for hardware or precision parameters.
        lora_config:
            Configuration for Low-Rank Adapters.
        logging_steps:
            See `DPOConfig` documentation for the parameter of the same name.
    """

    if isinstance(dataset, datasets.Dataset):
        if len(dataset) == 0:
            return
        logger.info(f"DPO training launched with {len(dataset)} preference pairs")
    else:
        logger.info("DPO training launched")

    processor = AutoProcessor.from_pretrained(model_dir)

    training_args = DPOConfig(
        output_dir=checkpoints_dir,
        beta=hyperparameters.beta,
        
        # gradient descent params
        learning_rate=hyperparameters.learning_rate,
        per_device_train_batch_size=hyperparameters.batch_size,
        gradient_accumulation_steps=hyperparameters.gradient_accumulation_steps,
        num_train_epochs=hyperparameters.epochs,
        lr_scheduler_type=hyperparameters.lr_scheduler,
        warmup_ratio=hyperparameters.warmup_ratio,
        
        # precision/hardware
        fp16=hardware_config.fp16,
        bf16=hardware_config.bf16,
        
        # logging
        logging_steps=logging_steps,
        save_strategy="epoch",
        
        # data pre-processing
        max_length=hyperparameters.max_length,
        max_prompt_length=hyperparameters.max_prompt_length,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model_dir,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=lora_config.to_peft_config(),
    )

    logger.info("Starting DPO training")
    trainer.train()

    logger.info(f"Saving model to {save_dir}...")
    
    trainer.save_model(save_dir)
    processor.save_pretrained(save_dir)

    logger.info("Training complete")