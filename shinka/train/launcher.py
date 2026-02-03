import pathlib
import logging
import math
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from shinka.core import TunaEvolutionRunner
from .dpo import launch_dpo
from .configuration import DPOTrainingConfig
from .dataset import DatabaseWrapper
from shinka.vllm import VLLMServer, VLLMConfig

logger = logging.getLogger(__name__)

class TunaEvolveLauncher:
    def __init__(
        self,
        evolution_runner: TunaEvolutionRunner,
        training_config: DPOTrainingConfig,
        vllm_config: VLLMConfig,
        verbose: bool = False,
    ):
        self.evolution_runner = evolution_runner
        self.training_config = training_config
        self.vllm_config = vllm_config
        self.verbose = verbose

        self.database_wrapper = DatabaseWrapper(evolution_runner.db)
        self.base_output_dir = pathlib.Path(self.training_config.base_output_dir)

    async def launch(self) -> None:
        """Start this launcher."""

        await self.evolution_runner.start()
        num_generations = self.evolution_runner.evo_config.num_generations

        if num_generations == 0:
            logger.warning(f"num_generations was 0, exiting...")
            return

        if not self.training_config.evotune.training_enabled:
            logger.warning(f"Training disabled, running evolution runner for {num_generations} generations")
            if (model_id := self._get_model_id()) is None:
                return

            vllm_server = VLLMServer(
                model_path_or_id=model_id,
                served_model_name=model_id,
                host="0.0.0.0",
                port=8000,
                config=self.vllm_config,
            )
            with vllm_server:
                await self.evolution_runner.run(num_steps=num_generations)

            return
        
        
        if (model_id := self._get_model_id()) is None:
            return

        logger.info(f"Loading model {model_id}...")
        model = AutoModelForCausalLM.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)

        period_index = 0
        logger.info(f"Saving model for period 0 to {self._save_dir(period_index=period_index)}...")
        model.save_pretrained(self._save_dir(period_index=period_index))
        processor.save_pretrained(self._save_dir(period_index=period_index))

        num_generations_per_period = self.training_config.evotune.num_generations_per_period

        while True:
            # ==================================================================
            #       Evolution
            # ==================================================================
            vllm_server = VLLMServer(
                model_path_or_id=self._save_dir(period_index=period_index),
                served_model_name=model_id,
                host="0.0.0.0",
                port=8000,
                config=self.vllm_config,
            )
            with vllm_server:
                await self.evolution_runner.run(num_steps=num_generations_per_period)

            if self.evolution_runner.completed_generations >= num_generations:
                break

            # ==================================================================
            #       Training
            # ==================================================================
            dataset = self.database_wrapper.build_dpo_dataset()
            launch_dpo(
                dataset=dataset,
                model_dir=self._save_dir(period_index=period_index),
                checkpoints_dir=self._base_checkpoints_dir(period_index=period_index + 1),
                save_dir=self._save_dir(period_index=period_index + 1),
                hyperparameters=self.training_config.hyperparameters,
                hardware_config=self.training_config.hardware,
                lora_config=self.training_config.lora,
                logging_steps=self.training_config.logging_steps,
            )
            torch.cuda.empty_cache()
            period_index += 1
    
    def _base_checkpoints_path(self, period_index: int) -> pathlib.Path:
        num_digits = math.ceil(math.log(self._num_training_periods, 10))
        return self.base_output_dir.joinpath(f"checkpoints_for_period_{period_index:0{num_digits}d}")
    
    def _base_checkpoints_dir(self, period_index: int) -> str:
        return str(self._base_checkpoints_path(period_index))
    
    def _save_path(self, period_index: int) -> pathlib.Path:
        """The path where the final model of a training period should be saved."""
        return self._base_checkpoints_path(period_index).joinpath("latest")
    
    def _save_dir(self, period_index: int) -> str:
        """The directory where the final model of a training period should be saved."""
        return str(self._save_path(period_index))

    @property
    def _num_training_periods(self):
        if self.evolution_runner.evo_config.num_generations <= 0:
            return 0
        return (self.evolution_runner.evo_config.num_generations - 1) // self.training_config.evotune.num_generations_per_period
    
    def _get_model_id(self) -> Optional[str]:
        model_id = self.training_config.model_id
        valid = any([
            model_id in [model_name, model_name.split("local-")[-1]]
            for model_name in self.evolution_runner.llm.model_names
        ])
        if not valid:
            # This could be a warning, and we could continue,
            # but it's better to crash as soon as possible
            logger.error(
                f"The trained model id was {model_id}, but it was not found "
                f"in the list of model names"
            )
            return None
        return model_id