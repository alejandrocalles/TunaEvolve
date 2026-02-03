#!/usr/bin/env python3
from pathlib import Path
from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
from shinka.core import TunaEvolutionRunner
import asyncio
from shinka.train import TunaEvolveLauncher


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    print("Experiment configurations:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    job_cfg = hydra.utils.instantiate(cfg.job_config)
    db_cfg = hydra.utils.instantiate(cfg.db_config)
    evo_cfg = hydra.utils.instantiate(cfg.evo_config)
    training_cfg = hydra.utils.instantiate(cfg.training_config)

    evo_runner = TunaEvolutionRunner(
        evo_config=evo_cfg,
        job_config=job_cfg,
        db_config=db_cfg,
        verbose=cfg.verbose,
    )

    launcher = TunaEvolveLauncher(
        evolution_runner=evo_runner,
        training_config=training_cfg,
        verbose=cfg.verbose,
    )

    asyncio.run(launcher.launch())


if __name__ == "__main__":
    main()

