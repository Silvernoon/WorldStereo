"""
WorldStereo WAM Training Script.

Usage:
    python scripts/train.py

Override config values via command line:
    python scripts/train.py learning_rate=1e-5 batch_size=4
    python scripts/train.py model=worldstereo data=custom_dataset
"""

import hydra
from omegaconf import DictConfig

from worldstereo_wam.runtime import run_training
from worldstereo_wam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
