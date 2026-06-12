"""
WorldStereo WAM Inference Script.

Usage:
    # Camera control (single view)
    python scripts/inference.py task_type=camera_control input_path=examples/images

    # Multi-trajectory panorama
    python scripts/inference.py task_type=panorama input_path=examples/panorama

    # Reconstruction
    python scripts/inference.py task_type=reconstruction input_path=examples/reconstruction

    # With different model types
    python scripts/inference.py model_type=worldstereo-memory-dmd
    python scripts/inference.py model_type=worldstereo-memory
    python scripts/inference.py model_type=worldstereo-camera

    # With quantization
    python scripts/inference.py w8a8=true w8a8_save_path=quantized/transformer.pt
"""

import hydra
from omegaconf import DictConfig

from worldstereo_wam.runtime import run_inference
from worldstereo_wam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="inference", version_base="1.3")
def main(cfg: DictConfig):
    run_inference(cfg)


if __name__ == "__main__":
    main()
