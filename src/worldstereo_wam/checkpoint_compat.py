"""
Checkpoint compatibility utilities for WorldStereo WAM.

This module helps answer the question: *Can I reuse a FastWAM checkpoint?*

Short answer: a FastWAM checkpoint cannot be loaded into WorldStereo directly,
because the two models have fundamentally different architectures:

    FastWAM   = Wan2.2-TI2V-5B + MoT(video expert + action expert)
                + ActionDiT + proprio/action heads + StereoEncoder
    WorldStereo = WanTransformer3DModel + ControlNet (camera / GGM / SSM)
                + diffusers pipeline

So a full `strict=True` load is impossible. What *is* possible:

  1. Both are built on the Wan DiT backbone, so a subset of transformer-block
     weights may share names/shapes. Those can be transplanted as an
     initialization (not a guarantee of quality).
  2. This tool inspects a FastWAM state dict against a WorldStereo model and
     reports which tensors are name+shape compatible, then optionally builds a
     filtered state dict you can load with `strict=False`.

Use `inspect_compatibility()` to get a report, and `extract_loadable_subset()`
to build a transplantable state dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .utils.logging_config import get_logger

logger = get_logger(__name__)


def _load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a state dict from a .pt/.pth/.bin or .safetensors file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path), device="cpu")

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    # Unwrap common container keys
    for key in ("state_dict", "model", "module", "weights"):
        if isinstance(payload, dict) and key in payload and isinstance(payload[key], dict):
            payload = payload[key]
            break
    if not isinstance(payload, dict):
        raise ValueError(f"Could not interpret checkpoint at {path} as a state dict.")
    return payload


@dataclass
class CompatibilityReport:
    """Summary of how compatible a source checkpoint is with a target model."""

    matched: list[str] = field(default_factory=list)            # name + shape match
    shape_mismatch: list[tuple[str, tuple, tuple]] = field(default_factory=list)
    missing_in_source: list[str] = field(default_factory=list)  # target has, source lacks
    unused_in_source: list[str] = field(default_factory=list)   # source has, target lacks

    @property
    def num_matched(self) -> int:
        return len(self.matched)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "Checkpoint compatibility report",
            "=" * 60,
            f"  Name+shape matched : {len(self.matched)}",
            f"  Shape mismatch     : {len(self.shape_mismatch)}",
            f"  Missing in source  : {len(self.missing_in_source)} (target params not provided)",
            f"  Unused in source   : {len(self.unused_in_source)} (source params dropped)",
            "=" * 60,
        ]
        if self.matched:
            preview = self.matched[:5]
            lines.append(f"  Example matched keys: {preview}")
        if self.shape_mismatch:
            ex = self.shape_mismatch[:3]
            lines.append(f"  Example shape mismatch: {ex}")
        if len(self.matched) == 0:
            lines.append(
                "  => No transplantable weights found. FastWAM and WorldStereo "
                "do not share a compatible parameter namespace."
            )
        return "\n".join(lines)


def inspect_compatibility(
    source_checkpoint: str | Path,
    target_model: torch.nn.Module,
    *,
    strip_prefixes: tuple[str, ...] = ("module.", "model.", "_orig_mod."),
) -> CompatibilityReport:
    """
    Compare a source checkpoint against a target model's state dict.

    Args:
        source_checkpoint: Path to a FastWAM (or other) checkpoint file.
        target_model: The WorldStereo model (or any nn.Module) to load into.
        strip_prefixes: Leading prefixes to strip from source keys before matching.

    Returns:
        CompatibilityReport describing matched / mismatched / unused tensors.
    """
    source = _load_state_dict(source_checkpoint)

    def _strip(name: str) -> str:
        for prefix in strip_prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
        return name

    source = {_strip(k): v for k, v in source.items()}
    target = target_model.state_dict()

    report = CompatibilityReport()
    for name, t_tensor in target.items():
        if name not in source:
            report.missing_in_source.append(name)
            continue
        s_tensor = source[name]
        if tuple(s_tensor.shape) == tuple(t_tensor.shape):
            report.matched.append(name)
        else:
            report.shape_mismatch.append(
                (name, tuple(s_tensor.shape), tuple(t_tensor.shape))
            )

    target_keys = set(target.keys())
    for name in source:
        if name not in target_keys:
            report.unused_in_source.append(name)

    logger.info(report.summary())
    return report


def extract_loadable_subset(
    source_checkpoint: str | Path,
    target_model: torch.nn.Module,
    *,
    strip_prefixes: tuple[str, ...] = ("module.", "model.", "_orig_mod."),
) -> dict[str, torch.Tensor]:
    """
    Build a filtered state dict containing only name+shape compatible tensors.

    The returned dict can be loaded with `target_model.load_state_dict(sd, strict=False)`.
    This is an *initialization* helper, not a guarantee of correctness.

    Args:
        source_checkpoint: Path to the source checkpoint.
        target_model: Target model to match against.
        strip_prefixes: Leading prefixes to strip from source keys.

    Returns:
        A state dict subset whose keys/shapes match the target model.
    """
    source = _load_state_dict(source_checkpoint)

    def _strip(name: str) -> str:
        for prefix in strip_prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
        return name

    source = {_strip(k): v for k, v in source.items()}
    target = target_model.state_dict()

    loadable = {
        name: tensor
        for name, tensor in source.items()
        if name in target and tuple(tensor.shape) == tuple(target[name].shape)
    }
    logger.info(
        "Extracted %d/%d transplantable tensors from %s",
        len(loadable),
        len(target),
        source_checkpoint,
    )
    return loadable
