"""Efficiency instrumentation for quantization evaluation.

Zero-overhead when disabled: the entry scripts only build a live profiler
when ``WORLDSTEREO_PROFILE`` is set, otherwise ``get_profiler`` returns a
no-op object whose methods do nothing.

Enable via environment variables (read once, at ``get_profiler`` time):

* ``WORLDSTEREO_PROFILE``      – any non-empty value turns profiling on.
* ``WORLDSTEREO_PROFILE_OUT``  – output directory (default ``profile_out``).
* ``WORLDSTEREO_PROFILE_LABEL``– config label written into the report
                                 (e.g. ``fp``, ``w8a8``, ``w8a8_all``).

Each rank writes ``<out>/<label>_rank<r>.json`` containing per-clip latency
(CUDA-event milliseconds), peak allocated / reserved VRAM, and the
model-resident VRAM measured right after load.

Usage (inside an entry script)::

    from eval.profiler import get_profiler
    prof = get_profiler(device)
    prof.snapshot_model_memory()          # after model load, before inference
    prof.reset_peak()                     # start of inference region
    ...
    with prof.clip():                     # around each pipeline() call
        output = pipeline(**kwargs)
    ...
    prof.finalize()                       # once, at the end
"""

from __future__ import annotations

import json
import os
import statistics
import time
from contextlib import contextmanager

import torch


_MB = 1024.0 * 1024.0


class _NoOpProfiler:
    """Returned when profiling is disabled; every method is a cheap no-op."""

    enabled = False

    def snapshot_model_memory(self) -> None: ...
    def reset_peak(self) -> None: ...
    def finalize(self) -> None: ...

    @contextmanager
    def clip(self):
        yield


class Profiler:
    """Records per-clip latency and VRAM peaks for one inference run."""

    enabled = True

    def __init__(self, device: torch.device, out_dir: str, label: str) -> None:
        self.device = device
        self.out_dir = out_dir
        self.label = label
        self.rank = int(os.environ.get("RANK", "0"))
        self.clip_ms: list[float] = []
        self.model_resident_mb: float | None = None
        self._t_start = time.perf_counter()

    # -- memory ---------------------------------------------------------
    def snapshot_model_memory(self) -> None:
        """Record VRAM held by weights/buffers right after model load."""
        torch.cuda.synchronize(self.device)
        self.model_resident_mb = torch.cuda.memory_allocated(self.device) / _MB

    def reset_peak(self) -> None:
        """Reset peak trackers so the inference region is measured cleanly."""
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)

    # -- latency --------------------------------------------------------
    @contextmanager
    def clip(self):
        """Time one pipeline call with CUDA events (accurate GPU wall time)."""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            torch.cuda.synchronize(self.device)
            self.clip_ms.append(start.elapsed_time(end))

    # -- report ---------------------------------------------------------
    def finalize(self) -> None:
        torch.cuda.synchronize(self.device)
        peak_alloc_mb = torch.cuda.max_memory_allocated(self.device) / _MB
        peak_reserved_mb = torch.cuda.max_memory_reserved(self.device) / _MB
        wall_s = time.perf_counter() - self._t_start

        # First clip includes warmup (CUDA graph / compile / alloc); report
        # both the raw list and a warmup-excluded summary.
        warm = self.clip_ms[1:] if len(self.clip_ms) > 1 else self.clip_ms
        summary = {
            "label": self.label,
            "rank": self.rank,
            "device_name": torch.cuda.get_device_name(self.device),
            "num_clips": len(self.clip_ms),
            "clip_ms": [round(x, 2) for x in self.clip_ms],
            "clip_ms_mean_excl_first": round(statistics.mean(warm), 2) if warm else None,
            "clip_ms_median_excl_first": round(statistics.median(warm), 2) if warm else None,
            "model_resident_mb": round(self.model_resident_mb, 1) if self.model_resident_mb is not None else None,
            "peak_allocated_mb": round(peak_alloc_mb, 1),
            "peak_reserved_mb": round(peak_reserved_mb, 1),
            "wall_seconds": round(wall_s, 2),
        }
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f"{self.label}_rank{self.rank}.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        if self.rank == 0:
            print(f"[profiler] wrote {path}")
            print(f"[profiler] {self.label}: "
                  f"resident={summary['model_resident_mb']}MB "
                  f"peak_alloc={summary['peak_allocated_mb']}MB "
                  f"median_clip={summary['clip_ms_median_excl_first']}ms")


_INSTANCE: Profiler | _NoOpProfiler | None = None


def get_profiler(device: torch.device) -> Profiler | _NoOpProfiler:
    """Return a shared profiler; a no-op unless ``WORLDSTEREO_PROFILE`` is set."""
    global _INSTANCE
    if _INSTANCE is None:
        if os.environ.get("WORLDSTEREO_PROFILE"):
            out_dir = os.environ.get("WORLDSTEREO_PROFILE_OUT", "profile_out")
            label = os.environ.get("WORLDSTEREO_PROFILE_LABEL", "run")
            _INSTANCE = Profiler(device, out_dir, label)
        else:
            _INSTANCE = _NoOpProfiler()
    return _INSTANCE
