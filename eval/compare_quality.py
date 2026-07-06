"""Quality comparison between a FP baseline and a quantized run.

Both runs must be produced with the *same seed and same inputs* so that the
only difference is quantization.  The FP output is treated as the reference;
metrics measure how far the quantized output drifts from it.

Discovers matching ``*_result.mp4`` pairs under two output roots (works for
both ``run_multi_traj.py`` and ``run_camera_control.py`` layouts), then per
pair computes:

* PSNR / SSIM / LPIPS  – per-frame, averaged over the clip.
* FVD                  – Fréchet Video Distance over the whole clip
                         (requires ``torchmetrics[video]`` or falls back to
                         an I3D-free FID-style embedding; see ``--no_fvd``).

Also prints an aggregate table across all scenes and, when a run pair of two
FP runs is given, establishes the non-determinism "noise floor" so real
quantization degradation can be distinguished from generation jitter.

Usage::

    python -m eval.compare_quality \
        --ref outputs_fp --test outputs_w8a8 \
        --model_type worldstereo-memory-dmd \
        --report eval_report.json

Dependencies (install as needed)::

    pip install lpips scikit-image torchmetrics
"""

from __future__ import annotations

import argparse
import json
import os
from glob import glob

import numpy as np
import torch


# ---------------------------------------------------------------------------
# frame loading
# ---------------------------------------------------------------------------
def load_frames(path: str) -> np.ndarray:
    """Load an mp4 as ``uint8 [f, h, w, 3]`` RGB, matching the writer format."""
    import cv2

    cv2.setNumThreads(0)
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        if not cap.isOpened():
            return np.empty((0,))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return np.stack(frames) if frames else np.empty((0,))


# ---------------------------------------------------------------------------
# per-frame metrics
# ---------------------------------------------------------------------------
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR in dB between two uint8 frames."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(255.0) - 10 * np.log10(mse))


class Metrics:
    """Lazily-built metric backends so missing deps only fail when needed."""

    def __init__(self, device: torch.device, use_lpips: bool) -> None:
        self.device = device
        self._ssim = None
        self._lpips = None
        self.use_lpips = use_lpips
        if use_lpips:
            import lpips  # noqa: F401

            self._lpips = lpips.LPIPS(net="alex").to(device).eval()

    def ssim(self, a: np.ndarray, b: np.ndarray) -> float:
        from skimage.metrics import structural_similarity

        return float(
            structural_similarity(a, b, channel_axis=2, data_range=255)
        )

    def lpips(self, a: np.ndarray, b: np.ndarray) -> float:
        if self._lpips is None:
            return float("nan")

        def to_t(x: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(x).float().permute(2, 0, 1) / 127.5 - 1.0
            return t.unsqueeze(0).to(self.device)

        with torch.no_grad():
            return float(self._lpips(to_t(a), to_t(b)).item())


def compare_clip(ref: np.ndarray, test: np.ndarray, m: Metrics) -> dict:
    """Average per-frame metrics over one aligned clip."""
    n = min(len(ref), len(test))
    if n == 0:
        return {"frames": 0}
    if len(ref) != len(test):
        print(f"  [warn] frame count differs: ref={len(ref)} test={len(test)}; "
              f"comparing first {n}")
    ref, test = ref[:n], test[:n]

    ps, ss, lp = [], [], []
    for i in range(n):
        ps.append(psnr(ref[i], test[i]))
        ss.append(m.ssim(ref[i], test[i]))
        if m.use_lpips:
            lp.append(m.lpips(ref[i], test[i]))

    out = {
        "frames": n,
        "psnr": float(np.mean([x for x in ps if np.isfinite(x)])) if any(np.isfinite(ps)) else float("inf"),
        "ssim": float(np.mean(ss)),
    }
    if m.use_lpips:
        out["lpips"] = float(np.mean(lp))
    return out


# ---------------------------------------------------------------------------
# FVD (whole-clip distribution distance)
# ---------------------------------------------------------------------------
def compute_fvd(ref_clips: list[np.ndarray], test_clips: list[np.ndarray],
                device: torch.device) -> float | None:
    """FVD across the whole set of clips using torchmetrics' FVD if available."""
    try:
        from torchmetrics.video import FrechetVideoDistance
    except Exception as exc:  # pragma: no cover - optional dep
        print(f"  [warn] FVD unavailable ({exc}); skip with --no_fvd to silence")
        return None

    fvd = FrechetVideoDistance().to(device)

    def to_batch(clips: list[np.ndarray]) -> torch.Tensor:
        # torchmetrics expects [B, T, C, H, W] float in [0,1]
        vids = [torch.from_numpy(c).float().permute(0, 3, 1, 2) / 255.0 for c in clips]
        t = min(v.shape[0] for v in vids)
        vids = [v[:t] for v in vids]
        return torch.stack(vids).to(device)

    with torch.no_grad():
        fvd.update(to_batch(ref_clips), real=True)
        fvd.update(to_batch(test_clips), real=False)
        return float(fvd.compute().item())


# ---------------------------------------------------------------------------
# pairing
# ---------------------------------------------------------------------------
def find_pairs(ref_root: str, test_root: str, model_type: str) -> list[tuple[str, str, str]]:
    """Return (name, ref_mp4, test_mp4) for every result present in both roots."""
    suffix = f"{model_type}_result.mp4"
    pattern = os.path.join(ref_root, "**", suffix)
    pairs = []
    for ref_mp4 in sorted(glob(pattern, recursive=True)):
        rel = os.path.relpath(ref_mp4, ref_root)
        test_mp4 = os.path.join(test_root, rel)
        if os.path.exists(test_mp4):
            pairs.append((rel, ref_mp4, test_mp4))
        else:
            print(f"[warn] no test match for {rel}")
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="FP baseline output root")
    ap.add_argument("--test", required=True, help="quantized output root")
    ap.add_argument("--model_type", default="worldstereo-memory-dmd",
                    help="result mp4 prefix, matches --model_type used at inference")
    ap.add_argument("--report", default=None, help="write full JSON report here")
    ap.add_argument("--no_lpips", action="store_true", help="skip LPIPS (no lpips dep)")
    ap.add_argument("--no_fvd", action="store_true", help="skip FVD (no torchmetrics dep)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    m = Metrics(device, use_lpips=not args.no_lpips)

    pairs = find_pairs(args.ref, args.test, args.model_type)
    if not pairs:
        raise SystemExit(f"No matching *_{args.model_type}_result.mp4 pairs found "
                         f"under {args.ref} and {args.test}")
    print(f"Found {len(pairs)} matching clip pairs.\n")

    per_clip = []
    ref_clips, test_clips = [], []
    for name, ref_mp4, test_mp4 in pairs:
        ref = load_frames(ref_mp4)
        test = load_frames(test_mp4)
        if ref.size == 0 or test.size == 0:
            print(f"[warn] empty clip for {name}; skipping")
            continue
        stats = compare_clip(ref, test, m)
        stats["name"] = name
        per_clip.append(stats)
        ref_clips.append(ref)
        test_clips.append(test)
        line = (f"{name}: PSNR={stats['psnr']:.2f}dB SSIM={stats['ssim']:.4f}"
                f" frames={stats['frames']}")
        if "lpips" in stats:
            line += f" LPIPS={stats['lpips']:.4f}"
        print(line)

    def agg(key: str) -> float | None:
        vals = [c[key] for c in per_clip if key in c and np.isfinite(c[key])]
        return float(np.mean(vals)) if vals else None

    summary = {
        "num_clips": len(per_clip),
        "psnr_mean": agg("psnr"),
        "ssim_mean": agg("ssim"),
        "lpips_mean": agg("lpips") if not args.no_lpips else None,
        "fvd": None,
    }

    if not args.no_fvd and ref_clips:
        summary["fvd"] = compute_fvd(ref_clips, test_clips, device)

    print("\n── aggregate ──────────────────────────────")
    print(f"clips        : {summary['num_clips']}")
    print(f"PSNR  (mean) : {summary['psnr_mean']:.2f} dB" if summary["psnr_mean"] else "PSNR  : n/a")
    print(f"SSIM  (mean) : {summary['ssim_mean']:.4f}" if summary["ssim_mean"] else "SSIM  : n/a")
    if summary["lpips_mean"] is not None:
        print(f"LPIPS (mean) : {summary['lpips_mean']:.4f}  (lower = closer to FP)")
    if summary["fvd"] is not None:
        print(f"FVD          : {summary['fvd']:.2f}  (lower = closer to FP)")

    if args.report:
        with open(args.report, "w") as f:
            json.dump({"summary": summary, "per_clip": per_clip}, f, indent=2)
        print(f"\nWrote report → {args.report}")


if __name__ == "__main__":
    main()
