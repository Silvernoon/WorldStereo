"""
WorldStereo WAM runtime module.

Provides unified API for model creation, inference, and training entry points,
mirroring the FastWAM structure.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

# Add project root to path for importing original modules
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def _normalize_mixed_precision(mixed_precision: str) -> str:
    """Normalize mixed precision string to one of: no, fp16, bf16."""
    if not isinstance(mixed_precision, str):
        raise ValueError(f"`mixed_precision` must be str, got {type(mixed_precision)}")
    key = mixed_precision.strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    """Convert mixed precision string to torch dtype."""
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def create_worldstereo(
    model_path: str = "hanshanxue/WorldStereo",
    model_type: str = "worldstereo-memory-dmd",
    *,
    local_files_only: bool = False,
    sp_world_size: int = 1,
    fsdp: bool = False,
    device_mesh=None,
    device: str | torch.device = "cuda",
    quantize_w8a8: bool = False,
    quantize_transformer_only: bool = True,
    w8a8_save_path: str | None = None,
):
    """
    Create a WorldStereo model instance.

    This is the main entry point for loading WorldStereo models, equivalent to
    FastWAM's `create_fastwam` function.

    Args:
        model_path: Model directory or HuggingFace repo ID.
        model_type: One of 'worldstereo-camera', 'worldstereo-memory', 'worldstereo-memory-dmd'.
        local_files_only: If True, only use cached files.
        sp_world_size: Sequence-parallel degree (1 = disabled).
        fsdp: Enable FSDP model sharding.
        device_mesh: DeviceMesh for distributed training.
        device: Target device.
        quantize_w8a8: Apply W8A8 quantization.
        quantize_transformer_only: Only quantize transformer (not VAE/encoders).
        w8a8_save_path: Path to save/load quantized weights.

    Returns:
        WorldStereo model instance.
    """
    from models.worldstereo_wrapper import WorldStereo

    if isinstance(device, str):
        device = torch.device(device)

    return WorldStereo.from_pretrained(
        model_path,
        subfolder=model_type,
        local_files_only=local_files_only,
        sp_world_size=sp_world_size,
        fsdp=fsdp,
        device_mesh=device_mesh,
        device=device,
        quantize_w8a8=quantize_w8a8,
        quantize_transformer_only=quantize_transformer_only,
        w8a8_save_path=w8a8_save_path,
    )


def run_inference(cfg: DictConfig) -> None:
    """
    Run WorldStereo inference with Hydra config.

    This is the unified inference entry point, equivalent to FastWAM's inference
    runner. Supports both single-view camera control and multi-trajectory modes.

    Args:
        cfg: Hydra DictConfig containing all inference parameters.
    """
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from glob import glob

    from src.general_utils import set_seed, rank0_log, load_video
    from src.sp_utils.parallel_states import initialize_parallel_state
    from diffusers.utils import export_to_video

    setup_logging()

    # Extract config values
    model_path = cfg.get("model_path", "hanshanxue/WorldStereo")
    model_type = cfg.get("model_type", "worldstereo-memory-dmd")
    input_path = cfg.get("input_path", "examples/images")
    output_path = cfg.get("output_path", "outputs")
    task_type = cfg.get("task_type", "camera_control")
    seed = cfg.get("seed", 1024)
    local_files_only = cfg.get("local_files_only", False)
    fsdp = cfg.get("fsdp", False)
    w8a8 = cfg.get("w8a8", False)
    w8a8_all = cfg.get("w8a8_all", False)
    w8a8_save_path = cfg.get("w8a8_save_path", None)

    # Distributed setup. Keep behavior aligned with original WorldStereo scripts:
    # initialize process group even for single-process inference because
    # sequence-parallel helpers query torch.distributed rank/group state.
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            rank=rank,
            world_size=world_size,
        )
    device_num = torch.cuda.device_count()
    device_mesh = init_device_mesh(
        "cuda",
        (world_size // device_num, device_num),
        mesh_dim_names=("rep", "shard"),
    )

    # Sequence parallel setup
    parallel_dims = initialize_parallel_state(sp=world_size)
    sp_size = parallel_dims.sp if parallel_dims.sp_enabled else 1
    sp_rank = parallel_dims.sp_rank if parallel_dims.sp_enabled else 0
    data_rank = rank // sp_size

    global_seed = seed + data_rank
    set_seed(global_seed)
    logger.info(
        f"Global rank:{rank}, Local rank:{local_rank}, "
        f"SP_rank:{sp_rank}, SP_group:{data_rank}, seed:{global_seed}."
    )

    # Load model
    torch.set_default_dtype(torch.float32)
    worldstereo = create_worldstereo(
        model_path=model_path,
        model_type=model_type,
        local_files_only=local_files_only,
        sp_world_size=sp_size,
        fsdp=fsdp,
        device_mesh=device_mesh,
        device=device,
        quantize_w8a8=w8a8 or w8a8_all,
        quantize_transformer_only=not w8a8_all,
        w8a8_save_path=w8a8_save_path,
    )

    # Run appropriate inference mode
    if task_type == "camera_control":
        _run_camera_control_inference(
            worldstereo=worldstereo,
            cfg=cfg,
            input_path=input_path,
            output_path=output_path,
            model_type=model_type,
            device=device,
            seed=seed,
            sp_size=sp_size,
            sp_rank=sp_rank,
            rank=rank,
        )
    else:
        _run_multi_traj_inference(
            worldstereo=worldstereo,
            cfg=cfg,
            input_path=input_path,
            output_path=output_path,
            model_type=model_type,
            task_type=task_type,
            device=device,
            seed=seed,
            sp_size=sp_size,
            sp_rank=sp_rank,
            rank=rank,
        )

    if dist.is_initialized():
        dist.destroy_process_group()


def _run_camera_control_inference(
    worldstereo,
    cfg: DictConfig,
    input_path: str,
    output_path: str,
    model_type: str,
    device: torch.device,
    seed: int,
    sp_size: int,
    sp_rank: int,
    rank: int,
) -> None:
    """Run single-view camera control inference."""
    import torch.distributed as dist
    from glob import glob
    from tqdm import tqdm

    from moge.model.v2 import MoGeModel
    from src.general_utils import rank0_log
    from src.data_utils import load_single_view_data
    from diffusers.utils import export_to_video

    # Load depth model for warp rendering
    depth_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device).eval()

    if dist.is_initialized():
        dist.barrier()
    generator = torch.Generator(device=device).manual_seed(seed)

    # Select autocast dtype
    if torch.cuda.is_bf16_supported():
        autocast_dtype = torch.bfloat16
    elif torch.cuda.get_device_capability(device)[0] >= 7:
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None

    # Load scenes
    if os.path.exists(f"{input_path}/image.png"):
        scene_list = [input_path]
    else:
        scene_list = sorted(glob(f"{input_path}/*"))

    rank0_log(f"Processing {len(scene_list)} scenes.")

    for scene_path in tqdm(scene_list):
        scene_name = os.path.basename(scene_path)
        scene_output_path = f"{output_path}/{scene_name}"

        if rank == 0:
            os.makedirs(scene_output_path, exist_ok=True)

        with torch.no_grad():
            meta_data = load_single_view_data(
                cfg=worldstereo.cfg,
                input_path=scene_path,
                output_path=scene_output_path,
                model_type=model_type,
                depth_model=depth_model,
                device=device,
                sp_size=sp_size,
                sp_rank=sp_rank,
            )

            pipeline_kwargs = {k: v for k, v in meta_data.items() if v is not None}
            pipeline_kwargs.update(
                negative_prompt=worldstereo.cfg.get("negative_prompt", ""),
                generator=generator,
                output_type="pt",
                latent_cond_mode=worldstereo.cfg.latent_cond_mode,
            )

            if model_type == "worldstereo-memory-dmd":
                pipeline_kwargs["mode"] = "test"
            else:
                pipeline_kwargs["guidance_scale"] = 5.0

            with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                output = worldstereo.pipeline(**pipeline_kwargs).frames[0].float()
            output = output.cpu().permute(0, 2, 3, 1).numpy()
            torch.cuda.empty_cache()

        if rank == 0:
            export_to_video(output, f"{scene_output_path}/{model_type}_result.mp4", fps=16)


def _run_multi_traj_inference(
    worldstereo,
    cfg: DictConfig,
    input_path: str,
    output_path: str,
    model_type: str,
    task_type: str,
    device: torch.device,
    seed: int,
    sp_size: int,
    sp_rank: int,
    rank: int,
) -> None:
    """Run multi-trajectory inference (panorama or reconstruction)."""
    import json
    import numpy as np
    import torch.distributed as dist
    from glob import glob
    from tqdm import tqdm
    import imagesize

    from src.general_utils import rank0_log, load_video
    from src.data_utils import sort_trajs, recon_sort_trajs, load_mutli_traj_dataset
    from src.retrieval_wm import SimpleMemoryBank
    from diffusers.utils import export_to_video

    if dist.is_initialized():
        dist.barrier()
    generator = torch.Generator(device=device).manual_seed(seed)
    align_nframe = cfg.get("align_nframe", 8)

    # Select autocast dtype
    if torch.cuda.is_bf16_supported():
        autocast_dtype = torch.bfloat16
    elif torch.cuda.get_device_capability(device)[0] >= 7:
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None

    # Load scenes
    if os.path.exists(f"{input_path}/pano_bank") or os.path.exists(f"{input_path}/image.png"):
        scene_list = [input_path]
    else:
        scene_list = sorted(glob(f"{input_path}/*"))

    rank0_log(f"Processing {len(scene_list)} scenes.")

    for scene_path in tqdm(scene_list):
        scene_name = os.path.basename(scene_path)
        scene_output_path = f"{output_path}/{scene_name}"

        if rank == 0:
            os.makedirs(scene_output_path, exist_ok=True)

        # Get image dimensions
        if task_type == "panorama":
            width, height = imagesize.get(f"{scene_path}/pano_bank/images/0000.png")
        else:
            width, height = imagesize.get(f"{scene_path}/image.png")

        with torch.no_grad():
            memory_bank = SimpleMemoryBank(
                cfg=worldstereo.cfg,
                root_path=scene_path,
                image_width=width,
                image_height=height,
                device=device,
                max_reference=8,
                align_nframe=align_nframe,
                rank=sp_rank,
                world_size=sp_size,
            )

            if task_type == "panorama":
                render_list = sort_trajs(scene_path)
            else:
                render_list = recon_sort_trajs(scene_path)

            rank0_log(f"Scene {scene_name}: {len(render_list)} renderings found.")

        for render_path in render_list:
            view_id, traj_id = render_path.split('/')[-3], render_path.split('/')[-2]
            if task_type == "reconstruction":
                view_id = "renders"

            target_cameras = json.load(open(f"{scene_path}/{view_id}/{traj_id}/camera.json"))
            tar_w2cs = torch.from_numpy(np.array(target_cameras["extrinsic"])).to(
                dtype=torch.float32, device=device
            )
            tar_Ks = torch.from_numpy(np.array(target_cameras["intrinsic"])).to(
                dtype=torch.float32, device=device
            )

            result_path = f"{scene_output_path}/{view_id}/{traj_id}/{model_type}_result.mp4"
            if os.path.exists(result_path):
                gen_frames = load_video(result_path)
                memory_bank.update_memory(
                    gen_frames=gen_frames,
                    tar_w2cs_full=tar_w2cs,
                    tar_Ks_full=tar_Ks,
                    view_id=view_id,
                    traj_id=traj_id,
                )
                continue

            # Retrieval
            retrieved_frames, ref_index, ref_index_dict, ref_w2cs, _ = memory_bank.retrieval(
                tar_w2cs, tar_Ks, view_id=view_id, traj_id=traj_id
            )

            if rank == 0:
                os.makedirs(f"{scene_output_path}/{view_id}/{traj_id}/memory_inputs", exist_ok=True)
                export_to_video(
                    retrieved_frames / 255,
                    f"{scene_output_path}/{view_id}/{traj_id}/memory_inputs/{model_type}.mp4",
                    fps=16,
                )
                if ref_index_dict is not None:
                    with open(
                        f"{scene_output_path}/{view_id}/{traj_id}/memory_inputs/{model_type}_ref_index.json",
                        "w",
                    ) as w:
                        json.dump(ref_index_dict, w, indent=2)
                if ref_w2cs is not None:
                    ref_w2cs_list = ref_w2cs.cpu().numpy().tolist()
                    with open(
                        f"{scene_output_path}/{view_id}/{traj_id}/memory_inputs/{model_type}_ref_w2cs.json",
                        "w",
                    ) as w:
                        json.dump(ref_w2cs_list, w, indent=2)

            if dist.is_initialized():
                dist.barrier()

            # Prepare inputs
            meta_data = load_mutli_traj_dataset(
                cfg=worldstereo.cfg,
                input_path=scene_path,
                output_path=scene_output_path,
                view_id=view_id,
                traj_id=traj_id,
                device=device,
                ref_index=ref_index,
                model_type=model_type,
                task_type=task_type,
            )

            pipeline_kwargs = {k: v for k, v in meta_data.items() if v is not None}
            pipeline_kwargs.update(
                negative_prompt=worldstereo.cfg.get("negative_prompt", ""),
                generator=generator,
                output_type="pt",
                latent_cond_mode=worldstereo.cfg.latent_cond_mode,
            )

            if model_type == "worldstereo-memory-dmd":
                pipeline_kwargs["mode"] = "test"
            else:
                pipeline_kwargs["guidance_scale"] = 5.0

            with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                output = worldstereo.pipeline(**pipeline_kwargs).frames[0].float()
            output = output.cpu().permute(0, 2, 3, 1).numpy()
            torch.cuda.empty_cache()

            if rank == 0:
                os.makedirs(f"{scene_output_path}/{view_id}/{traj_id}", exist_ok=True)
                export_to_video(output, result_path, fps=16)

            if dist.is_initialized():
                dist.barrier()

            # Update memory bank
            gen_frames = load_video(result_path)
            memory_bank.update_memory(
                gen_frames=gen_frames,
                tar_w2cs_full=tar_w2cs,
                tar_Ks_full=tar_Ks,
                view_id=view_id,
                traj_id=traj_id,
            )

            if dist.is_initialized():
                dist.barrier()

        # Convert to WorldMirror format
        memory_bank.apply_worldmirror(
            f"{scene_output_path}/world_mirror_data/{model_type}", skip_exist=True
        )


def run_training(cfg: DictConfig) -> None:
    """
    Run WorldStereo training with Hydra config.

    This is the unified training entry point, equivalent to FastWAM's training
    runner.

    Args:
        cfg: Hydra DictConfig containing all training parameters.
    """
    setup_logging()
    logger.info("Starting WorldStereo training...")

    # Instantiate model
    model = instantiate(cfg.model)
    logger.info(f"Model instantiated: {type(model).__name__}")

    # Instantiate datasets
    train_dataset = instantiate(cfg.data.train)
    val_dataset = instantiate(cfg.data.val) if cfg.data.get("val") else None
    logger.info(f"Train dataset size: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Val dataset size: {len(val_dataset)}")

    # Create trainer and run
    from .trainer import WorldStereoTrainer

    trainer = WorldStereoTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        cfg=cfg,
    )
    trainer.train()
    logger.info("Training complete.")
