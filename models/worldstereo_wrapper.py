"""
WorldStereo unified inference class.

Bundles all sub-models (transformer, text/image encoders, VAE) and the
matching inference pipeline under a single diffusers-style interface::

    worldstereo = WorldStereo.from_pretrained(
        "/path/to/checkpoint_root",
        device=device,
    )
    output = worldstereo(**pipeline_inputs)

Hugging Face format expects ``config.json`` plus ``model.safetensors``
in the same directory.

The config must include a ``model_type`` field with one of the
supported values:

* ``worldstereo-camera``      – keyframe + camera control
* ``worldstereo-memory``      – keyframe + camera control + GGM + SSM
* ``worldstereo-memory-dmd``  – DMD (distribution matching distillation) mode
"""

from __future__ import annotations

import gc
import json
import os
import types
from typing import Any

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DIFFUSERS_VERBOSITY"] = "error"

import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler
from omegaconf import OmegaConf
from safetensors.torch import load_file as load_safetensors
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from models.attention import WanAttnProcessorSP
from models.dmd_scheduler import FlowGeneratorScheduler
from models.pipelines.pipeline_dmd_keyframe import RefKFDMDGeneratorPipeline
from models.pipelines.pipeline_pcd_keyframe import KFPCDControllerPipeline
from models.pipelines.pipeline_ref_keyframe import KFPCDControllerRefPipeline
from models.worldstereo import WorldStereoModel, WorldStereoRefSModel
from src.general_utils import rank0_log

# ── suppress noisy third-party logs ───────────────────────────────────
import logging
import warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DIFFUSERS_VERBOSITY"] = "error"

# transformers / diffusers print a wall of "Some weights were not
# initialized / unexpected keys" on every load.  We already inspect
# load_state_dict results ourselves in worldstereo_wrapper.py, so
# silence their own reporting.
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("diffusers.modeling_utils").setLevel(logging.ERROR)

# huggingface_hub HTTP request logs (newer versions use httpx as the HTTP client)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("filelock").setLevel(logging.ERROR)

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

from diffusers.utils import logging as diffusers_logging
diffusers_logging.set_verbosity_error()

# torch.compile / inductor verbose output
logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
logging.getLogger("torch._inductor").setLevel(logging.WARNING)

# misc deprecation / user warnings from HF internals
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="diffusers")
# ──────────────────────────────────────────────────────────────────────

SUPPORTED_MODEL_TYPES = ("worldstereo-camera", "worldstereo-memory", "worldstereo-memory-dmd")


def _get_half_dtype() -> torch.dtype:
    """Select the best half-precision dtype based on current GPU capability: bf16 > fp16 > fp32."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    elif torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        return torch.float16
    else:
        return torch.float32


class WorldStereo:
    """Diffusers-style wrapper that owns every sub-model and its pipeline."""

    def __init__(self, pipeline: Any, cfg: Any) -> None:
        self.pipeline = pipeline
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        subfolder: str = "",
        local_files_only: bool = False,
        sp_world_size: int = 1,
        fsdp: bool = False,
        device_mesh=None,
        device: torch.device | None = None,
        quantize_w8a8: bool = False,
        quantize_transformer_only: bool = True,
        w8a8_save_path: str | None = None,
    ) -> "WorldStereo":
        """
        Build a WorldStereo instance from Hugging Face format
        (``config.json`` + ``model.safetensors``).

        Args:
            repo_id: Model directory or HF repo ID.
            subfolder: Subfolder within the HF repo or local directory. This is equivalent to the `model_type` (e.g., 'worldstereo-camera').
            local_files_only: If True, avoid downloading the file and return the path to the local cached file if it exists.
            sp_world_size: Sequence-Parallel degree (1 = disabled).
            fsdp: Wrap models with PyTorch FSDP.  Requires ``device_mesh``.
            device_mesh: ``DeviceMesh`` with dims ``("rep", "shard")``.
            device: Target CUDA device.
            quantize_w8a8: Apply W8A8 (INT8 weight + INT8 dynamic activation) quantization to the
                transformer (and optionally aux models) via ``torchao``.  Requires
                ``pip install torchao`` and a CUDA GPU with SM ≥ 80 (e.g. A100 / RTX 3090/4090).
                Reduces VRAM by ~50 % with minimal quality loss.
            quantize_transformer_only: When ``quantize_w8a8=True``, only quantize the DiT
                transformer (default).  Set to ``False`` to also quantize the VAE and text/image
                encoders – further saves VRAM but may slightly affect quality.
            w8a8_save_path: Optional file path (e.g. ``"quantized/transformer_w8a8.pt"``).
                • If the file **does not exist** and ``quantize_w8a8=True``, the quantized
                  transformer is saved there after quantization (rank-0 only).
                • If the file **already exists**, it is loaded directly, skipping re-quantization
                  (much faster cold start).  ``quantize_w8a8`` must still be ``True`` so the
                  model architecture is prepared correctly before loading.
        """
        if os.path.isdir(repo_id):
            json_cfg_path = os.path.join(repo_id, subfolder, "config.json")
            safetensors_path = os.path.join(repo_id, subfolder, "model.safetensors")

            if not os.path.exists(json_cfg_path):
                raise FileNotFoundError(f"config.json not found under {json_cfg_path!r}")
            if not os.path.exists(safetensors_path):
                raise FileNotFoundError(f"model.safetensors not found at {safetensors_path!r}")
        else:
            from huggingface_hub import hf_hub_download
            json_cfg_path = hf_hub_download(
                repo_id=repo_id,
                filename="config.json",
                subfolder=subfolder if subfolder else None,
                local_files_only=local_files_only,
            )
            safetensors_path = hf_hub_download(
                repo_id=repo_id,
                filename="model.safetensors",
                subfolder=subfolder if subfolder else None,
                local_files_only=local_files_only,
            )

        cfg = OmegaConf.create(cls._load_hf_config(json_cfg_path))
        model_weights_path = safetensors_path

        model_type = subfolder
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type {model_type!r}. "
                f"Expected one of {SUPPORTED_MODEL_TYPES}."
            )

        transformer = cls._load_transformer(
            cfg,
            model_type,
            model_weights_path,
            sp_world_size=sp_world_size,
            fsdp=fsdp,
            device_mesh=device_mesh,
            device=device,
            quantize_w8a8=quantize_w8a8,
            w8a8_save_path=w8a8_save_path,
        )

        text_encoder, image_clip, vae = cls._load_aux(
            cfg, device=device, device_mesh=device_mesh, fsdp=fsdp, local_files_only=local_files_only
        )

        # Optionally extend W8A8 quantization to aux models (VAE + encoders)
        if quantize_w8a8 and not quantize_transformer_only:
            cls._apply_w8a8(text_encoder, label="text_encoder")
            cls._apply_w8a8(image_clip, label="image_clip")
            cls._apply_w8a8(vae, label="vae")
        image_processor = CLIPImageProcessor.from_pretrained(
            cfg.base_model, do_rescale=False, subfolder="image_processor", local_files_only=local_files_only
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, subfolder="tokenizer", local_files_only=local_files_only)

        pipeline = cls._build_pipeline(
            model_type,
            cfg,
            transformer=transformer,
            text_encoder=text_encoder,
            image_clip=image_clip,
            image_processor=image_processor,
            tokenizer=tokenizer,
            vae=vae,
            device=device,
            local_files_only=local_files_only,
        )

        rank0_log(f"WorldStereo ({model_type}) ready.")
        return cls(pipeline=pipeline, cfg=cfg)

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward all arguments to the underlying inference pipeline."""
        return self.pipeline(*args, **kwargs)

    def to(self, device: torch.device) -> "WorldStereo":
        self.pipeline = self.pipeline.to(device)
        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_hf_config(config_json_path: str) -> dict[str, Any]:
        with open(config_json_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        required_keys = ["base_model", "controlnet_cfg"]
        missing = [k for k in required_keys if k not in cfg]
        if missing:
            raise ValueError(
                f"config.json missing required keys: {missing}. "
                "Please use the conversion script to export a valid HF package."
            )

        return cfg

    # ------------------------------------------------------------------
    # W8A8 quantization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_w8a8(model: torch.nn.Module, label: str = "model") -> None:
        """Apply torchao W8A8 (INT8 dynamic activation + INT8 weight) quantization in-place.

        Only ``nn.Linear`` layers with both input and output features ≥ 16 are
        quantized; smaller projections (e.g. norm biases, tiny heads) are skipped
        automatically by torchao.

        Requires ``pip install torchao`` and CUDA SM ≥ 80.
        """
        try:
            from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight
        except ImportError as exc:
            raise ImportError(
                "W8A8 quantization requires torchao.  "
                "Install it with:  pip install torchao"
            ) from exc

        rank0_log(f"Applying W8A8 quantization to {label}…")
        quantize_(model, int8_dynamic_activation_int8_weight())
        rank0_log(f"W8A8 quantization done for {label}.")

    @staticmethod
    def save_w8a8(
        model: torch.nn.Module,
        save_path: str,
        *,
        rank: int = 0,
    ) -> None:
        """Serialize a torchao-quantized model to ``save_path`` (a ``.pt`` file).

        Only rank-0 writes to disk in distributed settings.

        Args:
            model: A quantized ``nn.Module`` (output of ``_apply_w8a8``).
            save_path: Destination file path, e.g. ``"quantized/transformer_w8a8.pt"``.
            rank: Current distributed rank; only rank 0 writes.
        """
        try:
            from torchao.quantization import quantized_decomposed  # noqa: F401 – registers ops
            from torchao._torchao_C import torchao as _  # noqa: F401
        except ImportError:
            pass  # older torchao versions don't need this

        if rank != 0:
            return

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        rank0_log(f"Saving W8A8 quantized model to {save_path}…")
        torch.save(model.state_dict(), save_path)
        rank0_log(f"Saved → {save_path}")

    @staticmethod
    def _load_w8a8_checkpoint(
        model: torch.nn.Module,
        load_path: str,
        device: torch.device,
    ) -> torch.nn.Module:
        """Reload a previously saved W8A8 state-dict back into a fresh *quantized* model.

        The model must already have been quantized with ``_apply_w8a8`` before
        calling this (so the parameter layout matches the saved state-dict).

        Args:
            model: Already-quantized model whose weights will be overwritten.
            load_path: Path to the ``.pt`` file written by :meth:`save_w8a8`.
            device: Target device.

        Returns:
            The same ``model`` with weights loaded from disk.
        """
        rank0_log(f"Loading W8A8 checkpoint from {load_path}…")
        state_dict = torch.load(load_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        rank0_log("W8A8 checkpoint loaded.")
        return model

    @staticmethod
    def _load_transformer(
        cfg,
        model_type: str,
        weights_path: str,
        *,
        sp_world_size: int,
        fsdp: bool,
        device_mesh,
        device,
        quantize_w8a8: bool = False,
        w8a8_save_path: str | None = None,
    ):

        half_dtype = _get_half_dtype()
        rank0_log(f"Loading transformer ({model_type})… dtype={half_dtype}")

        if model_type == "worldstereo-camera":
            transformer = WorldStereoModel.from_pretrained(
                cfg.base_model,
                subfolder="transformer",
                controlnet_cfg=cfg.controlnet_cfg,
                torch_dtype=half_dtype,
            )
        else:
            transformer = WorldStereoRefSModel.from_pretrained(
                cfg.base_model,
                subfolder="transformer",
                controlnet_cfg=cfg.controlnet_cfg,
                torch_dtype=half_dtype,
            )

        rank0_log("Building ControlNet…")
        transformer.build_controlnet(load_uni3c=False, freeze_backbone=cfg.freeze_backbone)

        if sp_world_size > 1:
            transformer.sp_size = sp_world_size
            for layer in transformer.controlnet.controlnet_blocks:
                layer.self_attn.processor.sp_size = sp_world_size
            for block in transformer.blocks:
                if model_type == "worldstereo-camera":
                    block.attn1.set_processor(WanAttnProcessorSP(sp_size=sp_world_size))
                else:
                    block.attn1.processor.sp_size = sp_world_size

        rank0_log(f"Loading HF safetensors weights from {weights_path}…")
        weights = load_safetensors(weights_path, device="cpu")

        result = transformer.load_state_dict(weights, strict=False)

        def _summarize_keys(keys: list[str], label: str) -> None:
            if not keys:
                return
            from collections import Counter
            # Count unloaded parameters
            total_params = sum(
                transformer.state_dict()[k].numel()
                for k in keys
                if k in transformer.state_dict()
            )
            # Count occurrence frequency of each field (split by ".") across all keys, take top-2
            field_counter: Counter[str] = Counter()
            for k in keys:
                parts = k.split(".")
                # Skip pure numeric indices (e.g. blocks.0) and common prefixes/suffixes
                field_counter.update(p for p in parts if not p.isdigit())
            top_fields = [f for f, _ in field_counter.most_common(2)]
            # Filter representative keys using top-2 fields (prefer keys that contain both fields)
            repr_keys = sorted([k for k in keys if all(f in k.split(".") for f in top_fields)])
            if not repr_keys:
                repr_keys = sorted(keys)
            sample_keys = repr_keys[:3]
            rank0_log(
                f"{label}: {len(keys)} keys ({total_params / 1e6:.1f}M params), "
                f"top fields: {top_fields}. "
                f"Representative: {sample_keys}"
                + (f" … and {len(keys) - len(sample_keys)} more" if len(keys) > len(sample_keys) else "")
            )
            rank0_log(f"These are frozen backbone weights initialized by the base video model ({cfg.base_model}).")

        _summarize_keys(result.unexpected_keys, "Unexpected keys")
        _summarize_keys(result.missing_keys, "Missing keys")

        if fsdp:
            if quantize_w8a8:
                rank0_log("W8A8 quantization is not compatible with FSDP – skipping quantization.", "WARNING")
            fsdp_kwargs = dict(
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=half_dtype,
                    reduce_dtype=torch.float32,
                ),
                mesh=device_mesh["rep", "shard"],
                reshard_after_forward=True,
            )
            transformer = transformer.to(half_dtype)
            for layer in transformer.blocks:
                fully_shard(layer, **fsdp_kwargs)
            for layer in transformer.controlnet.controlnet_blocks:
                fully_shard(layer, **fsdp_kwargs)
            fully_shard(transformer, **fsdp_kwargs)
            rank0_log("FSDP wrapping done for transformer.")
        else:
            transformer = transformer.to(device=device)
            # ── W8A8 量化（torchao）─────────────────────────────────────
            if quantize_w8a8:
                cached = (
                    w8a8_save_path is not None
                    and os.path.isfile(w8a8_save_path)
                )
                if cached:
                    # Fast path: architecture must be quantized first so layer
                    # types match the saved state-dict, then overwrite weights.
                    WorldStereo._apply_w8a8(transformer, label="transformer (layout)")
                    WorldStereo._load_w8a8_checkpoint(transformer, w8a8_save_path, device)
                else:
                    WorldStereo._apply_w8a8(transformer, label="transformer")
                    if w8a8_save_path is not None:
                        rank = dist.get_rank() if dist.is_initialized() else 0
                        WorldStereo.save_w8a8(transformer, w8a8_save_path, rank=rank)

        gc.collect()
        torch.cuda.empty_cache()
        return transformer.eval()

    @staticmethod
    def _load_aux(cfg, *, device, device_mesh, fsdp: bool, local_files_only: bool = False):
        import transformers as _tr
        from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling

        # ---- text encoder ----
        rank0_log("Loading TextEncoder (UMT5)…")
        text_encoder = UMT5EncoderModel.from_pretrained(
            cfg.base_model, subfolder="text_encoder", torch_dtype=torch.float32, local_files_only=local_files_only
        ).eval()
        if _tr.__version__ >= "5.0.0":
            rank0_log("Patching text_encoder.encoder.embed_tokens for transformers>=5.0.0", "WARNING")
            text_encoder.encoder.embed_tokens = text_encoder.shared
        text_encoder = torch.compile(text_encoder)

        # ---- image encoder ----
        rank0_log("Loading ImageEncoder (CLIP)…")
        image_clip = CLIPVisionModel.from_pretrained(
            cfg.base_model, subfolder="image_encoder", torch_dtype=torch.float32, local_files_only=local_files_only
        ).eval()
        if _tr.__version__ >= "5.0.0":
            rank0_log("Patching CLIP vision forward for transformers>=5.0.0", "WARNING")

            def _clip_vision_forward(self, pixel_values=None, interpolate_pos_encoding=False, **kwargs):
                if pixel_values is None:
                    raise ValueError("pixel_values is required")
                hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
                hidden_states = self.pre_layrnorm(hidden_states)
                encoder_outputs = self.encoder(inputs_embeds=hidden_states, **kwargs)
                pooled_output = self.post_layernorm(encoder_outputs.last_hidden_state[:, 0, :])
                return BaseModelOutputWithPooling(
                    last_hidden_state=encoder_outputs.last_hidden_state,
                    pooler_output=pooled_output,
                    hidden_states=encoder_outputs.hidden_states,
                )

            def _clip_encoder_forward(self, inputs_embeds, attention_mask=None, **kwargs):
                hidden_states = inputs_embeds
                encoder_states = ()
                for layer in self.layers:
                    encoder_states = encoder_states + (hidden_states,)
                    hidden_states = layer(hidden_states, attention_mask, **kwargs)
                encoder_states = encoder_states + (hidden_states,)
                return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states)

            image_clip.vision_model.forward = types.MethodType(_clip_vision_forward, image_clip.vision_model)
            image_clip.vision_model.encoder.forward = types.MethodType(_clip_encoder_forward, image_clip.vision_model.encoder)

        # ---- VAE ----
        vae_dtype = _get_half_dtype()
        rank0_log(f"Loading 3D-VAE… dtype={vae_dtype}")
        vae = AutoencoderKLWan.from_pretrained(
            cfg.base_model, subfolder="vae", torch_dtype=vae_dtype, local_files_only=local_files_only
        ).eval()
        vae = torch.compile(vae)

        if fsdp:
            fsdp_kwargs = dict(
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.float32, reduce_dtype=torch.float32,
                ),
                mesh=device_mesh["rep", "shard"],
                reshard_after_forward=True,
            )
            for layer in text_encoder.encoder.block:
                fully_shard(layer, **fsdp_kwargs)
            fully_shard(text_encoder, **fsdp_kwargs)
            rank0_log("FSDP wrapping done for T5.")

            for layer in image_clip.vision_model.encoder.layers:
                fully_shard(layer, **fsdp_kwargs)
            fully_shard(image_clip, **fsdp_kwargs)
            rank0_log("FSDP wrapping done for CLIP.")

            gc.collect()
            torch.cuda.empty_cache()
        else:
            text_encoder = text_encoder.to(device=device)
            image_clip = image_clip.to(device=device)

        vae = vae.to(device=device)
        return text_encoder, image_clip, vae

    @staticmethod
    def _build_pipeline(
        model_type: str,
        cfg,
        *,
        transformer,
        text_encoder,
        image_clip,
        image_processor,
        tokenizer,
        vae,
        device,
        local_files_only: bool = False,
    ):
        common = dict(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            image_encoder=image_clip,
            image_processor=image_processor,
            transformer=transformer,
            vae=vae,
        )
        if model_type == "worldstereo-camera":
            scheduler = UniPCMultistepScheduler.from_pretrained(
                cfg.base_model, subfolder="scheduler", local_files_only=local_files_only
            )
            return KFPCDControllerPipeline(**common, scheduler=scheduler)

        if model_type == "worldstereo-memory":
            scheduler = UniPCMultistepScheduler.from_pretrained(
                cfg.base_model, subfolder="scheduler", local_files_only=local_files_only
            )
            return KFPCDControllerRefPipeline(**common, scheduler=scheduler)

        if model_type == "worldstereo-memory-dmd":
            scheduler = FlowGeneratorScheduler(
                start_timesteps=cfg.dmd_start_steps,
                num_train_timesteps=cfg.dmd_end_steps,
                shift=cfg.gen_shift,
                use_timestep_transform=True,
                dmd_steps=cfg.dmd_steps,
                rank=dist.get_rank(),
            )
            return RefKFDMDGeneratorPipeline(
                **common,
                scheduler=scheduler,
                device=device,
                vae_compile=False,
                vae_compile_mode="max-autotune",
            )

        raise ValueError(f"Unknown model_type: {model_type!r}")