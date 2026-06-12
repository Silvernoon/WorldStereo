"""
WorldStereo WAM Trainer.

Provides a unified training interface similar to FastWAM's Wan22Trainer.
"""

from __future__ import annotations

import logging
import json
import os
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler

logger = get_logger(__name__)


class WorldStereoTrainer:
    """
    Trainer class for WorldStereo models.

    Mirrors FastWAM's Wan22Trainer structure with Accelerate/DeepSpeed support,
    gradient accumulation, mixed precision training, and checkpoint management.
    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        *,
        cfg: DictConfig,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg

        # Extract training config
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.get("max_steps")
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.get("eval_num_inference_steps", 10))
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)

        self.resume = cfg.get("resume")
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled) if cfg.get("wandb") else False

        # Initialize Accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )

        logger.info(
            "Accelerate training: distributed_type=%s world_size=%d process_index=%d "
            "cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("Using accelerator.device=%s", self.accelerator.device)

        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)

        # Set up trainable parameters
        trainable_params = self._get_trainable_params()
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )

        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )

        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        # Set up checkpoint directories
        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        # Prepare with Accelerate
        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)

        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _get_trainable_params(self):
        """Get list of trainable parameters from model."""
        trainable_params = []

        # Check for different model components
        if hasattr(self.model, "transformer"):
            trainable_params.extend(list(self.model.transformer.parameters()))
        elif hasattr(self.model, "dit"):
            trainable_params.extend(list(self.model.dit.parameters()))

        if hasattr(self.model, "controlnet"):
            trainable_params.extend(list(self.model.controlnet.parameters()))

        # If no specific components found, train all parameters
        if not trainable_params:
            trainable_params = list(self.model.parameters())
            logger.warning("No specific trainable components found, training all parameters.")

        return trainable_params

    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.get("workspace"),
            project=self.cfg.wandb.get("project", "worldstereo-wam"),
            name=self.cfg.wandb.get("name", "train"),
            group=None if self.cfg.wandb.get("group") in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.get("mode", "online"),
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: project=%s name=%s",
            self.cfg.wandb.get("project"),
            self.cfg.wandb.get("name"),
        )

    def _wandb_log(self, payload: dict):
        """Log metrics to wandb."""
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        """Finish wandb run."""
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        """Build data loader with resumable sampler."""
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )

    def _estimate_total_train_steps(self) -> int:
        """Estimate total training steps."""
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        """Build learning rate scheduler."""
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps > 0:
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_steps],
            )
        else:
            scheduler = main_scheduler

        return scheduler

    def _resume_or_load_checkpoint(self):
        """Resume training from checkpoint if specified."""
        if self.resume is None:
            return

        checkpoint_path = self.resume
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint path {checkpoint_path} does not exist, starting from scratch.")
            return

        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        self.accelerator.load_state(checkpoint_path)

        # Load training state
        state_file = os.path.join(checkpoint_path, "training_state.json")
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                state = json.load(f)
            self.global_step = state.get("global_step", 0)
            self.epoch = state.get("epoch", 0)
            self.batch_in_epoch = state.get("batch_in_epoch", 0)
            logger.info(f"Resumed at step {self.global_step}, epoch {self.epoch}")

    def _save_checkpoint(self, checkpoint_name: str = None):
        """Save training checkpoint."""
        if checkpoint_name is None:
            checkpoint_name = f"step_{self.global_step}"

        checkpoint_path = os.path.join(self.state_dir, checkpoint_name)
        self.accelerator.save_state(checkpoint_path)

        # Save training state
        if self.accelerator.is_main_process:
            state = {
                "global_step": self.global_step,
                "epoch": self.epoch,
                "batch_in_epoch": self.batch_in_epoch,
            }
            with open(os.path.join(checkpoint_path, "training_state.json"), "w") as f:
                json.dump(state, f)

        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def train(self):
        """Main training loop."""
        logger.info(f"Starting training for {self.max_steps} steps")

        self.model.train()
        start_time = time.time()

        for epoch in range(self.epoch, self.num_epochs):
            self.epoch = epoch
            self.train_sampler.set_epoch(epoch)

            for batch_idx, batch in enumerate(self.train_loader):
                if batch_idx < self.batch_in_epoch:
                    continue
                self.batch_in_epoch = batch_idx

                with self.accelerator.accumulate(self.model):
                    loss = self._training_step(batch)

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.max_grad_norm,
                        )

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                if self.accelerator.sync_gradients:
                    self.global_step += 1

                    # Logging
                    if self.global_step % self.log_every == 0:
                        elapsed = time.time() - start_time
                        lr = self.scheduler.get_last_lr()[0]
                        logger.info(
                            f"Step {self.global_step}/{self.max_steps} | "
                            f"Loss: {loss.item():.4f} | LR: {lr:.2e} | "
                            f"Time: {elapsed:.1f}s"
                        )
                        self._wandb_log({
                            "train/loss": loss.item(),
                            "train/lr": lr,
                            "train/epoch": epoch,
                        })

                    # Evaluation
                    if self.global_step % self.eval_every == 0 and self.val_dataset is not None:
                        self._evaluate()

                    # Checkpointing
                    if self.global_step % self.save_every == 0:
                        self._save_checkpoint()

                    if self.global_step >= self.max_steps:
                        break

            self.batch_in_epoch = 0

            if self.global_step >= self.max_steps:
                break

        # Final checkpoint
        self._save_checkpoint("final")
        self._finish_wandb()
        logger.info("Training completed.")

    def _training_step(self, batch) -> torch.Tensor:
        """Single training step. Override this for custom training logic."""
        # Default: assume model returns dict with 'loss' key or just loss tensor
        outputs = self.model(**batch)

        if isinstance(outputs, dict):
            loss = outputs.get("loss", outputs.get("total_loss"))
        elif isinstance(outputs, torch.Tensor):
            loss = outputs
        else:
            raise ValueError(f"Unexpected model output type: {type(outputs)}")

        return loss

    def _evaluate(self):
        """Run evaluation on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                outputs = self.model(**batch)

                if isinstance(outputs, dict):
                    loss = outputs.get("loss", outputs.get("total_loss", 0))
                else:
                    loss = outputs

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        logger.info(f"Evaluation at step {self.global_step}: avg_loss={avg_loss:.4f}")
        self._wandb_log({"eval/loss": avg_loss})

        self.model.train()
