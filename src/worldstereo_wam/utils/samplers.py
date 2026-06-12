"""Custom samplers for distributed training."""

from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset, Sampler


class ResumableEpochSampler(Sampler[int]):
    """
    A sampler that supports resuming from a specific position within an epoch.

    This sampler ensures deterministic shuffling across epochs and can resume
    from any position when training is interrupted.
    """

    def __init__(
        self,
        dataset: Dataset,
        seed: int = 0,
        batch_size: int = 1,
        num_processes: int = 1,
        start_index: int = 0,
    ):
        """
        Args:
            dataset: Dataset to sample from.
            seed: Random seed for reproducible shuffling.
            batch_size: Batch size (for calculating effective samples per process).
            num_processes: Number of distributed processes.
            start_index: Index to start sampling from (for resuming).
        """
        self.dataset = dataset
        self.seed = seed
        self.batch_size = batch_size
        self.num_processes = num_processes
        self.start_index = start_index
        self.epoch = 0

        self.num_samples = len(dataset)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic shuffling."""
        self.epoch = epoch

    def set_start_index(self, start_index: int) -> None:
        """Set the starting index for resuming."""
        self.start_index = start_index

    def __iter__(self) -> Iterator[int]:
        """Generate indices for the current epoch."""
        # Create a generator with epoch-based seed for reproducibility
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Generate shuffled indices
        indices = torch.randperm(self.num_samples, generator=g).tolist()

        # Resume from start_index
        indices = indices[self.start_index:]

        # Reset start_index for next epoch
        self.start_index = 0

        return iter(indices)

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.num_samples - self.start_index
