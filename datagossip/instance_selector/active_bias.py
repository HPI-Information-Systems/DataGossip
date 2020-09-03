import torch
from torch import Tensor
import numpy as np
from logging import Logger, WARNING
from typing import List
from threading import Thread

from .base import InstanceSelector, neg_normalize


logger = Logger("Active Bias", level=WARNING)


class ActiveBias(InstanceSelector):
    def __init__(self, dataset_size: int, local_data: torch.Tensor, local_targets: torch.Tensor, epochs: int = 30):
        self.epochs = epochs
        super().__init__(dataset_size, local_data, local_targets)

    def _set_sequence_size(self) -> int:
        return self.epochs

    def _get_current_epochs(self) -> Tensor:
        result = self.memory.argmin(dim=1)
        return result

    def select(self, data: torch.Tensor, targets:torch.Tensor, k: int):
        history_so_far = torch.from_numpy(np.nanvar(self.memory.numpy(), axis=1))
        # settings NaNs to 0 which results in 0 probability they are getting chosen
        v = (history_so_far + ((history_so_far ** 2) / self._get_current_epochs())).sqrt()
        v = torch.from_numpy(np.nan_to_num(v))
        n_samples = min((v > 0).sum(), k)
        if n_samples > 0:
            if self.negative:
                v = neg_normalize(v)
            indices = v.multinomial(n_samples, False)
            remaining = k - len(indices)
            if remaining > 0:
                random_remainders = np.random.choice(v, remaining, replace=False)
                indices = torch.cat([indices, random_remainders])
            assert len(indices) == k, f"length of indices is {len(indices)}, expected {k}"
        else:
            indices = torch.from_numpy(np.random.choice(np.arange(self.memory.shape[0]), k, replace=False))
        self._fill_tensors(indices, data, targets)

    def update(self, loss: Tensor):
        self.memory[self.processed_ids, self.epoch.value] = loss
        self.processed_ids = []
