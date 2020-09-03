import torch
from torch import Tensor
import numpy as np
from .base import InstanceSelector


class Random(InstanceSelector):
    def select(self, data: torch.Tensor, targets: torch.Tensor, k: int):
        indices = torch.from_numpy(np.random.choice(np.arange(self.memory.shape[0]), k, replace=False))
        self._fill_tensors(indices, data, targets)
