import numpy as np
import torch
import torch.distributions as distributions

from .base import InstanceSelector, neg_normalize


class SelfPacedLearning(InstanceSelector):
    def __init__(self, dataset_size: int, local_data: torch.Tensor, local_targets: torch.Tensor, threshold: float = 0.1, growth_factor: float = 1.2, **kwargs):
        super(SelfPacedLearning, self).__init__(dataset_size, local_data, local_targets)
        self.threshold = threshold
        self.growth_factor = growth_factor
        self.check_id = None

    def select(self, data: torch.Tensor, targets: torch.Tensor, k: int):
        flat_memory = self.memory.flatten()
        current_threshold = self.threshold * (self.growth_factor ** self.epoch.value)
        mask = (~torch.isnan(flat_memory)) & (flat_memory < current_threshold)
        if not any(mask):
            indices = torch.from_numpy(np.random.choice(np.arange(self.memory.shape[0]), k, replace=False))
        else:
            v = flat_memory[mask]
            if self.negative:
                v = neg_normalize(v)
            indices = v.multinomial(k, False)
        self._fill_tensors(indices, data, targets)
