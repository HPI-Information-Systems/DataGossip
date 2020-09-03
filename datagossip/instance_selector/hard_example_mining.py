import torch

from .base import InstanceSelector


class HardExampleMining(InstanceSelector):
    def select(self, data: torch.Tensor, targets:torch.Tensor, k: int):
        descending = not self.negative
        indices = self.memory.view(-1).argsort(descending=descending)[:k]
        self._fill_tensors(indices, data, targets)
