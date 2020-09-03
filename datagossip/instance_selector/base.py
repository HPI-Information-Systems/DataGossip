import torch
from torch import Tensor
import torch.multiprocessing as mp
from ctypes import c_int


def neg_normalize(t: torch.Tensor):
    return 1 - (t / (t.max()))


class InstanceSelector:
    def __init__(self, dataset_size: int, local_data: torch.Tensor, local_targets: torch.Tensor, **kwargs):
        self.local_data = local_data
        self.local_targets = local_targets
        self.memory = torch.zeros(dataset_size, self._set_sequence_size()) + float('NaN')
        self.memory.share_memory_()
        self.deselected_ids = []
        self.processed_ids = []
        self.local_training_ = True
        self.epoch = mp.Value(c_int, 0)
        self.negative = False

    def negate(self):
        self.negative = not self.negative

    def local_training(self):
        self.local_training_ = True

    def foreign_training(self):
        self.local_training_ = False

    def _set_sequence_size(self) -> int:
        return 1

    def _fill_tensors(self, indices: torch.Tensor, data: torch.Tensor, targets: torch.Tensor):
        data[:] = self.local_data[indices]
        targets[:] = self.local_targets[indices]

    def select(self, data: torch.Tensor, targets: torch.Tensor, k: int):
        raise NotImplementedError()

    def update(self, loss: Tensor):
        self.memory[self.processed_ids, 0] = loss
        self.processed_ids = []

    def add_index(self, idx: torch.Tensor):
        self.processed_ids += idx.tolist()

    def increment_epoch(self):
        with self.epoch.get_lock():
            self.epoch.value += 1
