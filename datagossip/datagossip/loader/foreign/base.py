import torch
import numpy as np
from typing import Iterator


class ForeignDataIterator(Iterator):
    def __init__(self, data: torch.Tensor, targets: torch.Tensor, args):
        rank = args.rank - 1
        size = args.size
        all_ranks = list(range(size-1))
        self.other_ranks = all_ranks[:rank]+all_ranks[rank+1:]

        self.data = data
        self.targets = targets
        self.indices = np.arange(self.targets.shape[1] * len(self.other_ranks), dtype=np.int)
        np.random.shuffle(self.indices)
        self.batch_size = args.batch_size
        self.idx_pointer = 0
        self.received = False

    def _increase_pointer(self):
        self.idx_pointer += self.batch_size

    def _check_idx_pointer(self):
        if len(self.indices) <= self.idx_pointer:
            self.idx_pointer = 0
            raise StopIteration()

    def reset(self):
        self.idx_pointer = 0

    def __next__(self):
        self._check_idx_pointer()
        indices = self.indices[self.idx_pointer:self.idx_pointer+self.batch_size]
        data = self.data[self.other_ranks].reshape(-1, *self.data.shape[2:])[indices]
        targets = self.targets[self.other_ranks].flatten()[indices]

        self._increase_pointer()
        return data, targets

    def __len__(self):
        return int(len(self.targets[self.other_ranks].flatten()) / self.batch_size)

    @property
    def ready(self):
        if self.received:
            return True

        if self.targets[self.targets == -1].sum() == 0:
            self.received = True

        return self.received
