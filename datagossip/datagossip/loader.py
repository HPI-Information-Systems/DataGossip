import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from typing import Iterator, List
import numpy as np

from .process import DataGossipProcess
from ..instance_selector.base import InstanceSelector


class ForeignDataLoader(Iterator):
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


class DataGossipLoader:
    def __init__(self,
                 data_loader: DataLoader,
                 instance_selector_class: InstanceSelector.__class__,
                 data_shape: List[int],
                 args):
        self.local_data_loader = data_loader

        # foreign data
        n_neighbors = args.size # group size
        self.foreign_data = torch.zeros(n_neighbors - 1, args.k, *data_shape) - 1
        self.foreign_targets = torch.zeros(n_neighbors - 1, args.k).long() - 1
        self.foreign_data.share_memory_()
        self.foreign_targets.share_memory_()
        self.foreign_data_loader = ForeignDataLoader(self.foreign_data, self.foreign_targets, args)
        foreign_batches = int(((n_neighbors - 1) * args.k) / args.batch_size)
        self.foreign_every = int(len(self.local_data_loader) / foreign_batches)
        self.n_gather = int(len(self.local_data_loader) / args.n_gather)

        local_dataset = data_loader.dataset.tensors

        self.instance_selector: InstanceSelector = instance_selector_class(len(data_loader.dataset), local_dataset[0], local_dataset[1], epochs=args.epochs)

        self.datagossip_process = DataGossipProcess(self.foreign_data,
                                                    self.foreign_targets,
                                                    self.instance_selector,
                                                    args)
        self.datagossip_process.start()

    def stop(self):
        self.datagossip_process.stop()

    def _instance_selection(self):
        self.datagossip_process.run_instance_selection()

    def _gossip(self):
        self.datagossip_process.run_all_gather()

    def __iter__(self):
        self.foreign_data_loader.reset()
        foreign_active_iteration = True

        for i, (data, targets, indices) in enumerate(self.local_data_loader):
            self.instance_selector.add_index(indices)
            if i % self.n_gather == 0:
                self._instance_selection()
                self._gossip()
            if foreign_active_iteration and i > 0 and i % self.foreign_every == 0 and self.foreign_data_loader.ready:
                try:
                    self.instance_selector.foreign_training()
                    yield next(self.foreign_data_loader)
                except StopIteration:
                    foreign_active_iteration = False
            self.instance_selector.local_training()
            yield data, targets

        self.instance_selector.increment_epoch()

    def __len__(self):
        return len(self.local_data_loader) + len(self.foreign_data_loader)
