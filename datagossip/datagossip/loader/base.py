import torch
from torch.utils.data import DataLoader
from typing import List, Iterator, Type

from .foreign import ForeignDataIterator
from ..process import DataGossipProcess
from ...instance_selector import InstanceSelector, InstanceSelectorChooser


class DataGossipLoader:
    def __init__(self,
                 data_loader: DataLoader,
                 instance_selector: InstanceSelectorChooser,
                 data_shape: List[int],
                 args,
                 foreign_data_loader: Type[ForeignDataIterator] = ForeignDataIterator):
        self.local_data_loader = data_loader

        # foreign data
        n_neighbors = args.size # group size
        self.foreign_data = torch.zeros(n_neighbors - 1, args.k, *data_shape) - 1
        self.foreign_targets = torch.zeros(n_neighbors - 1, args.k).long() - 1
        self.foreign_data.share_memory_()
        self.foreign_targets.share_memory_()
        self.foreign_data_loader = foreign_data_loader(self.foreign_data, self.foreign_targets, args)
        foreign_batches = int(((n_neighbors - 1) * args.k) / args.batch_size)
        self.foreign_every = int(len(self.local_data_loader) / foreign_batches)
        self.n_gather = int(len(self.local_data_loader) / args.n_gather)

        local_dataset = data_loader.dataset.tensors

        self.instance_selector: InstanceSelector = instance_selector.get_instance_selector_class()(len(data_loader.dataset), local_dataset[0], local_dataset[1], epochs=args.epochs)

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
