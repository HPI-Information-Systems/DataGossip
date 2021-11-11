from typing import Type, List
from torch.utils.data import DataLoader
from .base import DataGossipLoader
from .foreign import ForeignDataIterator
from ...instance_selector import InstanceSelector


class DataGossipCycleLoader(DataGossipLoader):
    def __init__(self,
                 data_loader: DataLoader,
                 instance_selector_class: Type[InstanceSelector],
                 data_shape: List[int],
                 args,
                 foreign_data_loader: Type[ForeignDataIterator] = ForeignDataIterator):
        super(DataGossipCycleLoader, self).__init__(data_loader, instance_selector_class, data_shape, args, foreign_data_loader)

        self.foreign_every = args.remote_train_frequency

    def __iter__(self):
        self.foreign_data_loader.reset()

        for i, (data, targets, indices) in enumerate(self.local_data_loader):
            self.instance_selector.add_index(indices)
            if i % self.n_gather == 0:
                self._instance_selection()
                self._gossip()
            if i > 0 and i % self.foreign_every == 0 and self.foreign_data_loader.ready:
                self.instance_selector.foreign_training()
                yield next(self.foreign_data_loader)

            self.instance_selector.local_training()
            yield data, targets

        self.instance_selector.increment_epoch()
