import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional
from logging import Logger, WARNING

logger = Logger("DistributedDataLoader", level=WARNING)


class DistributedDataLoader(DataLoader):
    def __init__(self,
                 dataset: Optional[TensorDataset],
                 imbalanced: bool = False,
                 partition: bool = True,
                 parameter_server: bool = True,
                 overlap: int = 0,
                 with_indices: bool = False,
                 **kwargs):

        assert dist.is_initialized(), "Please initialize 'torch.distributed' first!"

        if not partition and overlap > 0:
            logger.warning("A value for overlap > 0 gets ignored when broadcasting (partition == False)")

        self.with_indices = with_indices
        self.imbalanced = imbalanced
        self.dataset = dataset
        self.rank = dist.get_rank()
        self.overlap = overlap

        if partition:
            if parameter_server:
                self.partition_parameter_server(self.rank == 0)
            else:
                self.partition_all(self.rank == 0)
        else:
            if parameter_server:
                self.broadcast_parameter_server(self.rank == 0)
            else:
                self.broadcast_all(self.rank == 0)

        if kwargs.get("sampler", None) is not None:
            sampler_class = kwargs.get("sampler")
            kwargs["sampler"] = sampler_class(self.dataset)

        super().__init__(self.dataset, **kwargs)

    def partition_parameter_server(self, server: bool = False):
        if server:
            self._partition_server([r for r in range(dist.get_world_size()) if r != 0])
        else:
            self._receive_client()

    def partition_all(self, server: bool = False):
        if server:
            self._partition_server(list(range(dist.get_world_size())))
        else:
            self._receive_client()

    def _partition_server(self, dst_ranks: List[int]):
        size = len(dst_ranks)
        if self.imbalanced:
            indices = self.dataset.tensors[1].argsort()
            indices = np.array_split(indices, size)
        else:
            indices = np.arange(len(self.dataset))
            np.random.shuffle(indices)
            indices = np.array_split(indices, size)

        indices = self._add_overlap(indices) if self.overlap > 0 else indices

        for i, rank in enumerate(dst_ranks):
            data = self.dataset.tensors[0][indices[i]]
            targets = self.dataset.tensors[1][indices[i]]
            data_shape = torch.IntTensor([data.shape])

            if rank == self.rank:
                continue

            dist.send(data_shape, dst=rank)
            dist.send(data, dst=rank)
            dist.send(targets, dst=rank)

        if self.rank in dst_ranks:
            self._update_dataset(self.dataset.tensors[0][indices[self.rank]],
                                 self.dataset.tensors[1][indices[self.rank]])

    def _add_overlap(self, indices: List[np.ndarray]) -> List[np.ndarray]:
        overlaps = []
        for rank in range(len(indices)):
            other_indices = np.concatenate([ind for i, ind in enumerate(indices) if i != rank])
            overlap = np.random.choice(other_indices, self.overlap, replace=False)
            overlaps.append(overlap)
        return [np.concatenate([i, o]) for i, o in zip(indices, overlaps)]

    def broadcast_parameter_server(self, server: bool = False):
        if server:
            self._broadcast_server([r for r in range(dist.get_world_size()) if r != 0])
        else:
            self._receive_client()

    def broadcast_all(self, server: bool = False):
        print("broadcast all")
        if server:
            print("broadcast server")
            self._broadcast_server(list(range(dist.get_world_size())))
        else:
            print("receive client")
            self._receive_client()

    def _broadcast_server(self, dst_ranks: List[int]):
        for i, rank in enumerate(dst_ranks):
            print(f"broadcasting to rank {rank}")
            data = self.dataset.tensors[0]
            targets = self.dataset.tensors[1]
            data_shape = torch.IntTensor([data.shape])

            if rank == self.rank:
                continue
            
            print(f"broadcasting to rank {rank} with shape {data_shape}")
            dist.send(data_shape, dst=rank)
            print(f"broadcasting to rank {rank} with data {data}")
            dist.send(data, dst=rank)
            print(f"broadcasting to rank {rank} with targets {targets}")
            dist.send(targets, dst=rank)

        if self.rank in dst_ranks:
            self._update_dataset(self.dataset.tensors[0], self.dataset.tensors[1])

    def _receive_client(self):
        print("receiving client")
        data_shape = torch.zeros(4).int() - 1
        dist.recv(data_shape, src=0)
        data_shape = data_shape[data_shape != -1]
        print(f"received data shape {data_shape} from rank 0")

        data = torch.zeros(torch.Size(data_shape))
        targets = torch.zeros(data_shape[0].item()).long()
        dist.recv(data, src=0)
        print(f"received data {data} from rank 0")
        dist.recv(targets, src=0)
        print(f"received targets {targets} from rank 0")

        self._update_dataset(data, targets)

    def _update_dataset(self, data: torch.Tensor, targets: torch.Tensor):
        if self.with_indices:
            self.dataset = TensorDataset(data, targets, torch.arange(data.shape[0]))
        else:
            self.dataset = TensorDataset(data, targets)
