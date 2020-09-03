import torch
import torch.distributed as dist
from .connector import Connector
from .message_types import BARRIER_MSG
import logging


def setup_cluster(connector: Connector):
    if connector.backend == 'gloo':
        dist.init_process_group(connector.backend, init_method=f"tcp://{connector.master_address}:{connector.master_port}", rank=connector.rank, world_size=connector.size)
    elif connector.backend == 'mpi':
        dist.init_process_group(connector.backend)
    else:
        raise ValueError(f"Distributed backend '{connector.backend}' is not supported!")
    return dist.get_rank(), dist.get_world_size()


def wait_for_finish(connector: Connector):
    connector.master_port = int(connector.master_port) + 1
    rank, size = setup_cluster(connector)
    logging.debug(f"barrier {rank}")
    if dist.get_backend() == dist.Backend.GLOO:
        dist.barrier()
    elif dist.get_backend() == dist.Backend.MPI:
        custom_barrier(rank, size)


def custom_barrier(rank: int, size: int):
    t = torch.zeros(1).int()
    next = (rank + 1) % size
    prev = (rank - 1) % size
    for i in range(2):
        if rank == 0:
            dist.send(t, dst=next, tag=BARRIER_MSG)
            dist.recv(t, src=prev, tag=BARRIER_MSG)
        else:
            dist.recv(t, src=prev, tag=BARRIER_MSG)
            dist.send(t, dst=next, tag=BARRIER_MSG)
