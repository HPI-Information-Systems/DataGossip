from typing import Callable, Tuple, Dict, List
import torch.multiprocessing as mp
from torch.distributed import new_group
from ..connection import setup_cluster, Connector
import logging
from ctypes import c_bool


class Communicator:
    def __init__(self, connector: Connector, parameter_server: bool = False, total_local_ranks: int = 1):
        self.processes: List[mp.Process] = []
        self.shared_bool: List[mp.Value] = []
        self.connector = connector
        self.size = 0
        self.parameter_server = parameter_server
        self.total_local_ranks = total_local_ranks
        self.global_rank = connector.rank

    def setup(self, local_rank: int) -> Tuple[int, int, object]:
        self.connector.rank = self.connector.rank * self.size + local_rank
        n_nodes = self.connector.size
        self.connector.size = n_nodes * self.size
        rank, size = setup_cluster(self.connector)
        group = self.setup_groups(local_rank, n_nodes, self.size)
        logging.debug(f"{rank}: group joined")
        return rank, size, group

    def setup_groups(self, local_rank: int, n_nodes: int, ranks_per_node: int) -> object:
        rank_combinations = [
            [node * ranks_per_node + local_rank for node in range(n_nodes)]
            for local_rank in range(ranks_per_node)
        ]
        groups = []
        for ranks in rank_combinations:
            groups.append(new_group(ranks))
        return groups[local_rank]

    def task_wrapper(self, target: Callable, kwargs: Dict, local_rank: int):
        rank, size, group = self.setup(local_rank=local_rank)
        kwargs.update({"local_rank": local_rank})
        target(rank=rank, size=size, group=group, kwargs=kwargs)

    def add_process(self, target: Callable, kwargs: Dict) -> int:
        self.size += 1
        self.shared_bool.append(mp.Value(c_bool, True))
        kwargs["is_running"] = self.shared_bool[-1]
        self.processes.append(mp.Process(target=self.task_wrapper, args=(target, kwargs, self.size-1)))
        return self.size

    def start(self):
        for p in self.processes:
            p.start()

    def end(self):
        for is_running in self.shared_bool:
            with is_running.get_lock():
                is_running.value = False

    def killall(self):
        for p in self.processes:
            p.kill()

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()
