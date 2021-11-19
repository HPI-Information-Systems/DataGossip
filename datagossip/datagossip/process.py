import torch
import torch.multiprocessing as mp
from queue import Empty
import torch.distributed as dist
from threading import Thread
from typing import Callable, Optional
from ctypes import c_bool
from logging import Logger, WARNING

from ..instance_selector import InstanceSelector
from ..utils.distributed import Cluster

logger = Logger("DataGossipProcess", level=WARNING)


def daemon_thread(func: Callable):
    def inner_func(*args):
        t = Thread(target=func, args=args)
        t.daemon = True
        t.start()
        return t

    return inner_func


class DataGossipProcess(mp.Process):
    INSTANCE_SELECTION = "instance_selection"
    ALL_GATHER = "all_gather"

    def __init__(self,
                 data: torch.Tensor,
                 targets: torch.Tensor,
                 instance_selector: InstanceSelector,
                 args):
        super().__init__()
        self.daemon = True
        self.is_running = mp.Value(c_bool, True)

        self.data = data
        self.targets = targets
        self.instance_selector = instance_selector
        self.k = args.k
        self.queue = mp.Queue()
        self.rank = args.rank
        self.size = args.size
        self.instance_selection_req = None
        self.all_gather_req = None
        self.main_address = args.main_address
        self.main_port = args.main_port
        self.dg_group = None

    def run_instance_selection(self):
        self.queue.put(DataGossipProcess.INSTANCE_SELECTION)

    def run_all_gather(self):
        self.queue.put(DataGossipProcess.ALL_GATHER)

    @daemon_thread
    def _instance_selection(self):
        self.instance_selector.select(self.data[self.dg_group.rank()],
                                      self.targets[self.dg_group.rank()],
                                      self.k)

    @daemon_thread
    def _all_gather(self):
        if self.instance_selection_req is not None and self.instance_selection_req.is_alive():
            self.instance_selection_req.join()
        self._all_gather_foreign_tensor(self.data),
        self._all_gather_foreign_tensor(self.targets)

    def _all_gather_foreign_tensor(self, tensor: torch.Tensor):
        group_rank = self.dg_group.rank()
        group_size = self.dg_group.size()
        dist.all_gather([tensor[r] for r in range(group_size)], tensor[group_rank], group=self.dg_group)

    def stop(self):
        with self.is_running.get_lock():
            self.is_running.value = False

    def _setup_cluster(self) -> Cluster:
        rank = self.rank * 2 + 1
        size = self.size * 2
        return Cluster(
                    rank=rank,
                    size=size,
                    main_address=self.main_address,
                    main_port=self.main_port
                )

    def _setup_groups(self):
        dist.new_group(ranks=[r for r in range(dist.get_world_size()) if (r % 2) == 0])
        self.dg_group = dist.new_group(ranks=[r for r in range(2, dist.get_world_size()) if (r % 2) == 1])

    def run(self) -> None:
        with self._setup_cluster():
            self._setup_groups()
            while self.is_running.value:
                try:
                    work = self.queue.get(timeout=5)
                    if work == DataGossipProcess.INSTANCE_SELECTION:
                        if self.instance_selection_req is None or not self.instance_selection_req.is_alive():
                            self.instance_selection_req = self._instance_selection()
                    elif work == DataGossipProcess.ALL_GATHER:
                        if self.all_gather_req is None or not self.all_gather_req.is_alive():
                            self.all_gather_req = self._all_gather()
                    else:
                        logger.warning(f"'{work}' is not a valid work task")
                except Empty:
                    pass
