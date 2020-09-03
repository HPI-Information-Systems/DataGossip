import torch.distributed as dist
import torch.multiprocessing as mp
from ctypes import c_bool
from typing import Dict

from .. import ProcessTemplate


class Conductor(ProcessTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_spinning = mp.Value(c_bool, True)

    def loop(self, rank: int, size: int, group, kwargs: Dict):
        while self.is_spinning.value:
            pass
        dist.barrier(group=group)
        dist.destroy_process_group()

    def end(self):
        with self.is_spinning.get_lock():
            self.is_spinning.value = False
