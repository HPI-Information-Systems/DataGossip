import torch.multiprocessing as mp
from .communicator import Communicator
from typing import Dict
from ctypes import c_bool


class ProcessTemplate:
    def __init__(self, communicator: Communicator):
        self.communicator = communicator
        self.process_id = self.communicator.add_process(target=self.loop, kwargs=self._get_args()) - 1

    def loop(self, rank: int, size: int, group, kwargs: Dict):
        raise NotImplementedError()

    def _get_args(self) -> Dict:
        return dict()

    def is_alive(self) -> bool:
        return self.communicator.processes[self.process_id].is_alive()
