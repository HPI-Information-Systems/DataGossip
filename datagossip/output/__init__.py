from typing import Dict


class BaseLogger:
    def __init__(self):
        self.config = self.set_config()
        self.pre_steps()

    def write(self, experiment_name, tag, value):
        raise NotImplementedError()

    def pre_steps(self):
        raise NotImplementedError()

    def set_config(self) -> Dict:
        return dict()
