from typing import Dict

from . import BaseLogger

from tensorboard import program
from torch.utils.tensorboard import SummaryWriter

LOG_DIR = "./logs"
TB_PORT = "6006"


class Tensorboard(BaseLogger):
    def set_config(self) -> Dict:
        default_config = super(Tensorboard, self).set_config()
        default_config.update({
            "log_dir": "./logs",
            "tb_port": "6006"
        })
        return default_config

    def pre_steps(self):
        self.writer = SummaryWriter(self.config.get("log_dir"))
        try:
            self._init_tb()
        except:
            pass

    def _init_tb(self):
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.config.get("log_dir"), '--port', self.config.get("tb_port"), '--bind_all'])
        url = tb.launch()

    def write(self, experiment_name, value_name, value, step=0):
        self.writer.add_scalars(experiment_name, {value_name: value}, step)
        self.writer.flush()