import torch.distributed as dist
import datetime


class Cluster:
    def __init__(self,
                 rank: int,
                 size: int,
                 master_address: str = "127.0.0.1",
                 master_port: int = 29900,
                 backend: str = "gloo"):
        self.rank = rank
        self.size = size
        self.master_address = master_address
        self.master_port = master_port
        self.backend = backend

    def _setup_cluster(self, rank, size):
        dist.init_process_group(self.backend,
                                init_method=f"tcp://{self.master_address}:{self.master_port}",
                                rank=rank, world_size=size, timeout=datetime.timedelta(0, 3600))

    def __enter__(self):
        self._setup_cluster(self.rank, self.size)

    def __exit__(self, exc_type, exc_val, exc_tb):
        dist.barrier()
        dist.destroy_process_group()
