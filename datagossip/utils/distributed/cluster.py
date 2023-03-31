import torch.distributed as dist
import datetime


class Cluster:
    def __init__(self,
                 rank: int,
                 size: int,
                 main_address: str = "127.0.0.1",
                 main_port: int = 29900,
                 backend: str = "gloo"):
        self.rank = rank
        self.size = size
        self.main_address = main_address
        self.main_port = main_port
        self.backend = backend

    def _setup_cluster(self, rank, size):
        dist.init_process_group(self.backend,
                                init_method=f"tcp://{self.main_address}:{self.main_port}",
                                rank=rank, world_size=size, timeout=datetime.timedelta(0, 3600))

    def __enter__(self):
        print("Setting up cluster")
        self._setup_cluster(self.rank, self.size)
        print("Cluster setup done")

    def __exit__(self, exc_type, exc_val, exc_tb):
        dist.barrier()
        dist.destroy_process_group()
