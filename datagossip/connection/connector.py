class Connector:
    def __init__(self, rank: int, size: int, master_address: str = '127.0.0.1', master_port: str = '29900', backend: str = 'gloo'):
        self.rank = rank
        self.size = size
        self.master_address = master_address
        self.master_port = master_port
        self.backend = backend

    def to_dict(self) -> dict:
        return dict(
            rank=self.rank,
            size=self.size,
            master_address=self.master_address,
            master_port=self.master_port,
            backend=self.backend
        )