class Connector:
    def __init__(self, rank: int, size: int, main_address: str = '127.0.0.1', main_port: str = '29900', backend: str = 'gloo'):
        self.rank = rank
        self.size = size
        self.main_address = main_address
        self.main_port = main_port
        self.backend = backend

    def to_dict(self) -> dict:
        return dict(
            rank=self.rank,
            size=self.size,
            main_address=self.main_address,
            main_port=self.main_port,
            backend=self.backend
        )
