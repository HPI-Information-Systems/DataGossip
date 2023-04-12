from enum import Enum

one_dim_datasets = ["mnist", "fashionmnist", "emnist", "cifar10bw", "time"]


class ModelSize(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    TIME = "time"
    INCEPTION = "inception"

    def __str__(self):
        return self.value
    
    def __setstate__(self, state):
        print("setstate", state)
