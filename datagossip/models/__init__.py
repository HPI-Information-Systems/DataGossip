import torch.nn as nn
from enum import Enum

from .small import SmallModel
from .medium import MediumModel
from .large import LargeModel
from .small_time import SmallTimeModel
from .inception_time import InceptionNetwork

one_dim_datasets = ["mnist", "fashionmnist", "emnist", "cifar10bw", "time"]


class ModelSize(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    TIME = "time"
    INCEPTION = "inception"

    def get_model_by_size(self, dataset: str) -> nn.Module:
        model_kwargs = {}
        model_kwargs.update({
            "in_channels": 1 if dataset in one_dim_datasets else 3,
            #"out_channels": 62 if dataset == "emnist" else 10
        })
        if dataset == "time":
            model_kwargs.update({
                "out_channels": 7
            })

        if self == self.SMALL:
            return SmallModel(**model_kwargs)
        elif self == self.MEDIUM:
            return MediumModel(**model_kwargs)
        elif self == self.LARGE:
            return LargeModel(**model_kwargs)
        elif self == self.TIME:
            return SmallTimeModel(**model_kwargs)
        else:  # if self == self.INCEPTION:
            return InceptionNetwork(**model_kwargs)

    def __str__(self):
        return self.value


def load_model(model_name, dataset):
    model_kwargs = {}

    if model_name == "large":
        model_class = LargeModel
        model_kwargs.update({
            "in_channels": 1 if dataset in one_dim_datasets else 3,
            "out_channels": 62 if dataset == "emnist" else 10
        })

    return model_class(**model_kwargs)
