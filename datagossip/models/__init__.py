import torch.nn as nn
from enum import Enum

from .small import SmallModel
from .medium import MediumModel
from .large import LargeModel

one_dim_datasets = ["mnist", "fashionmnist", "emnist"]


class ModelSize(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

    def get_model_by_size(self, dataset: str) -> nn.Module:
        model_kwargs = {}
        model_kwargs.update({
            "in_channels": 1 if dataset in one_dim_datasets else 3,
            #"out_channels": 62 if dataset == "emnist" else 10
        })

        if self == self.SMALL:
            return SmallModel(**model_kwargs)
        elif self == self.MEDIUM:
            return MediumModel(**model_kwargs)
        else:  # if self == self.LARGE:
            return LargeModel(**model_kwargs)

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
