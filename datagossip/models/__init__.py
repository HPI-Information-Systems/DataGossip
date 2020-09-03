import torch.nn as nn

from .small import SmallModel
from .medium import MediumModel
from .large import LargeModel


SMALL  = 0
MEDIUM = 1
LARGE  = 2


def get_model_by_size(size: int, dataset: int) -> nn.Module:
    if dataset == 2:
        in_channels = 3
    else:
        in_channels = 1
    if size == SMALL:
        return SmallModel(in_channels)
    elif size == MEDIUM:
        return MediumModel(in_channels)
    elif size == LARGE:
        return LargeModel(in_channels)
    else:
        raise ValueError(f"Size of value {size} does not exist!")


def load_model(model_name, dataset):
    one_dim_datasets = ["mnist", "fashionmnist", "emnist"]

    model_kwargs = {}

    if model_name == "large":
        model_class = LargeModel
        model_kwargs.update({
            "in_channels": 1 if dataset in one_dim_datasets else 3,
            "out_channels": 62 if dataset == "emnist" else 10
        })

    return model_class(**model_kwargs)