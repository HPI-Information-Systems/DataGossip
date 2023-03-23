from .transform_vision_datasets import Transformer
from .data_loader import DistributedDataLoader
from .time import load_time_datasets
from typing import Tuple
from torch.utils.data.dataset import TensorDataset
import os
import logging
from logging import Logger, INFO

logger = Logger("Dataset Preparation", level=INFO)
handler = logging.StreamHandler()
handler.setLevel(INFO)
logger.addHandler(handler)

parent_dir = os.path.abspath(".")
data_dir = os.path.join(parent_dir, "data")

MNIST = "mnist"
FASHIONMNIST = "fashionmnist"
CIFAR10 = "cifar10"
CIFAR10BW = "cifar10bw"
EMNIST = "emnist"
TIME = "time"


def load_dataset(dataset: str = MNIST) -> Tuple[TensorDataset, TensorDataset]:
    train_dataset: TensorDataset = None
    test_dataset: TensorDataset = None
    try:
        if dataset == MNIST:
            dataset_name = "mnist_%s"
        elif dataset == FASHIONMNIST:
            dataset_name = "fashionmnist_%s"
        elif dataset == CIFAR10:
            dataset_name = "cifar10_%s"
        elif dataset == CIFAR10BW:
            dataset_name = "cifar10bw_%s"
        elif dataset == EMNIST:
            dataset_name = "emnist_%s"
        elif dataset == TIME:
            return load_time_datasets()
        else:
            return train_dataset, test_dataset
        train_dataset = Transformer(directory=data_dir, dataset_name=dataset_name % "train").load()
        test_dataset = Transformer(directory=data_dir, dataset_name=dataset_name % "test").load()
    except Exception as error:
        print(f"Dataset couldn't be loaded due to {error}. Waiting for main node to share it.")
    return train_dataset, test_dataset
