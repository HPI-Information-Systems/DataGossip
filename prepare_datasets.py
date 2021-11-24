from datagossip.dataset.transform_vision_datasets import TransformScripts
import logging
from logging import Logger, INFO
import os


logger = Logger("Dataset Preparation", level=INFO)
handler = logging.StreamHandler()
handler.setLevel(INFO)
logger.addHandler(handler)


if __name__ == "__main__":
    location = os.path.abspath("data")
    logger.info(f"saving to location {location}")

    logger.info("Preparing FashionMNIST")
    TransformScripts.fashionmnist(location)
    logger.info("Preparing CIFAR10bw")
    TransformScripts.cifar10bw(location)
