from typing import Tuple
import torch
from torch.utils.data.dataset import TensorDataset
import pandas as pd


DATASET_NAME = "ElectricDevices"
DATASET_PATH = f"~/datasets/UCRArchive_2018/{DATASET_NAME}/{DATASET_NAME}_%s.tsv"


def load_time_datasets() -> Tuple[TensorDataset, TensorDataset]:
    df_train = pd.read_csv(DATASET_PATH % "TRAIN", header=None, sep="\t")
    df_test = pd.read_csv(DATASET_PATH % "TEST", header=None, sep="\t")

    X_train = df_train.iloc[:, 1:].values
    y_train = df_train.iloc[:, 0].values

    X_test = df_test.iloc[:, 1:].values
    y_test = df_test.iloc[:, 0].values

    # print the distict classes
    print("classes:", set(y_train))

    # add channel dimension
    X_train = X_train[:, None, :]
    X_test = X_test[:, None, :]

    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())

    # sample test data; only take 1000 samples
    test_dataset = TensorDataset(torch.from_numpy(X_test[:1000]).float(), torch.from_numpy(y_test[:1000]).long())

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = load_time_datasets()
    