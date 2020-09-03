import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, EMNIST
from torchvision import transforms
import os
import fire
import tqdm


class Transformer:
    def __init__(self, dataset_name: str, directory: str = ".", dataset: Dataset = None):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.directory = directory
        self.tensor_dataset = None

    def transform(self):
        assert self.dataset is not None, "You need to set self.dataset to transform it to a TensorDataset"
        list_data = [self.dataset[i] for i in tqdm.trange(len(self.dataset), desc=f"Loading {self.dataset_name}...")]
        data, targets = list(zip(*list_data))
        data = torch.stack(data)
        targets = torch.LongTensor(targets)
        self.tensor_dataset = TensorDataset(data, targets)
        return self

    def save(self):
        assert self.tensor_dataset is not None, "You need to self.transform() your dataset first"
        data_file = f"{self.dataset_name}_data.pt"
        targets_file = f"{self.dataset_name}_targets.pt"
        torch.save(self.tensor_dataset.tensors[0], os.path.join(self.directory, data_file))
        torch.save(self.tensor_dataset.tensors[1], os.path.join(self.directory, targets_file))

    def load(self) -> TensorDataset:
        data_file = f"{self.dataset_name}_data.pt"
        targets_file = f"{self.dataset_name}_targets.pt"
        data = torch.load(os.path.join(self.directory, data_file))
        targets = torch.load(os.path.join(self.directory, targets_file))
        return TensorDataset(data, targets)


def transforms_simplecnn() -> transforms.Compose:
    return transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])


def transforms_pretrained() -> transforms.Compose:
    return transforms.Compose([
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class TransformScripts:
    @staticmethod
    def mnist(location="../data", pretrained=False):
        transform = transforms_pretrained() if pretrained else transforms_simplecnn()

        dataset = MNIST(location, train=True, download=True,
                        transform=transform)
        test_dataset = MNIST(location, train=False, download=True,
                             transform=transform)
        Transformer("mnist_train", location, dataset).transform().save()
        Transformer("mnist_test", location, test_dataset).transform().save()

    @staticmethod
    def fashionmnist(location="../data", pretrained=False):
        transform = transforms_pretrained() if pretrained else transforms_simplecnn()

        dataset = FashionMNIST(location, train=True, download=True,
                        transform=transform)
        test_dataset = FashionMNIST(location, train=False, download=True,
                             transform=transform)
        Transformer("fashionmnist_train", location, dataset).transform().save()
        Transformer("fashionmnist_test", location, test_dataset).transform().save()

    @staticmethod
    def cifar10(location="../data", pretrained=False):
        transform = transforms_pretrained() if pretrained else transforms_simplecnn()

        dataset = CIFAR10(location, train=True, download=True,
                               transform=transform)
        test_dataset = CIFAR10(location, train=False, download=True,
                                    transform=transform)
        Transformer("cifar10_train", location, dataset).transform().save()
        Transformer("cifar10_test", location, test_dataset).transform().save()

    @staticmethod
    def emnist(location="../data", pretrained=False):
        transform = transforms_pretrained() if pretrained else transforms_simplecnn()

        dataset = EMNIST(location, train=True, download=True,
                          transform=transform, split="balanced")
        test_dataset = EMNIST(location, train=False, download=True,
                               transform=transform, split="balanced")
        Transformer("emnist_train", location, dataset).transform().save()
        Transformer("emnist_test", location, test_dataset).transform().save()


if __name__ == "__main__":
    fire.Fire(TransformScripts)
