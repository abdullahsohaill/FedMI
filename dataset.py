import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import Tuple, List

def get_transforms(dataset_name="MNIST"):
    if dataset_name == "MNIST":
        return transforms.Compose([
            transforms.ToTensor(),
            # MNIST Mean and Std
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        # CIFAR defaults
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

def get_dataset(config):
    transform = get_transforms(config.dataset_name)
    
    if config.dataset_name == "MNIST":
        trainset = torchvision.datasets.MNIST(
            root=config.data_root, train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.MNIST(
            root=config.data_root, train=False, download=True, transform=transform
        )
    elif config.dataset_name == "CIFAR10":
        trainset = torchvision.datasets.CIFAR10(
            root=config.data_root, train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root=config.data_root, train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset_name}")
        
    return trainset, testset

def partition_iid(dataset, num_clients: int) -> List[List[int]]:
    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)
    split_indices = np.array_split(indices, num_clients)
    return [list(idx) for idx in split_indices]

def get_dataloader(dataset, indices: List[int], config, shuffle: bool = True) -> DataLoader:
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers
    )

def get_test_dataloader(testset, config) -> DataLoader:
    return DataLoader(
        testset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )