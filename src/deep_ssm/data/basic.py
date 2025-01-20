"""
basic.py

Extremely lightweight dataloaders for prototyping
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# the dataloader for the sequential mnist/cifar dataset
def load_sequential_pixels(split, batch_size, dataset_name="mnist"):
    """
    Args:
        split (str): 'train' or 'test'
        batch_size (int): batch size
        dataset_name: 'mnist' or 'cifar'
    """
    # Define the transform
    # note that transforms.ToTensor() scales the input to be between 0 and 1 (and to be a float)
    # https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html
    if dataset_name == "mnist":
        d_input=1
    elif dataset_name == "cifar":
        d_input=3
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(d_input, -1))]
    )

    # Load the MNIST dataset
    is_train = split == "train"
    if dataset_name == "mnist":
        dataset = datasets.MNIST(
            root="./data", train=is_train, download=True, transform=transform
        )
    elif dataset_name == "cifar":
        dataset = datasets.CIFAR10(
            root="./data", train=is_train, download=True, transform=transform
        )

    # Create DataLoader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=is_train, num_workers=0
    )

    return dataloader


def split_train_val(train, val_split, seed):
    train_len = int(len(train) * (1.0 - val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        generator=torch.Generator().manual_seed(seed),
    )
    return train, val


def load_sequential_pixels_all(batch_size, val_split=0.1, seed=0, explore=False, dataset_name="mnist"):
    """
    It is frusturating that the MNIST dataset comes in train and test split only, so that the val_split has to be done manually out of the train
    So, we write a function that returns train, val, and test, all together!

    Args:
        batch_size (int): batch size
        val_split (float): proportion of the training data to be used as validation
        seed (int): seed for the random split
        explore (bool): if explore is True, we don't use the hold out test set
    """
    if dataset_name == "mnist":
        d_input = 1
    elif dataset_name == "cifar":
        d_input = 3
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(d_input, -1))]
    )

    if dataset_name == "mnist":
        bigtrainset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
    elif dataset_name == "cifar":
        bigtrainset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )

    trainset, valset = split_train_val(bigtrainset, val_split, seed)
    if explore:
        trainset, testset = split_train_val(trainset, val_split, seed)
    else:
        if dataset_name == "mnist":
            testset = datasets.MNIST(
                root="./data", train=False, download=True, transform=transform
            )
        elif dataset_name == "cifar":
            testset = datasets.CIFAR10(
                root="./data", train=False, download=True, transform=transform
            )

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, valloader, testloader
