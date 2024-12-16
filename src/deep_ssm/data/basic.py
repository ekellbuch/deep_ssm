"""
basic.py

Extremely lightweight dataloaders for prototyping
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# the dataloader for the sequential mnist dataset
def load_sequential_mnist(split, batch_size):
    # Define the transform
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(1, -1))]
    )

    # Load the MNIST dataset
    is_train = split == "train"
    dataset = datasets.MNIST(
        root="./data", train=is_train, download=True, transform=transform
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=is_train, num_workers=0
    )

    return dataloader
