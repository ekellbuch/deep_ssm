from .data_loader import SpeechDataModule
from .sequential_mnist_loader import SequentialMNISTDataModule, SequentialCIFARDataModule

all_datasets = {
    "speech": SpeechDataModule,
    "sequential_mnist": SequentialMNISTDataModule,
    "sequential_cifar10": SequentialCIFARDataModule,
}