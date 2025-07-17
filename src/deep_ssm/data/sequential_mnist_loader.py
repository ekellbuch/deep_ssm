import torch
import torchvision
import torchvision.transforms as transforms
import lightning as L
from torch.utils.data import DataLoader, random_split
from typing import Optional


def split_train_val(train, val_split):
    """Split training data into train and validation sets."""
    train_len = int(len(train) * (1.0 - val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        generator=torch.Generator().manual_seed(42),
    )
    return train, val


def _mnist_sequential_transform(x):
    """Transform MNIST tensor to sequential format (784,) -> (784, 1)"""
    return x.view(1, 784).t()


class SequentialMNISTDataModule(L.LightningDataModule):
    """Lightning DataModule for Sequential MNIST."""
    
    def __init__(self, data_cfg):
        super().__init__()
        self.batch_size = data_cfg.get('batchSize', 50)
        self.num_workers = data_cfg.get('num_workers', 4)
        self.train_split = data_cfg.get('train_split', 1.0)
        self.data_dir = data_cfg.get('data_dir', './data')
        
        # For compatibility with speech BCI data loader interface
        
        # Transform to convert MNIST to sequential format (784,) -> (784, 1)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,)),  # MNIST normalization
            transforms.Lambda(_mnist_sequential_transform)  # (784, 1)
        ])
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def prepare_data(self):
        """Download MNIST data."""
        # Download train data
        torchvision.datasets.MNIST(
            root=self.data_dir, train=True, download=True
        )
        # Download test data
        torchvision.datasets.MNIST(
            root=self.data_dir, train=False, download=True
        )
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for different stages."""
        
        # Create train dataset
        full_trainset = torchvision.datasets.MNIST(
            root=self.data_dir, train=True, transform=self.transform
        )
        
        # Split into train and validation
        self.train_dataset, self.val_dataset = split_train_val(
            full_trainset, 1 - self.train_split
        )
        
        # Create test dataset
        self.test_dataset = torchvision.datasets.MNIST(
            root=self.data_dir, train=False, transform=self.transform
        )
    
    def train_dataloader(self):
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
    
    def val_dataloader(self):
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


def _cifar_sequential_transform_grayscale(img):
    """Transform grayscale CIFAR tensor to sequential format"""
    return img.view(1, 1024).t()  # (1024, 1)


def _cifar_sequential_transform_color(img):
    """Transform color CIFAR tensor to sequential format"""
    return img.view(3, 1024).t()  # (1024, 3)


class SequentialCIFARDataModule(L.LightningDataModule):
    """Lightning DataModule for Sequential CIFAR-10."""
    
    def __init__(self, data_cfg):
        super().__init__()
        self.batch_size = data_cfg.get('batchSize', 50)
        self.num_workers = data_cfg.get('num_workers', 4)
        self.train_split = data_cfg.get('train_split', 1.0)
        self.data_dir = data_cfg.get('data_dir', './data/cifar')
        self.grayscale = data_cfg.get('grayscale', False)
        
        # For compatibility with speech BCI data loader interface
        
        # Transform to convert CIFAR to sequential format
        if self.grayscale:
            self.transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=122.6 / 255.0, std=61.0 / 255.0),
                transforms.Lambda(_cifar_sequential_transform_grayscale)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.Lambda(_cifar_sequential_transform_color)
            ])
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def prepare_data(self):
        """Download CIFAR-10 data."""
        # Download train data
        torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=True
        )
        # Download test data
        torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=True
        )
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for different stages."""
        
        if stage == "fit" or stage is None:
            # Create train dataset
            full_trainset = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=True, transform=self.transform
            )
            
            # Split into train and validation
            self.train_dataset, self.val_dataset = split_train_val(
                full_trainset, 1 - self.train_split
            )
        
        if stage == "test" or stage is None:
            # Create test dataset
            self.test_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=False, transform=self.transform
            )
    
    def train_dataloader(self):
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
    
    def val_dataloader(self):
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )