'''
Train an S4 model on sequential CIFAR10 / sequential MNIST with PyTorch for demonstration purposes.
This code borrows heavily from https://github.com/kuangliu/pytorch-cifar.

This file only depends on the standalone S4 layer
available in /models/s4/

* Train standard sequential CIFAR:
    python -m example --wandb
* Train sequential CIFAR grayscale:
    python -m example --grayscale --wandb
* Train MNIST:
    python -m example --dataset mnist --d_model 256 --weight_decay 0.0

The `S4Model` class defined in this file provides a simple backbone to train S4 models.
This backbone is a good starting point for many problems, although some tasks (especially generation)
may require using other backbones.

The default CIFAR10 model trained by this file should get
89+% accuracy on the CIFAR10 test set in 80 epochs.

Each epoch takes approximately 7m20s on a T4 GPU (will be much faster on V100 / A100).
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from einops import rearrange

import os
import argparse

#from models.s4.s4 import S4Block as S4  # Can use full version instead of minimal S4D standalone below
#from models.s4.s4d import S4D
from deep_ssm.models.s5_fjax.ssm import S5
from tqdm.auto import tqdm
from typing import Literal, Tuple, Optional, Union
from torchtyping import TensorType
import torch.nn.functional as F

import wandb


def split_train_val(train, val_split):
  train_len = int(len(train) * (1.0 - val_split))
  train, val = torch.utils.data.random_split(
    train,
    (train_len, len(train) - train_len),
    generator=torch.Generator().manual_seed(42),
  )
  return train, val


def view_transform(img, grayscale=False):
  if grayscale:
    return img.view(1, 1024).t()
  else:
    return img.view(3, 1024).t()

def view_transform_gs(img, grayscale=True):
  # tqdm has compatibility when calling functions using lambda 
  if grayscale:
    return img.view(1, 1024).t()
  else:
    return img.view(3, 1024).t()

class SequenceLayer(torch.nn.Module):
  def __init__(
    self,
    d_model: int,
    ssm_size: int,
    blocks: int = 1,
    dt_min: float = 0.001,
    dt_max: float = 0.1,
    bidirectional: bool = False,
    C_init: str = "complex_normal",
    bandlimit: Optional[float] = None,
    conj_sym: bool = False,
    clip_eigs: bool = False,
    step_rescale: float = 1.0,
    discretization: Optional[str] = "bilinear",
    # layer parameters
    dropout: float = 0.0,
    activation: str = "gelu",
    prenorm: bool = False,
    batchnorm: bool = False,
    bn_momentum: float = 0.9,
    **kwargs,
  ):
    super().__init__()
    self.d_model = d_model
    self.prenorm = prenorm
    self.batchnorm = batchnorm
    self.activation = activation

    self.seq = S5(
      d_model=d_model,
      ssm_size=ssm_size,
      blocks=blocks,
      dt_min=dt_min,
      dt_max=dt_max,
      bidirectional=bidirectional,
      C_init=C_init,
      bandlimit=bandlimit,
      conj_sym=conj_sym,
      clip_eigs=clip_eigs,
      step_rescale=step_rescale,
      discretization=discretization,
    )

    if self.activation in ["full_glu"]:
      self.out1 = torch.nn.Linear(d_model)
      self.out2 = torch.nn.Linear(d_model)
    elif self.activation in ["half_glu1", "half_glu2"]:
      self.out2 = torch.nn.Linear(d_model)

    if self.batchnorm:
      self.norm = torch.nn.BatchNorm1d(d_model, momentum=bn_momentum, track_running_stats=False)
    else:
      self.norm = torch.nn.LayerNorm(d_model)

    self.drop = torch.nn.Dropout(p=dropout)

    self.gelu = F.gelu  # if glu else None

  def apply_activation(self, x):
    # Apply activation
    if self.activation in ["full_glu"]:
      x = self.drop(self.gelu(x))
      x = self.out1(x) * torch.sigmoid(self.out2(x))
      x = self.drop(x)
    elif self.activation in ["half_glu1"]:
      x = self.drop(self.gelu(x))
      x = x * torch.sigmoid(self.out2(x))
      x = self.drop(x)
    elif self.activation in ["half_glu2"]:
      # Only apply GELU to the gate input
      x1 = self.drop(self.gelu(x))
      x = x * torch.sigmoid(self.out2(x1))
      x = self.drop(x)
    elif self.activation in ["gelu"]:
      x = self.drop(self.gelu(x))
    else:
      raise NotImplementedError(
        "Activation: {} not implemented".format(self.activation))
    return x

  def forward(self,
              x: TensorType["batch_size", "seq_length", "num_features"],
              state: Optional[TensorType["batch_size", "num_states"]] = None,
              **kwargs):

    skip = x
    if self.prenorm:
      if self.batchnorm:
        x
        x = self.norm(x.transpose(-2, -1)).transpose(-2,-1)
      else:
        x = self.norm(x)

    # Apply sequence model
    x, new_state = self.seq(signal=x, prev_state=state, **kwargs)

    x = self.apply_activation(x)

    # residual connection
    x = skip + x

    if not self.prenorm:
      if self.batchnorm:
        x = self.norm(x.transpose(-2, -1)).transpose(-2,-1)
    return x, new_state


class S5Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        ssm_size=384,
        d_model=256,
        n_layers=4,
        **kwargs,
    ):
        super().__init__()

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S5 layers as residual blocks
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                SequenceLayer(d_model, ssm_size, **kwargs)
            )

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        for layer in self.layers:
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)
            x, _ = layer(x)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x


def setup_optimizer(model, lr, weight_decay, epochs):
  """
  S4 requires a specific optimizer setup.

  The S4 layer (A, B, C, dt) parameters typically
  require a smaller learning rate (typically 0.001), with no weight decay.

  The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
  and weight decay (if desired).
  """

  # All parameters in the model
  all_parameters = list(model.parameters())

  # General parameters don't contain the special _optim key
  params = [p for p in all_parameters if not hasattr(p, "_optim")]

  # Create an optimizer with the general parameters
  optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

  # Add parameters with special hyperparameters
  hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
  hps = [
    dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
  ]  # Unique dicts
  for hp in hps:
    params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
    optimizer.add_param_group(
      {"params": params, **hp}
    )

  # Create a lr scheduler
  # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

  # Print optimizer info
  keys = sorted(set([k for hp in hps for k in hp.keys()]))
  for i, g in enumerate(optimizer.param_groups):
    group_hps = {k: g.get(k, None) for k in keys}
    print(' | '.join([
                       f"Optimizer group {i}",
                       f"{len(g['params'])} tensors",
                     ] + [f"{k} {v}" for k, v in group_hps.items()]))

  return optimizer, scheduler


###############################################################################
# Everything after this point is standard PyTorch training!
###############################################################################

# Training
def train():
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, desc="Training", unit="batch")):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        #tqdm.write(
        #    'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
        #    (batch_idx, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total, correct, total)
        #)

        wandb.log({"Train Loss": train_loss / (batch_idx + 1), "Train Accuracy": 100. * correct / total})


def eval(epoch, dataloader, checkpoint=False):
    global best_acc
    model.eval()
    eval_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc="Evaluating", unit="batch")):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            #tqdm.write(
            #    'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            #    (batch_idx, len(dataloader), eval_loss/(batch_idx+1), 100.*correct/total, correct, total)
            #)

            wandb.log({"Eval Loss": eval_loss / (batch_idx + 1), "Eval Accuracy": 100. * correct / total})

    # Save checkpoint.
    if checkpoint:
        acc = 100.*correct/total
        if acc > best_acc:
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc

        return acc


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
  # Optimizer
  parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
  parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
  # Scheduler
  # parser.add_argument('--patience', default=10, type=float, help='Patience for learning rate scheduler')
  parser.add_argument('--epochs', default=100, type=float, help='Training epochs')
  # Dataset
  parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'cifar10'], type=str, help='Dataset')
  parser.add_argument('--grayscale', action='store_true', help='Use grayscale CIFAR10')
  # Dataloader
  parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use for dataloader')
  parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
  # Model
  parser.add_argument('--n_layers', default=4, type=int, help='Number of layers')
  parser.add_argument('--d_model', default=512, type=int, help='Model dimension')
  # General
  parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
  parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')

  args = parser.parse_args()

  if args.wandb:
    wandb.init(project="example_cifar10", config=args)

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  best_acc = 0  # best test accuracy
  start_epoch = 0  # start from epoch 0 or last checkpoint epoch

  # Data
  print(f'==> Preparing {args.dataset} data..')




  if args.dataset == 'cifar10':

      if args.grayscale:
          transform = transforms.Compose([
              transforms.Grayscale(),
              transforms.ToTensor(),
              transforms.Normalize(mean=122.6 / 255.0, std=61.0 / 255.0),
              transforms.Lambda(view_transform_gs)
          ])
      else:
          transform = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
              transforms.Lambda(view_transform)
          ])

      # S4 is trained on sequences with no data augmentation!
      transform_train = transform_test = transform

      trainset = torchvision.datasets.CIFAR10(
          root='./data/cifar/', train=True, download=True, transform=transform_train)
      trainset, _ = split_train_val(trainset, val_split=0.1)

      valset = torchvision.datasets.CIFAR10(
          root='./data/cifar/', train=True, download=True, transform=transform_test)
      _, valset = split_train_val(valset, val_split=0.1)

      testset = torchvision.datasets.CIFAR10(
          root='./data/cifar/', train=False, download=True, transform=transform_test)

      d_input = 3 if not args.grayscale else 1
      d_output = 10

  elif args.dataset == 'mnist':

      transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Lambda(lambda x: x.view(1, 784).t())
      ])
      transform_train = transform_test = transform

      trainset = torchvision.datasets.MNIST(
          root='./data', train=True, download=True, transform=transform_train)
      trainset, _ = split_train_val(trainset, val_split=0.1)

      valset = torchvision.datasets.MNIST(
          root='./data', train=True, download=True, transform=transform_test)
      _, valset = split_train_val(valset, val_split=0.1)

      testset = torchvision.datasets.MNIST(
          root='./data', train=False, download=True, transform=transform_test)

      d_input = 1
      d_output = 10
  else: raise NotImplementedError

  # Dataloaders
  trainloader = torch.utils.data.DataLoader(
      trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
  valloader = torch.utils.data.DataLoader(
      valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
  testloader = torch.utils.data.DataLoader(
      testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


  # Model
  print('==> Building model..')
  model = S5Model(
      d_input=d_input,
      d_output=d_output,
      ssm_size=384,
      d_model=args.d_model,
      n_layers=args.n_layers,
      C_init="lecun_normal",
      batchnorm=True,
      bidirectional=True,
      blocks=3,
      clip_eigs=True,


  )

  model = model.to(device)
  if device == 'cuda':
      cudnn.benchmark = True

  if args.resume:
      # Load checkpoint.
      print('==> Resuming from checkpoint..')
      assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
      checkpoint = torch.load('./checkpoint/ckpt.pth')
      model.load_state_dict(checkpoint['model'])
      best_acc = checkpoint['acc']
      start_epoch = checkpoint['epoch']


  criterion = nn.CrossEntropyLoss()
  optimizer, scheduler = setup_optimizer(
      model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs
  )

  for epoch in tqdm(range(start_epoch, args.epochs)):
      if epoch == 0:
          pass #tqdm.write('Epoch: %d' % (epoch))
      else:
        wandb.log({"epoch": epoch, "Val Acc": val_acc})
      train()
      val_acc = eval(epoch, valloader, checkpoint=True)
      eval(epoch, testloader)
      scheduler.step()
      # print(f"Epoch {epoch} learning rate: {scheduler.get_last_lr()}")