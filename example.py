'''
Train an S5 model on sequential CIFAR10 / sequential MNIST with PyTorch for demonstration purposes.
This code borrows heavily from https://github.com/state-spaces/s4.

This file only depends on the standalone S5 layer
available in srd/deep_ssm/models/ssm.py

* Train standard sequential CIFAR:
    python -m example --wandb
* Train sequential CIFAR grayscale:
    python -m example --grayscale --wandb
* Train MNIST:
    python -m example --dataset mnist --d_model 256 --weight_decay 0.0

The `S5Model` class defined in this file provides a simple backbone to train S5 models.
This backbone is a good starting point for many problems, although some tasks (especially generation)
may require using other backbones.

The default CIFAR10 model trained by this file should get
88+% accuracy on the CIFAR10 test set in 250 epochs.

'''
import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from deep_ssm.mixers.s5_fjax.ssm import S5
from tqdm.auto import tqdm
from typing import Literal, Tuple, Optional, Union, List
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
    conj_sym: bool = False,
    clip_eigs: bool = False,
    step_rescale: float = 1.0,
    discretization: str = "bilinear",
    # layer parameters
    dropout: float = 0.0,
    activation: str = "gelu",
    prenorm: bool = False,
    batchnorm: bool = False,
    bn_momentum: float = 0.9,
    # optional parameters
    bandlimit: float = None,
  ):
    super(SequenceLayer, self).__init__()
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
      conj_sym=conj_sym,
      clip_eigs=clip_eigs,
      step_rescale=step_rescale,
      discretization=discretization,
      bandlimit=bandlimit,
    )

    if self.activation in ["full_glu"]:
      self.out1 = torch.nn.Linear(d_model, d_model)
      self.out2 = torch.nn.Linear(d_model, d_model)
    elif self.activation in ["half_glu1", "half_glu2"]:
      self.out1 = nn.Identity()  # No-op layer
      self.out2 = nn.Linear(d_model, d_model)
    else:
      self.out1 = nn.Identity()
      self.out2 = nn.Identity()

    if self.batchnorm:
      self.norm = torch.nn.BatchNorm1d(d_model, momentum=bn_momentum, track_running_stats=False)
    else:
      self.norm = torch.nn.LayerNorm(d_model)

    self.drop = torch.nn.Dropout(p=dropout)

    self.gelu = F.gelu  # if glu else None

  def apply_activation(self, x):
    # Apply activation
    if self.activation == "full_glu":
      x = self.drop(self.gelu(x))
      out2_result = torch.sigmoid(self.out2(x))
      x = self.out1(x) * out2_result
      x = self.drop(x)
    elif self.activation == "half_glu1":
      x = self.drop(self.gelu(x))
      out2_result = torch.sigmoid(self.out2(x))
      x = x * out2_result
      x = self.drop(x)
    elif self.activation == "half_glu2":
      # Only apply GELU to the gate input
      x1 = self.drop(self.gelu(x))
      out2_result = torch.sigmoid(self.out2(x1))
      x = x * out2_result
      x = self.drop(x)
    elif self.activation == "gelu":
      x = self.drop(self.gelu(x))
    else:
      raise NotImplementedError(
        "Activation: {} not implemented".format(self.activation))
    return x

  def forward(self,
              x: torch.Tensor,
              state: torch.Tensor) -> torch.Tensor:
    """
    """
    skip = x  # (B, L, d_input)
    if self.prenorm:
      if self.batchnorm:
        x = self.norm(x.transpose(-1, -2)).transpose(-1, -2)
      else:
        x = self.norm(x)

    # Apply sequence model
    x, state = self.seq(x, state)  # (B, L, d_input)

    x = self.apply_activation(x)  # (B, L, d_input)

    # residual connection
    x = skip + x

    if not self.prenorm:
      if self.batchnorm:
        x = self.norm(x.transpose(-1, -2)).transpose(-1, -2)
    return x, state  # (B, L, d_input)


class S5Model(nn.Module):

  def __init__(
    self,
    d_input,
    d_output=10,
    ssm_size=384,
    d_model=256,
    n_layers=4,
    blocks: int = 1,
    dt_min: float = 0.001,
    dt_max: float = 0.1,
    bidirectional: bool = False,
    C_init: str = "complex_normal",
    conj_sym: bool = False,
    clip_eigs: bool = False,
    step_rescale: float = 1.0,
    # layer parameters
    dropout: float = 0.0,
    activation: str = "gelu",
    prenorm: bool = False,
    batchnorm: bool = False,
    bn_momentum: float = 0.9,
    bandlimit: float = None,
    discretization: str = "bilinear",

  ):
    super(S5Model, self).__init__()
    self.n_layers = n_layers

    # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
    self.encoder = nn.Linear(d_input, d_model)

    self.layers = nn.Sequential(*[
      SequenceLayer(d_model=d_model,
                    ssm_size=ssm_size,
                    blocks=blocks,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    bidirectional=bidirectional,
                    C_init=C_init,
                    conj_sym=conj_sym,
                    clip_eigs=clip_eigs,
                    step_rescale=step_rescale,
                    discretization=discretization,
                    dropout=dropout,
                    activation=activation,
                    prenorm=prenorm,
                    batchnorm=batchnorm,
                    bn_momentum=bn_momentum,
                    bandlimit=bandlimit,
                    ) for _ in range(self.n_layers)
    ])

    # Linear decoder
    self.decoder = nn.Linear(d_model, d_output)

  def initial_state(self, batch_size):
    # init different A layer:
    states = []
    for layer_idx in range(self.n_layers):
      state_layer_idx = self.layers[layer_idx].seq.initial_state(batch_size)
      states.append(state_layer_idx)
    return states

  def forward(self, x: torch.Tensor, states: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor,torch.Tensor]:
    """
    Input x is shape (B, L, d_input)
    """
    output = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

    # Map the layers
    # Each iteration of this loop will map (B, L, d_model) -> (B, L, d_model)
    if states is None:
      states = self.initial_state(x.shape[0])

    # Process each layer with its corresponding state
    new_states = []
    for i, layer in enumerate(self.layers):
        output, state = layer(output, states[i])  # Pass the current state to the current layer
        new_states.append(state)  # Collect the updated state

    # Pooling: average pooling over the sequence length
    output = output.mean(dim=-2)

    # Decode the outputs
    output = self.decoder(output)  # (B, d_model) -> (B, d_output)

    return output, states


def setup_optimizer(model, lr_factor, ssm_lr_base, weight_decay, epochs, steps_per_epoch):
  """
  S5 requires a specific optimizer setup.

  The S5 layer (A, B, C, dt) parameters typically
  require a smaller learning rate (typically 0.001), with no weight decay.

  The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
  and weight decay (if desired).
  """
  # follow BfastandCdecay:
  ssm_fn_list = ["Lambda_re", "Lambda_im", "log_step","norm"]
  not_optim = []

  lr = lr_factor * ssm_lr_base
  ssm_lr = ssm_lr_base

  def ssm_fn(param):
    if any(keyword in param[0] for keyword in ssm_fn_list):
      return 'ssm'
    elif any(keyword in param[0] for keyword in not_optim):
      return 'none'
    else:
      return 'regular'

  # Separate parameter groups based on function
  params = list(model.named_parameters())
  param_groups = {'none': [], 'ssm': [], 'regular': []}
  param_groups_names = {'none': [], 'ssm': [], 'regular': []}
  for param in params:
    group = ssm_fn(param)
    param_groups[group].append(param[1])
    param_groups_names[group].append(param[0])

  # Define different optimizers for each group
  optimizer = torch.optim.AdamW([

    {'params': param_groups['ssm'], 'lr': torch.tensor(ssm_lr), 'weight_decay': torch.tensor(0.0)},
    {'params': param_groups['regular'], 'lr': torch.tensor(lr), 'weight_decay': torch.tensor(weight_decay)},
  ])

  # Create a lr scheduler
  # include warmup
  scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1 / steps_per_epoch, total_iters=steps_per_epoch)
  # then move on to optimization
  scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch, last_epoch=-1)
  scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2],
                                                    milestones=[steps_per_epoch])

  return optimizer, scheduler



###############################################################################
# Everything after this point is standard PyTorch training!
###############################################################################

# Training
def train(epoch, vmap_model, dataloader):
  train_loss = 0
  correct = 0
  total = 0
  for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, total=len(dataloader), desc="Training", unit="batch")):
    # for batch_idx, (inputs, targets) in enumerate(trainloader):#, desc="Training", unit="batch")):
    inputs, targets = inputs.to(device), targets.to(device)

    optimizer.zero_grad()
    if batch_idx == 0:
      states = None
    outputs, states = vmap_model(inputs, states)
    loss = criterion(outputs, targets)
    if loss.dim() > 0:
      loss = loss.mean()  # Ensure the loss is a scalar

    loss.backward()
    update_gradient()
    train_loss += loss.item()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()

  avg_loss = train_loss / (batch_idx + 1)
  accuracy = 100. * correct / total

  if wandb.run:
    wandb.log({"Train Loss": avg_loss, "Train Accuracy": accuracy, "epoch": epoch})

    # Log gradient norms
    for name, param in model.named_parameters():
      if param.grad is not None:
        grad_norm = param.grad.norm().item()
        wandb.log({f"grad_norm/{name}": grad_norm})

    # Log LR:
    for i, param_group in enumerate(optimizer.param_groups):
      wandb.log({f"lr/group_{i}": param_group['lr']})
  else:
    pass
    # tqdm.write(
    # 'Train: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
    # (batch_idx, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total, correct, total)
  return avg_loss, accuracy


def eval(epoch, vmap_model, dataloader, checkpoint=False, log_name='Eval'):
  global best_acc
  eval_loss = 0
  correct = 0
  total = 0

  with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(dataloader):  # , desc="Evaluating", unit="batch")):
      inputs, targets = inputs.to(device), targets.to(device)
      if batch_idx == 0:
        # TODO: fix
        states = None
      outputs, states = vmap_model(inputs, states)
      loss = criterion(outputs, targets)
      if loss.dim() > 0:
        loss = loss.mean()  # Ensure the loss is a scalar
      eval_loss += loss.item()
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()

  acc = 100. * correct / total
  eval_loss = eval_loss / (batch_idx + 1)

  if wandb.run:
    wandb.log({f"{log_name} Loss": eval_loss, f"{log_name} Accuracy": acc, "epoch": epoch})
  else:
    pass
    # tqdm.write(
    #  'Epoch Idx: (%d/%d) | Loss: %.3f | Eval Acc: %.3f%% (%d/%d)' %
    #  (epoch, len(dataloader), eval_loss, acc, correct, total)
    # )

  # Save checkpoint.
  if checkpoint:
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
  parser.add_argument('--lr_factor', default=4.5, type=float, help='Learning rate factor')
  parser.add_argument('--ssm_lr_base', default=0.001, type=float, help='SSM LR rate')
  parser.add_argument('--weight_decay', default=0.07, type=float, help='Weight decay')
  # Scheduler
  # parser.add_argument('--patience', default=10, type=float, help='Patience for learning rate scheduler')
  parser.add_argument('--epochs', default=250, type=int, help='Training epochs')
  # Dataset
  parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'cifar10'], type=str, help='Dataset')
  parser.add_argument('--grayscale', action='store_true', help='Use grayscale CIFAR10')
  # Dataloader
  parser.add_argument('--num_workers', default=16, type=int, help='Number of workers to use for dataloader')
  parser.add_argument('--batch_size', default=50, type=int, help='Batch size')
  # Model
  parser.add_argument('--n_layers', default=6, type=int, help='Number of layers')
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
  torch.manual_seed(42)

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

    # S5 is trained on sequences with no data augmentation!
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
  else:
    raise NotImplementedError

  # Dataloaders
  trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
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
    dropout=0.1,
    discretization="zoh",
    conj_sym=True,
    dt_min=0.001,
    dt_max=0.1,
    activation="half_glu1",
    prenorm=True,
    bn_momentum=0.95,
  )
  model = model.to(device)
  #compiled_model = torch.compile(model)
  compiled_model = model

  if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

  steps_per_epoch = len(trainloader)
  criterion = nn.CrossEntropyLoss()

  optimizer, scheduler = setup_optimizer(
    compiled_model, lr_factor=args.lr_factor, ssm_lr_base=args.ssm_lr_base, weight_decay=args.weight_decay,
    epochs=args.epochs,
    steps_per_epoch=steps_per_epoch,
  )


  # Step into both the optimizer and scheduler
  @torch.compile(fullgraph=False)
  def update_gradient():
    optimizer.step()
    scheduler.step()


  for epoch in tqdm(range(start_epoch, args.epochs), desc="Running ", unit="epoch"):
    if epoch == 0:
      pass
    else:
      if wandb.run:
        wandb.log({"epoch": epoch, "Val Acc": val_acc})
      else:
        tqdm.write('Epoch: {} Val Acc {}'.format(epoch, val_acc))
    compiled_model.train()
    train_loss, train_acc = train(epoch, compiled_model, trainloader)
    compiled_model.eval()
    val_acc = eval(epoch, compiled_model, valloader, checkpoint=False, log_name='Val')
  compiled_model.eval()
  eval(epoch, compiled_model, testloader)