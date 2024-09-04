"""
1 batch  and 1epoch:
1 sec, 270 sec = 4.5 min

Desired timing:
250 epochs in 4hours
1 epoch in 1 min

# in gpu config, timing is: 1.78 batch/sec 1 epoch in 8min 30 sec original code

"""

import time


from pathlib import Path
import os
import sys
# Assuming all required functions and classes are imported from the appropriate modules
from tqdm.auto import tqdm

S5_JAX_PATH  = os.path.join(str(Path(__file__).parents[2]),'S5')
DEEP_SSM_PATH  = os.path.join(str(Path(__file__).parents[1]),'src')
#S5_JAX_PATH ="/home/groups/swl1/ekb/Projects/deepseq/S5"
#DEEP_SSM_PATH = "/home/groups/swl1/ekb/Projects/deepseq/deep_ssm/src"

#print(S5_JAX_PATH)
sys.path.insert(0, S5_JAX_PATH)
#print(DEEP_SSM_PATH)

sys.path.insert(0, DEEP_SSM_PATH)


#from s5.train_helpers import create_train_state, train_epoch, linear_warmup
from s5.dataloading import Datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from deep_ssm.models.s5_fjax.ssm import S5 as deepS5


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

    self.seq = deepS5(
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
      self.out1 = nn.Linear(d_model, d_model)
      self.out2 = nn.Linear(d_model, d_model)
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

    self.activation_block = nn.Sequential(
      nn.GELU(),
      nn.Dropout(dropout)
    )

  def apply_activation(self, x):
    # Apply activation
    if self.activation ==  "full_glu":
      x = self.activation_block(x)
      out2_result = torch.sigmoid(self.out2(x))
      x = self.out1(x) * out2_result
      x = self.drop(x)
    elif self.activation == "half_glu1":
      x = self.activation_block(x)
      out2_result = torch.sigmoid(self.out2(x))
      x = x * out2_result
      x = self.drop(x)
    elif self.activation == "half_glu2":
      # Only apply GELU to the gate input
      x1 = self.activation_block(x)
      out2_result = torch.sigmoid(self.out2(x1))
      x =  x * out2_result
      x = self.drop(x)
    elif self.activation == "gelu":
      x = self.activation_block(x)
    else:
      raise NotImplementedError(
        "Activation: {} not implemented".format(self.activation))
    return x

  def forward(self,
              x: torch.Tensor) -> torch.Tensor:

    skip = x
    if self.prenorm:
      if self.batchnorm:
        x = self.norm(x)
      else:
        x = self.norm(x)

    # Apply sequence model
    x = self.seq(x)

    x = self.apply_activation(x)

    # residual connection
    x = skip + x

    if not self.prenorm:
      if self.batchnorm:
        x = self.norm(x)
    return x


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

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S5 layers as residual blocks
        """
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                SequenceLayer(
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
                    dropout=dropout,
                    activation=activation,
                    prenorm=prenorm,
                    batchnorm=batchnorm,
                    bn_momentum=bn_momentum,
                    bandlimit=bandlimit,
                )
            )
        """
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
                              ) for _ in range(n_layers)
        ])

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x is shape (L, d_input)
        """
        output = self.encoder(x)  # (L, d_input) -> (L, d_model)

        # Map the layers
        output = self.layers(output)
        # Each iteration of this loop will map (d_model, L) -> (d_model, L)

        #for layer in self.layers:
        #  output = layer(output)
        # Pooling: average pooling over the sequence length
        output = output.mean(dim=-2)

        # Decode the outputs
        output = self.decoder(output)  # (d_model) -> (d_output)

        return output



def test_epoch_time(args):
    """
    Function to test the time taken for one epoch
    """

    # Set randomness

    # Get dataset creation function
    create_dataset_fn = Datasets[args.dataset]

    # Create dummy dataset with minimal size
    #init_rng, key = random.split(init_rng, num=2)
    trainloader, _, _, _, n_classes, seq_len, in_dim, _ = \
        create_dataset_fn(args.dir_name, seed=args.jax_seed, bsz=args.bsz)

    # Limit traindataloader so that it only returns 1 batch
    trainloader = [next(iter(trainloader)), next(iter(trainloader)), next(iter(trainloader))] # Only 1 batch
    # opt_config = 'BfastandCdecay'
    # ssm_lr_base = 0.001
    # warmup_end = 1
    # weight_decay = 0.07
    # mode = 'pool'
    # dt_global = False
    # lr_min = 0

    d_input = 1
    d_output = 10
    # Model
    print('==> Building model..')
    model = S5Model(
        d_input=d_input,
        d_output=d_output,
        ssm_size=args.ssm_size_base,
        d_model=args.d_model,
        n_layers=args.n_layers,
        C_init=args.C_init,
        batchnorm=args.batchnorm,
        bidirectional=args.bidirectional,
        blocks=args.blocks,
        clip_eigs=args.clip_eigs,
        dropout=args.p_dropout,
        discretization=args.discretization,
        conj_sym=args.conj_sym,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        activation=args.activation_fn,
        prenorm=args.prenorm,
        bn_momentum=args.bn_momentum,
    )

    # Created dummy input
    # batched_inputs = torch.randn(1, 8, 1, requires_grad=False)


    # put on device
    model = model.to(device)
    # compile mode and vmap model: scripted_model = 
    #scripted_model = torch.compile(model)
    #exit()
    scripted_model = torch.jit.script(model)
    #vmap_model = torch.func.vmap(scripted_model, in_dims=0, out_dims=0, randomness='same', chunk_size=None)

    # can vmap model without scripting!
    #vmap_model = torch.func.vmap(model, in_dims=0, randomness='same')

    vmap_model = torch.func.vmap(scripted_model, in_dims=0, randomness='same')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(scripted_model.parameters(), lr=0.001)

    # Measure time for one epoch
    start_time = time.time()

    model.train()
    for batch_idx, batch in enumerate(tqdm(trainloader, total=len(trainloader), desc="Training", unit="batch")):
        inputs, targets, aux_inputs = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()    
        outputs = vmap_model(inputs)
        #outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    epoch_time = time.time() - start_time
    print(f"Time taken for one epoch: {epoch_time:.2f} seconds")


# Example usage
class Args:
    # Define all necessary arguments here
    dir_name = '../../data'
    #
    C_init = 'lecun_normal'
    batchnorm = True
    bidirectional = True
    blocks = 3
    bsz = 50
    clip_eigs = True
    d_model = 8  # 512
    dataset = 'lra-cifar-classification'
    jax_seed = 16416
    lr_factor = 4.5
    n_layers = 6
    opt_config = 'BfastandCdecay'
    p_dropout = 0.1
    ssm_lr_base = 0.001
    ssm_size_base=384
    warmup_end=1
    weight_decay = 0.07
    conj_sym = True
    discretization = 'zoh'
    dt_min = 0.001
    dt_max = 0.1
    activation_fn = 'half_glu1'
    mode = 'pool'
    prenorm = True
    bn_momentum = 0.95
    dt_global = False
    lr_min = 0

args = Args()
test_epoch_time(args)