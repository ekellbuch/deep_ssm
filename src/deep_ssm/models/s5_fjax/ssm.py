"""
References:

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Tuple, Optional, Union
import math
from deep_ssm.models.s5_fjax.jax_func import associative_scan, lecun_normal
from deep_ssm.models.s5_fjax.ssm_init import init_VinvB, init_CV, init_log_steps, make_DPLR_HiPPO, \
  trunc_standard_normal
from torchtyping import TensorType


# Discretization functions
def discretize_bilinear(Lambda: torch.Tensor,
                        B_tilde: torch.Tensor,
                        Delta: torch.Tensor,
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
  """Discretize a diagonalized, continuous-time linear SSM
  using bilinear transform method.
  Args:
      Lambda (complex64): diagonal state matrix              (P,)
          TensorType["num_states"]
      B_tilde (complex64): input matrix                      (P, H)
          TensorType["num_states", "num_features"]
      Delta (float32): discretization step sizes             (P,)
          TensorType["num_states"]
  Returns:
      discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
        Tuple[TensorType["num_states"], TensorType["num_states", "num_features"]]
  """
  # TODO: check complex vs real
  # Lambda = torch.view_as_complex(Lambda)
  Identity = torch.ones_like(Lambda)
  # compute bilinear transform
  BL = 1 / (Identity - (Delta / 2.0) * Lambda)
  # discretize the state matrix
  Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
  # discretize the input matrix
  B_bar = (BL * Delta)[..., None] * B_tilde

  # Lambda_bar = torch.view_as_real(Lambda_bar)
  # B_bar = torch.view_as_real(B_bar)

  return Lambda_bar, B_bar


def discretize_zoh(Lambda: torch.Tensor,
                   B_tilde: torch.Tensor,
                   Delta: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
  """Discretize a diagonalized, continuous-time linear SSM
  using zero-order hold method.
  Args:
      Lambda (complex64): diagonal state matrix              (P,)
        TensorType["num_states"]
      B_tilde (complex64): input matrix                      (P, H)
        TensorType["num_states", "num_features"]
      Delta (float32): discretization step sizes             (P,)
        TensorType["num_states"]
  Returns:
      discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
        Tuple[TensorType["num_states"], TensorType["num_states", "num_features"]]
  """
  # Identity = torch.ones(Lambda.shape[0], device=Lambda.device) # (replaced by -1)
  Lambda_bar = torch.exp(Lambda * Delta)
  B_bar = (1 / Lambda * (Lambda_bar - 1))[..., None] * B_tilde
  return Lambda_bar, B_bar


@torch.jit.script
def binary_operator(
  q_i: Tuple[torch.Tensor, torch.Tensor], q_j: Tuple[torch.Tensor, torch.Tensor]
):
  """Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
  Args:
      q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
      q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
  Returns:
      new element ( A_out, Bu_out )
  """
  A_i, b_i = q_i
  A_j, b_j = q_j
  # return A_j * A_i, A_j * b_i + b_j
  return A_j * A_i, torch.addcmul(b_j, A_j, b_i)


def apply_ssm(
  Lambda_bars: torch.Tensor,
  B_bars: torch.Tensor,
  C_tilde: torch.Tensor,
  D: torch.Tensor,
  input_sequence: torch.Tensor,
  conj_sym: bool = False,
  bidirectional: bool = False,
) -> Tuple[torch.Tensor,torch.Tensor]:
  """
  Apply a linear state-space model to an input sequence x_t and return y_{t+1}:
    x_{t+1} = A x_t + B u
    y_{t+1} = C_tilde x_{t+1} + D u

  :param Lambda_bars: diagonal state matrix: TensorType["num_states"]
  :param B_bars: input matrix: TensorType["num_states", "num_features"]
  :param C_tilde: output matrix: TensorType["num_features", "num_states"]
  :param D: feedthrough matrix: TensorType["num_features"]
  :param input_sequence:TensorType["seq_length", "num_features"]
  :param conj_sym:
  :param bidirectional:
  :return:  y, state:
    TensorType["seq_length", "num_features"], TensorType["num_states"]
  """
  cinput_sequence = input_sequence.type(
    Lambda_bars.dtype
  )  # Cast to correct complex type

  # compute Bu elements
  if B_bars.ndim == 3:
    # Dynamic timesteps (significantly more expensive)
    Bu_elements = torch.vmap(lambda B_bar, u: B_bar @ u)(B_bars, cinput_sequence)
  else:
    # Static timesteps
    Bu_elements = torch.vmap(lambda u: B_bars @ u)(cinput_sequence)

  if Lambda_bars.ndim == 1:  # Repeat for associative_scan
    Lambda_bars = Lambda_bars.tile(input_sequence.shape[0], 1)

  # compute state sequence using associative scan: x_{t+1} = A x_t + B u
  _, xs = associative_scan(binary_operator, (Lambda_bars, Bu_elements))

  if bidirectional:
    _, xs2 = associative_scan(
      binary_operator, (Lambda_bars, Bu_elements), reverse=True
    )
    xs = torch.cat((xs, xs2), axis=-1)

  # compute the feedthrough matrix:
  Du = torch.vmap(lambda u: D * u)(input_sequence)

  # TODO: the last element of xs (non-bidir) is the hidden state for bidir flag it

  # compute SSM output sequence y = C_tilde x + D u
  if conj_sym:
    y = torch.vmap(lambda x: 2 * (C_tilde @ x).real)(xs) + Du
  else:
    y = torch.vmap(lambda x: (C_tilde @ x).real)(xs) + Du
  return y, xs[-1]

Initialization = Literal["complex_normal", "lecun_normal", "truncate_standard_normal"]


class S5SSM(torch.nn.Module):
  def __init__(
    self,
    Lambda_re_init: torch.Tensor,
    Lambda_im_init: torch.Tensor,
    V: torch.Tensor,
    Vinv: torch.Tensor,
    H: int,
    P: int,
    C_init: str,
    dt_min: float,
    dt_max: float,
    conj_sym: bool = True,
    clip_eigs: bool = False,
    bidirectional: bool = False,
    step_rescale: float = 1.0,
    discretization: Literal["zoh", "bilinear"] = "bilinear",
    bandlimit: Optional[float] = None,
  ):
    # TODO: conj_sym,
    """The S5 SSM
    Args:
        Lambda_re_init  (complex64): Initial diagonal state matrix       (P,)
        V           (complex64): Eigenvectors used for init          (P,P)
        Vinv        (complex64): Inverse eigenvectors used for init  (P,P)
        h           (int32):     Number of features of input seq
        p           (int32):     state size
        k           (int32):     rank of low-rank factorization (if used)
        C_init      (string):    Specifies How B and C are initialized
                    Options: [factorized: low-rank factorization,
                            dense: dense matrix drawn from Lecun_normal]
                            dense_columns: dense matrix where the columns
                            of B and the rows of C are each drawn from Lecun_normal
                            separately (i.e. different fan-in then the dense option).
                            We found this initialization to be helpful for Pathx.
        discretization: (string) Specifies discretization method
                        options: [zoh: zero-order hold method,
                                bilinear: bilinear transform]
        dt_min:      (float32): minimum value to draw timescale values from when
                                initializing log_step
        dt_max:      (float32): maximum value to draw timescale values from when
                                initializing log_step
        step_rescale:  (float32): allows for changing the step size, e.g. after training
                                on a different resolution for the speech commands benchmark
    """
    super().__init__()

    self.conj_sym = conj_sym
    self.C_init = C_init
    self.bidirectional = bidirectional
    self.bandlimit = bandlimit
    self.step_rescale = step_rescale
    self.clip_eigs = clip_eigs

    if self.conj_sym:
      # Need to account for case where we actually sample real B and C, and then multiply
      # by the half sized Vinv and possibly V
      local_P = 2 * P
    else:
      local_P = P

    # Initialize diagonal state to state matrix Lambda (eigenvalues)
    self.Lambda_re = torch.nn.Parameter(Lambda_re_init)
    self.Lambda_im = torch.nn.Parameter(Lambda_im_init)

    Lambda = self.get_lambda()

    # Initialize input to state (B) matrix
    # TODO: remove torch.float
    self.B = torch.nn.Parameter(
      init_VinvB(lecun_normal(), Vinv)((local_P, H), torch.float)
    )

    #B_tilde = self.B[..., 0] + 1j * self.B[..., 1]

    # Initialize state to output (C) matrix
    if self.C_init in ["trunc_standard_normal"]:
      C_init = trunc_standard_normal
      # C_shape = (H, local_P, 2)
    elif self.C_init in ["lecun_normal"]:
      C_init = lecun_normal()
      # C_shape = (H, local_P, 2)
    elif self.C_init in ["complex_normal"]:
      C_init = torch.normal(0, 0.5 ** 0.5, (H, P, 2))
    else:
      raise NotImplementedError(
        "C_init method {} not implemented".format(self.C_init))

    if self.C_init in ["complex_normal"]:
      if self.bidirectional:
        # TODO check
        C = torch.cat((C_init, C_init), axis=-1, )
        self.C = torch.nn.Parameter(C)
      else:
        self.C = torch.nn.Parameter(C_init)
    else:
      if self.bidirectional:
        # TODO: check parametrization for optimization
        C1 = init_CV(C_init, (H, local_P), V)
        C2 = init_CV(C_init, (H, local_P), V)

        C_real = torch.cat((C1[..., 0], C2[..., 0]), axis=-1)
        C_imag = torch.cat((C1[..., 1], C2[..., 1]), axis=-1)

        C = torch.stack((C_real, C_imag), axis=-1,)
        self.C = torch.nn.Parameter(C)
      else:
        C = init_CV(C_init, (H, local_P), V)
        self.C = torch.nn.Parameter(C)
    # Initialize feedthrough (D) matrix
    self.D = torch.nn.Parameter(torch.rand(H, ))

    # Initialize learnable discretization timescale value
    self.log_step = torch.nn.Parameter(init_log_steps(P, dt_min, dt_max))

    if discretization == "zoh":
      self.discretize = discretize_zoh
    elif discretization == "bilinear":
      self.discretize = discretize_bilinear
    else:
      raise ValueError(f"Unknown discretization {discretization}")

    if self.bandlimit is not None:
      step = step_rescale * torch.exp(self.log_step)
      freqs = step / step_rescale * Lambda[:, 1].abs() / (2 * math.pi)
      mask = torch.where(freqs < bandlimit * 0.5, 1, 0)  # (64, )
      self.C = torch.nn.Parameter(
        torch.view_as_real(torch.view_as_complex(self.C) * mask)
      )

  def initial_state(self, batch_size: Optional[int]):
    batch_shape = (batch_size,) if batch_size is not None else ()
    return torch.zeros((*batch_shape, self.C.shape[-1]))

  def get_lambda(self):
    if self.clip_eigs:
      Lambda = torch.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
    else:
      Lambda = self.Lambda_re + 1j * self.Lambda_im
    return Lambda

  def get_BC_tilde(self):
    B_tilde = self.B[..., 0] + 1j * self.B[..., 1]
    C_tilde = self.C[..., 0] + 1j * self.C[..., 1]
    return B_tilde, C_tilde

  # NOTE: can only be used as RNN OR S5(MIMO) (no mixing)
  def forward(self,
              signal: torch.Tensor,
              prev_state: Optional[torch.Tensor] = None,
              step_rescale: Union[float, torch.Tensor] = 1.0)-> Tuple[torch.Tensor,torch.Tensor]:
    """

    Args:
      signal: TensorType["seq_length", "num_features"]
      prev_state: Optional[TensorType["num_states"]] = None
      step_rescale:

    Returns:
    """
    # get complex B and C
    B_tilde, C_tilde = self.get_BC_tilde()

    # get lambda
    Lambda = self.get_lambda()

    # set discretization step
    if isinstance(step_rescale, torch.Tensor):
      # TODO: check dim in step_rescale has ndim > 0
      step_scale = step_rescale * torch.exp(self.log_step)
    else:
      step_scale = torch.tensor(step_rescale) * torch.exp(self.log_step)

    # discretize A, B
    Lambda_bar, B_bar = self.discretize(Lambda, B_tilde, step_scale)

    # calculate y
    ys, state = apply_ssm(Lambda_bar, B_bar, C_tilde, self.D, signal, conj_sym=self.conj_sym, bidirectional=self.bidirectional)
    return ys, state

  def step(self,
            signal: torch.Tensor,
            prev_state: torch.Tensor,
            step_rescale: Union[float, torch.Tensor] = 1.0):
    """
    signal: TensorType["batch_size", "seq_length", "num_features"],
    prev_state: TensorType["batch_size", "num_states"],

    """
    B_tilde, C_tilde = self.get_BC_tilde()
    Lambda = self.get_lambda()

    step_scale = step_rescale * torch.exp(self.log_step)

    Lambda_bar, B_bar = self.discretize(Lambda, B_tilde, step_scale)

    Bu = B_bar @ signal.type(B_bar.dtype)

    # https://arxiv.org/abs/2208.04933v2, Eq. 2
    x = Lambda_bar * prev_state + Bu
    y = (C_tilde @ x + self.D * signal).real
    return y, x


class S5(torch.nn.Module):
  def __init__(
    self,
    d_model: int,
    ssm_size: Optional[int] = None,
    blocks: int = 1,
    dt_min: float = 0.001,
    dt_max: float = 0.1,
    bidirectional: bool = False,
    C_init: str = "lecun_normal",
    bandlimit: Optional[float] = None,
    conj_sym: bool = True,
    clip_eigs: bool = False,
    step_rescale: float = 1.0,
    discretization: Optional[str] = "bilinear",
  ):
    super().__init__()
    # init ssm
    assert (
      ssm_size % blocks == 0
    ), "blocks should be a factor of ssm_size"

    # init S5SSM
    block_size = ssm_size // blocks

    # Initialize state matrix A using approximation to HiPPO-LegS matrix
    Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)

    if conj_sym:
      block_size = block_size // 2
      ssm_size = ssm_size // 2

    Lambda, B, V, B_orig = map(
      lambda v: torch.tensor(v, dtype=torch.complex64),
      (Lambda, B, V, B_orig),
    )

    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T

    # If initializing state matrix A as block-diagonal, put HiPPO approximation
    Lambda = (Lambda * torch.ones((blocks, block_size))).ravel()
    V = torch.block_diag(*([V] * blocks))
    Vinv = torch.block_diag(*([Vc] * blocks))

    self.seq = S5SSM(
      H=d_model,
      P=ssm_size,
      Lambda_re_init=Lambda.real,
      Lambda_im_init=Lambda.imag,
      V=V,
      Vinv=Vinv,
      C_init=C_init,
      dt_min=dt_min,
      dt_max=dt_max,
      conj_sym=conj_sym,
      clip_eigs=clip_eigs,
      bidirectional=bidirectional,
      discretization=discretization,
      step_rescale=step_rescale,
      bandlimit=bandlimit)

  def initial_state(self, batch_size: Optional[int] = None) -> torch.Tensor:
    return self.seq.initial_state(batch_size)

  def apply_seq_without_state(self, s, rate):
    return self.seq(s, step_rescale=rate)

  def apply_seq_with_state(self, s, ps, rate):
    return self.seq(s, prev_state=ps, step_rescale=rate)

  def forward(self,
              signal: torch.Tensor,
              prev_state: Optional[torch.Tensor] = None,
              rate: Union[float, torch.Tensor] = 1.0) -> Tuple[torch.Tensor,torch.Tensor]:
    """
    Args:
      signal: TensorType["batch_size", "seq_length", "num_features"]
      prev_state:  Optional[TensorType["batch_size", "num_states"]] = None
      rate:
    Returns:

    """
    # rate can be a float or a tensor
    apply_s5_batch = torch.vmap(lambda signal, rate: self.seq(signal, step_rescale=rate), in_dims=(0, None), out_dims=(0,0))
    return apply_s5_batch(signal, rate)

  def step(self,
        signal: torch.Tensor,
        prev_state: Optional[torch.Tensor] = None,
        rate: Union[float, torch.Tensor] = 1.0):
    """
    Args:
      signal: TensorType["batch_size", "seq_length", "num_features"]
      prev_state: TensorType["batch_size", "num_states"]
      rate:
    """
    return self.seq(signal, prev_state, step_rescale=rate)

class GEGLU(torch.nn.Module):
  def forward(self, x):
    x, gates = x.chunk(2, dim=-1)
    return x * F.gelu(gates)


class S5Layer(torch.nn.Module):
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
    dropout: float = 0.0,
    activation: str = "gelu",
    batchnorm: bool = False,
    bn_momentum: float = 0.9,
    step_rescale: float = 1.0,
    bandlimit: Optional[float] = None,
    discretization: Optional[str] = "bilinear",
    # transposed=True,  # axis ordering (B, L, D) or (B, D, L)
  ):
    super().__init__()
    self.d_model = d_model
    self.batchnorm = batchnorm
    self.activation = activation
    #self.transposed = transposed

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
      self.out1 = torch.nn.Linear(d_model, d_model)
      self.out2 = torch.nn.Linear(d_model, d_model)
    elif self.activation in ["half_glu1", "half_glu2"]:
      self.out2 = torch.nn.Linear(d_model, d_model)

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
              rate: Optional[Union[float, torch.Tensor]] = 1.0):

    # Apply sequence model
    x, new_state = self.seq(signal=x, prev_state=state, rate=rate)

    x = self.apply_activation(x)

    return x, new_state

  def step(self, x, state, rate=None):
    """
    Step as a recurrent model, and apply continuously

    """
    # pass through step:
    x, new_state = self.seq.step(x, state, rate)
    x = self.apply_activation(x)

    return x, new_state


  def default_state(self, *batch_shape, device=None):
      return self.seq.initial_state(*batch_shape)

  @property
  def d_output(self):
      return self.d_model


if __name__ == "__main__":
  def tensor_stats(t: torch.Tensor):  # Clone of lovely_tensors for complex support
    return f"tensor[{t.shape}] n={t.shape.numel()}, u={t.mean()}, s={round(t.std().item(), 3)} var={round(t.var().item(), 3)}\n"

    # batch size, input_dim, output_dim


  batch_size = 1
  input_dim = 10
  seq_length = 15

  d_model = input_dim  # dimension of input and output embeddings
  ssm_size = 64
  x = torch.rand([batch_size, seq_length, input_dim])
  model = S5(d_model, ssm_size, )
  print("A", tensor_stats(model.seq.get_lambda().data))
  B, C_tilde = model.seq.get_BC_tilde()
  print("B", tensor_stats(B.data))
  print("C", tensor_stats(C_tilde.data))
  print("D", tensor_stats(model.seq.D.data))

  state = model.initial_state(batch_size)
  res = model(x, prev_state=state)
  print(res[0].shape, res[1].shape)

  # Hparam configuration
  hparams = {
    "d_model": 32,
    "ssm_size": 32,
    "blocks": 1,
  }

  model = S5(**hparams)
  # toy data:
  data = {"batch_size": 2,
          "sequence_length": 50,
          "input_size": 32,
          }

  (b, t, d) = (data["batch_size"], data['sequence_length'],
               data['input_size'])

  x = torch.randn((b, t, d))

  # check model
  state = model.initial_state(data['batch_size'])
  res = model(x, prev_state=state)
  print(res[0].shape, res[1].shape)

  outputs = res[0]
  assert outputs.shape == (b, t, data['input_size'])



