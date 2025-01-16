# %%
import torch
import math

from torch import nn
from einops import rearrange, einsum, repeat
from deep_ssm.mixers.prnn import quasi_deer_torch, checkpoint


class FastDepthwiseCausalConv1d(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, bias: bool=True, channel_last=False, activation=None, device=None, dtype=None):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(in_channels, kernel_size, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(in_channels, device=device, dtype=dtype)) if bias else None
        self.activation = activation
        self.channel_last = channel_last
        self.activations_dict = {"silu": torch.nn.functional.silu, None: lambda x: x, "exp": torch.exp}

        self._init_weights()
    
    def _init_weights(self):
        # handle xavier init for depthwise conv, since pytorch is wrong for grouped convs
        # see e.g. https://github.com/pytorch/pytorch/issues/23854
        # or https://github.com/huggingface/pytorch-image-models/issues/84
        fan_in = self.kernel_size
        fan_out = self.kernel_size
        a = math.sqrt(6 / (fan_in + fan_out))
        self.weight.data.uniform_(-a, a)
        nn.init.zeros_(self.bias.data) if self.bias is not None else None

    def forward(self, x):
        if self.channel_last:
            x = x.transpose(-1, -2)
        *batch_dims, dim, seq_len = x.shape
        x = rearrange(x, '... d l -> (...) d l')  # collapse batch dims
        y = self.activations_dict[self.activation](torch.nn.functional.conv1d(x, self.weight[:, None, :], bias=self.bias, groups=self.in_channels, padding=self.kernel_size-1)[..., :seq_len])
        y = y.reshape(*batch_dims, dim, seq_len)  # expand batch dims
        if self.channel_last:
            y = y.transpose(-1, -2)
        return y


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=-1) * self.scale * self.gamma


class TTTMLP(nn.Module):
    def __init__(
        self,
        d_model,
        d_conv,
        layer_idx=None,  # unused
        num_iters=2,  # number of iterations for quasi-DEER
        checkpoint=True,
        parallel=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_iters = num_iters
        self.checkpoint = checkpoint
        self.parallel = parallel

        self.q_conv1d = FastDepthwiseCausalConv1d(d_model, kernel_size=d_conv, bias=True, channel_last=True)
        self.k_conv1d = FastDepthwiseCausalConv1d(d_model, kernel_size=d_conv, bias=True, channel_last=True)
        # assume single head for now; need to change proj and out_proj if doing multihead
        self.in_proj = nn.Linear(d_model, 3 * d_model + 1, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = RMSNorm(d_model)

        # mamba style decay
        num_heads = 1
        A = torch.empty(num_heads, dtype=torch.float32).uniform_(0, 16)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        dt_min=0.001
        dt_max=0.1
        dt_init_floor=1e-4
        dt = torch.exp(
            torch.rand(num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        input_size = d_model
        hidden_size = d_model // 4
        output_size = d_model

        self.W1_size = (hidden_size, input_size)
        self.W2_size = (output_size, hidden_size)
        self.b1_size = (hidden_size,)

        # TODO: train these?
        W1_0 = nn.init.xavier_uniform_(torch.empty(1, *self.W1_size))
        W2_0 = nn.init.xavier_uniform_(torch.empty(1, *self.W2_size))
        b1_0 = torch.zeros(1, *self.b1_size)

        vectorized_params_0 = torch.cat([vectorize(p) for p in [W1_0, W2_0, b1_0]], dim=-1)
        #self.register_buffer("vectorized_params_0", vectorized_params_0)
        self.register_parameter("vectorized_params_0", nn.Parameter(vectorized_params_0))

    def transition_fn(self, input, prev_state):
        k, v, lr = torch.split(input, [self.d_model, self.d_model, 1], dim=-1)
        prev_vectorized_params = prev_state
        params = self.extract_params(prev_vectorized_params)
        new_params = gradient_update(params, k, v, lr)
        new_vectorized_params = torch.cat([vectorize(p) for p in new_params], dim=-1)
        return new_vectorized_params

    def diagonal_derivative(self, input, prev_state):
        k, v, lr = torch.split(input, [self.d_model, self.d_model, 1], dim=-1)
        params = self.extract_params(prev_state)
        diags = d_grad_update_d_params(params, k, v, lr, diagonal=True)
        vectorized_diags = torch.cat([vectorize(p) for p in diags], dim=-1)
        return vectorized_diags

    def extract_params(self, vectorized_params):
        B, _ = vectorized_params.shape
        W1, W2, b1 = torch.split(vectorized_params, [math.prod(self.W1_size), math.prod(self.W2_size), math.prod(self.b1_size)], dim=-1)
        W1, W2, b1 = W1.view(B, *self.W1_size), W2.view(B, *self.W2_size), b1.view(B, *self.b1_size)  # -1 for batch dim
        params = (W1, W2, b1)
        return params
    
    @checkpoint
    def forward(self, x):
        # x is (B, T, D)
        B, T, D = x.shape
        q, k, v, log_lr_ = torch.split(self.in_proj(x), [self.d_model, self.d_model, self.d_model, 1], dim=-1)
        q, k = self.q_conv1d(q), self.k_conv1d(k)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # mamba style decay 
        log_lr = -self.A_log.exp() * torch.nn.functional.softplus(log_lr_ + self.dt_bias)
        lr = 0.1 * log_lr.exp()  # Initialize at 0 learning rate to prevent blowing up
        #lr = 0.1 * torch.ones_like(log_lr)

        vectorized_params_0 = self.vectorized_params_0.expand(B, -1)
        if self.parallel:
            vectorized_params = quasi_deer_torch(
                f=self.transition_fn,
                diagonal_derivative=self.diagonal_derivative,
                initial_state=vectorized_params_0,
                states_guess=repeat(vectorized_params_0, "B D -> B D T", T=T),
                inputs=rearrange(torch.cat([k, v, lr], dim=-1), "B T D -> B D T"),
                num_iters=self.num_iters,
                k=0.1,
            )
            vectorized_params = rearrange(vectorized_params, "B D T -> (B T) D")
        else:
            curr_vectorized_params = vectorized_params_0
            vectorized_params = []
            for t in range(T):
                curr_vectorized_params = self.transition_fn(torch.cat([k[:, t], v[:, t], lr[:, t]], dim=-1), curr_vectorized_params)
                vectorized_params.append(curr_vectorized_params)
            vectorized_params = torch.stack(vectorized_params, dim=-2).reshape(B * T, -1)

        params = self.extract_params(vectorized_params)

        y = rearrange(mlp(params, rearrange(q, "B T D -> (B T) D")), "(B T) D -> B T D", B=B)
        #y = self.out_proj(y)  # causes nans
        #y = self.norm(y)
        return y  # TODO: use mamba block?


def batched_diag_kron(A, B):
    A_batch_size, *A_shape = A.shape
    B_batch_size, *B_shape = B.shape
    assert A_batch_size == B_batch_size
    A_diag = torch.diagonal(A, dim1=-2, dim2=-1)
    B_diag = torch.diagonal(B, dim1=-2, dim2=-1)
    # The diagonal of A ⊗ B is the (flattened) outer product of diag(A) and diag(B).
    kron_diag = einsum(A_diag, B_diag, "b n, b m -> b n m").flatten(1)
    return kron_diag

def batch_eye(n, batch_size, device=None, dtype=None):
    return torch.eye(n, device=device, dtype=dtype).repeat(batch_size, 1, 1)

def d_grad_loss_d_params(params, x, y_true, diagonal=True):
    W1, W2, b1 = params
    B, x_dim = x.shape
    B, hidden_size, input_size = W1.shape
    B, output_size, W2_in = W2.shape
    B, b1_dim = b1.shape
    assert hidden_size == W2_in
    assert b1_dim == hidden_size 
    assert x_dim == input_size

    #### first we get the forward pass
    phi = torch.relu(einsum(W1, x, "B hidden input, B input -> B hidden") + b1)
    #m_x = einsum(W2, phi, "B output hidden, B hidden -> B output")

    # derivative of relu
    dphi = (phi > 0).float()  # should be a diagonal matrix but we store as a vector

    #### now we compute the hessians
    #diff = y_true - m_x  # not needed because when phi is relu, all terms involving diff will be zero (since relu's second derivative is zero)
    
    # compute for W1
    W2_dphi = W2 * dphi[:, None, :]
    dphi_t_W2_t_W2_dphi = einsum(W2_dphi, W2_dphi, "B j i, B j k -> B i k")
    x_x_t = einsum(x, x, "B i, B j -> B i j") 
    d_grad_loss_d_W1 = matricize_linear_op(
        A=dphi_t_W2_t_W2_dphi,
        B=x_x_t,
    )

    # compute for W2
    phi_t_phi = einsum(phi, phi, "B hidden1, B hidden2 -> B hidden1 hidden2")
    d_grad_loss_d_W2 = matricize_linear_op(
        A=batch_eye(W2.shape[-2], B, device=W2.device, dtype=W2.dtype),
        B=phi_t_phi
    )

    # compute for b1
    d_grad_loss_d_b1 = dphi_t_W2_t_W2_dphi

    if diagonal:
        d_grad_loss_d_W1 = torch.diagonal(d_grad_loss_d_W1, dim1=-2, dim2=-1)
        d_grad_loss_d_W2 = torch.diagonal(d_grad_loss_d_W2, dim1=-2, dim2=-1)
        d_grad_loss_d_b1 = torch.diagonal(d_grad_loss_d_b1, dim1=-2, dim2=-1)

    return d_grad_loss_d_W1, d_grad_loss_d_W2, d_grad_loss_d_b1

def d_grad_update_d_params(params, x, y_true, lr, diagonal=True):
    W1, W2, b1 = params
    B, x_dim = x.shape
    B, hidden_size, input_size = W1.shape
    B, output_size, W2_in = W2.shape
    B, b1_dim = b1.shape
    assert hidden_size == W2_in
    assert b1_dim == hidden_size 
    assert x_dim == input_size

    hessians = d_grad_loss_d_params(params, x, y_true, diagonal=diagonal)

    return tuple(vectorize(torch.ones_like(p)) - add_non_batch_dims(lr, hessian) * hessian for p, hessian in zip(params, hessians))

def vectorize(X):
    # vectorize in column major order, assuming a batch dim
    batch_size, *X_shape = X.shape
    X = X.permute(0, *reversed(range(1, len(X_shape) + 1)))
    return X.flatten(1)


def matricize_linear_op(A, B):
    # vec(AXB) = (B^T ⊗ A) vec(X)
    # A, X, B are all batched 2D matrices, i.e. (B, _, _)
    B_T = B.transpose(-2, -1)
    B_T_kron_A = rearrange(einsum(B_T, A, "B M N, B P Q -> B M P N Q"), "B M P N Q -> B (M P) (N Q)")
    return B_T_kron_A

def mlp(params, x):
    W1, W2, b1 = params
    B, x_dim = x.shape
    B, hidden_size, input_size = W1.shape
    B, output_size, W2_in = W2.shape
    B, b1_dim = b1.shape
    assert hidden_size == W2_in
    assert b1_dim == hidden_size 
    assert x_dim == input_size

    h = torch.relu(einsum(W1, x, "B hidden input, B input -> B hidden") + b1)
    y = einsum(W2, h, "B output hidden, B hidden -> B output")
    return y

def l2_loss(model, params, x, y_true):
    y = model(params, x)
    return torch.sum((y - y_true) ** 2, axis=-1) / 2

def add_non_batch_dims(x, y):
    # x is (B,), y is (B, D1, D2, ..., DN)
    return x.view(-1, *([1] * (y.dim() - 1)))

def auto_gradient_update(model, params, x, y_true, lr):
    loss = l2_loss(model, params, x, y_true)
    # compute gradient with respect to parameters_tensor
    grads = torch.autograd.grad(loss, params, grad_outputs=torch.ones_like(loss), create_graph=True)
    new_params = tuple(p - add_non_batch_dims(lr, g) * g for p, g in zip(params, grads))
    return new_params

def gradient_update(params, x, y_true, lr):
    W1, W2, b1 = params
    B, x_dim = x.shape
    B, hidden_size, input_size = W1.shape
    B, output_size, W2_in = W2.shape
    B, b1_dim = b1.shape
    assert hidden_size == W2_in
    assert b1_dim == hidden_size 
    assert x_dim == input_size

    phi = torch.relu(einsum(W1, x, "B hidden input, B input -> B hidden") + b1)
    m_x = einsum(W2, phi, "B output hidden, B hidden -> B output")
    diff = y_true - m_x

    dphi = (phi > 0).float()  # should be a diagonal matrix but we store as a vector
    W2_dphi = W2 * dphi[:, None, :]
    dphi_t_W2_t_diff = einsum(W2_dphi, diff, "B j i, B j -> B i")

    grad_W1 = -einsum(dphi_t_W2_t_diff, x, "B hidden, B input -> B hidden input")
    grad_W2 = -einsum(diff, phi, "B output, B hidden -> B output hidden")
    grad_b1 = -dphi_t_W2_t_diff
    grads = grad_W1, grad_W2, grad_b1

    new_params = tuple(p - add_non_batch_dims(lr, g) * g for p, g in zip(params, grads))
    return new_params

# %%
#device = "cuda"
## for some reason the proger scan fails if we don't use cuda:0 or set CUDA_VISIBLE_DEVICES=0
#torch.cuda.set_device(1)
#model = TTTMLP(d_model=64, d_conv=4).to(device)
#x = torch.randn(2, 64, 64, device=device)
## %%
#model(x).shape
## %%
#