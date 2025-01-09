"""
parallel RNN mixer

Includes the code for quasi deer (see quasi_deer_torch). 
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.checkpoint import checkpoint as checkpoint_torch
import torch.nn.functional as F
try:
    from accelerated_scan.warp import (
        scan,
    )  # from https://github.com/proger/accelerated-scan
except:
    try:
        from accelerated_scan.ref import scan
    except ImportError:
        print("accelerated_scan not installed; skipping. You won't be able to use the parallel implementation of pRNN.")

def _batch_along_time(x):
    assert x.ndim == 3, "Expects x to be (B, D, T)"
    return x.permute(0, 2, 1).flatten(0, 1)

def _unbatch_to_time(x, B):
    assert x.ndim == 2, "Expects x to be (B*T, D)"
    T = x.shape[0] // B
    return x.view(B, T, -1).permute(0, 2, 1)

def quasi_deer_torch(
    f,
    diagonal_derivative,
    initial_state,  # (B,D)
    states_guess,  # (B,D, T)
    inputs,  # (B, d_input, T)
    num_iters=10,  # controls number of newton iterations
    k=0.0,  # controls the amount of damping
    clip=False, # instead of scaling the diagonal Jacobian, clip the entries if there are less than / bigger than 1
    last_only=False,  # only differentiate through the last state
    while_loop=False, # run to convergence, using a while loop
):
    """
    quasi-ELK in torch!
    Args:
      f: a forward fxn that takes in an input and a state, and outputs the next state. (following torch.nn.GRUCell convention)
          In the context of a GRU, f is a GRU cell
          In pytorch setting, f should be able to handle the batch dimension
      diagonal_derivative: a forward fxn that takes in an input and a state, and returns a length D diagonal derivative
          In pytorch setting, f should be able to handle the batch dimension
      initial_state: (B,D)
      states_guess, (B, D, T)
      inputs, (B, d_input, T)
      num_iters: number of iterations to run
      k: int, k=0 is quasi-DEER, k is between 0 and 1, nonzero k is quasi-slim-ELK
    Notes:
    - The initial_state is the fixed starting point.
    The structure looks like the following.
    Let h0 be the initial_state (fixed), h[1:T-1] be the states, and e[0:T-2] be the inputs
    Then our graph looks like
    h0 -----> h1 ---> h2 ---> ..... h_{T-1} ----> h_{T}
              ^       ^       ^          ^          ^
              |       |                   |          |
              e0      e1       ..... e_{T-2}      e_{T-1}
    Use the pytorch scan from: https://github.com/proger/accelerated-scan
    This scan expects inputs in the form (B,D,T)
    Note that the RNN standard (for inputs) in pytorch is (T,B,d_input)
    This scan also requires powers of 2, so padding for now...
    """
    B = states_guess.shape[0]
    D = states_guess.shape[1]
    T = states_guess.shape[-1]
    padded_T = int(2 ** np.ceil(np.log2(T)))  # must be a power of 2
    device = states_guess.device

    def step(states):
        """
        states: B,D,T
        """
        # Evaluate f and its Jacobian in parallel across timesteps 1,..,T-1
        B, D, T = states.shape
        # inputs and states are (B,D,T-1), the first interacton is f(e1, h1)
        fs = f(_batch_along_time(inputs[..., 1:]), _batch_along_time(states[..., :-1]))
        fs = _unbatch_to_time(fs, B)  # (B,D,T-1)

        # Compute the As and bs from fs and Jfs
        # inputs and states are (B,D,T-1)
        As = diagonal_derivative(_batch_along_time(inputs[..., 1:]), _batch_along_time(states[..., :-1]))
        As = _unbatch_to_time(As, B)  # (B, D, T-1)
        if clip:
            As.clamp_(min=-1, max=1)
        elif k > 0:
            As.mul_(1.0 - k)
        bs = fs - As * states[..., :-1]  # (B, D, T-1)

        # initial_state is h0
        b0 = f(inputs[..., 0], initial_state)  # h1 = f(e0, h0), (B,D)
        A0 = torch.zeros_like(As[..., 0])  # (B,D)
        A = torch.cat(
            [A0.unsqueeze(-1), As, torch.ones([B, D, padded_T - T], device=device)],
            dim=-1,
        )  # (B, D, T)
        b = torch.cat(
            [b0.unsqueeze(-1), bs, torch.zeros([B, D, padded_T - T], device=device)],
            dim=-1,
        )  # (B, D, T)

        # parallel asscociative scan
        new_states = scan(A, b)[..., :T]  # (B, D, T)
        if not torch.is_complex(new_states):
            new_states.nan_to_num()  # zero out nans, in place modification to be more memory efficient
        return new_states

    #deer_traces = []
    # last_only does not seem to be a good idea :(
    if last_only:
        for _ in range(num_iters - 1):
            with torch.no_grad():
                states_guess = step(states_guess)
        states_guess = step(states_guess)
    elif while_loop:
        err = 100
        count = 0
        while err > 1e-4:
            states_guess_new = step(states_guess)
            err = torch.max(torch.abs(states_guess_new - states_guess))
            states_guess = states_guess_new
            count += 1
            if count > 1000:
                break
    else:
        for _ in range(num_iters):
            states_guess = step(states_guess)
            #deer_traces.append(states_guess)
    if while_loop:
        print("Number of iterations: ", count)
    return states_guess #deer_traces[-1]  # (B, D, T)


class MinRNNCell(nn.Module):
    """
    Uses Xavier initialization
    """
    def __init__(self, input_size, hidden_size, bias=False):
        super(MinRNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        # Input weights and bias
        self.input_weights = nn.Parameter(torch.empty(hidden_size, input_size))
        self.input_bias = nn.Parameter(torch.zeros(hidden_size))

        # Recurrent weights
        self.recurrent_weights = nn.Parameter(torch.empty(hidden_size, hidden_size))

        # Additional parameters for the update gate
        self.U_z = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b_u = nn.Parameter(torch.zeros(hidden_size))

        # Xavier initialization
        nn.init.xavier_uniform_(self.input_weights)
        nn.init.xavier_uniform_(self.recurrent_weights)
        nn.init.xavier_uniform_(self.U_z)

    def forward(self, input, prev_state):
        z = torch.tanh(torch.matmul(input, self.input_weights.T) + self.input_bias)
        u = torch.sigmoid(
            torch.matmul(prev_state, self.recurrent_weights.T)
            + torch.matmul(z, self.U_z.T)
            + self.b_u
        )
        state = u * prev_state + (1 - u) * z
        return state

    def diagonal_derivative(self, input, prev_state):
        z = torch.tanh(torch.matmul(input, self.input_weights.T) + self.input_bias)
        u = torch.sigmoid(
            torch.matmul(prev_state, self.recurrent_weights.T)
            + torch.matmul(z, self.U_z.T)
            + self.b_u
        )
        derivative = (
            u + (prev_state - z) * u * (1 - u) * self.recurrent_weights.diagonal()
        )
        return derivative


def checkpoint(f):
    def f_(self, *args):
        if self.checkpoint:
            return checkpoint_torch(f, self, *args, use_reentrant=False)
        else:
            return f(self, *args)
    return f_

class AugmentedGRUCell(nn.GRUCell):
    """
    torch.nn.GRUCell with diagonal derivative:
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias)

    def diagonal_derivative(self, input, hidden):
        batch_size, hidden_size = hidden.shape
        input_size = input.shape[1]

        # Chunk the weight matrices for the reset (r), update (z), and candidate (n) gates
        w_ir, w_iz, w_in = self.weight_ih.chunk(3, 0)  # input-to-hidden weights
        w_hr, w_hz, w_hn = self.weight_hh.chunk(3, 0)  # hidden-to-hidden weights

        # Chunk the biases if present; otherwise set them to zero
        b_ir, b_iz, b_in = (
            self.bias_ih.chunk(3, 0) if self.bias_ih is not None else (0, 0, 0)
        )
        b_hr, b_hz, b_hn = (
            self.bias_hh.chunk(3, 0) if self.bias_hh is not None else (0, 0, 0)
        )

        # Compute the intermediate terms for the candidate gate
        # rcomp = (h_{t-1} W_{hn} + b_{hn})
        rcomp = hidden @ w_hn.t() + b_hn

        # Compute the reset gate activation: r = sigmoid(x W_{ir} + h_{t-1} W_{hr} + b_ir + b_hr)
        ract = input @ w_ir.t() + b_ir + hidden @ w_hr.t() + b_hr
        r = torch.sigmoid(ract)

        # Compute the update gate activation: z = sigmoid(x W_{iz} + h_{t-1} W_{hz} + b_iz + b_hz)
        zact = input @ w_iz.t() + b_iz + hidden @ w_hz.t() + b_hz
        z = torch.sigmoid(zact)

        # Compute the candidate hidden: n = tanh(x W_{in} + b_in + r * (h_{t-1} W_{hn} + b_hn))
        n_act = input @ w_in.t() + b_in + r * rcomp
        n = torch.tanh(n_act)

        # Now we compute the diagonal derivatives.
        # dzdh: derivative of z w.r.t. h_{t-1} along the diagonal
        # z'(zact) = z*(1-z), and we multiply by the diagonal of W_hz
        dzdh = z * (1 - z) * w_hz.diag()

        # drdh: derivative of r w.r.t. h_{t-1} along the diagonal
        # r'(ract) = r*(1-r), multiplied by diagonal of W_hr
        drdh = r * (1 - r) * w_hr.diag()

        # dndh: derivative of n w.r.t. h_{t-1} along the diagonal
        # n'(n_act) = (1 - n^2)
        # n_act depends on h_{t-1} through r * (h_{t-1} W_{hn} + b_hn) and also drdh * rcomp
        # Thus dndh = (1-n^2) * [r * W_hn(diag) + drdh * rcomp]
        dndh = (1 - n**2) * (r * w_hn.diag() + drdh * rcomp)

       # Finally, the GRU update equation:
        # h_t = z * h_{t-1} + (1 - z) * n
        # d(h_t)/d(h_{t-1}) = z + (1-z)*dndh + (-dzdh * n + dzdh * h_{t-1})
        # The code simplifies to:
        # diag_jacobian = -dzdh*n + (1-z)*dndh + dzdh*hidden + z
        diag_jacobian = -dzdh * n + (1 - z) * dndh + dzdh * hidden + z

        return diag_jacobian


class LearnableDiagonal(torch.nn.Module):
    """
    Note that you could go even simpler and have the diagonal not be input and hidden dependent but just a learnable parameter
    I think this would be more like LTI (S5), instead of LTV (S6).
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Linear layers for input and state transformations
        self.input_linear = nn.Linear(input_size, hidden_size)
        self.state_linear = nn.Linear(hidden_size, hidden_size)
        self.ffn = FeedForward(hidden_size, hidden_size)

    def forward(self, input, state):
        """
        Compute the diagonal parameters as a function of input and state.
        Args:
            input: (batch_size, input_size)
            state: (batch_size, size) - hidden state
        Returns:
            Diagonal parameters (batch_size, size), values bounded between -1 and 1
        """
        # Linear readout from input and state
        linear_out = self.input_linear(input) + self.state_linear(state)
        nonlinear_out = self.ffn(linear_out) # add more nonlinearity

        # Apply tanh to bound values between -1 and 1
        return torch.tanh(nonlinear_out)


class FeedForward(nn.Module):
    """
    2 layer MLP inspired by transformer FFN
    We use to learn the initial states
    """
    def __init__(self, input_dim, output_dim):
        super(FeedForward, self).__init__()
        self.intermediate_dim = 4 * output_dim
        self.w1 = nn.Linear(input_dim, self.intermediate_dim, bias=False)
        self.w2 = nn.Linear(self.intermediate_dim, output_dim, bias=False)
        self.w3 = nn.Linear(input_dim, self.intermediate_dim, bias=False)

    def forward(self, x):
        h = F.silu(self.w1(x))
        return self.w2(h * self.w3(x))


class ReparameterizedCell(nn.Module):
    """
    A cell that operates in the Z-space where the Jacobian is diagonalized.
    Takes an original cell as input and performs reparameterization.

    """
    def __init__(self, original_cell, P0, P0_inv, eigenvalues, complex=False):
        """
        Args:
            original_cell (nn.GRUCell): The original GRU cell to reparameterize.
            P0 (torch.Tensor): Matrix of eigenvectors of the Jacobian, shape (hidden_size, hidden_size).
            P0_inv (torch.Tensor): Inverse of P0, shape (hidden_size, hidden_size).
        """
        super().__init__()
        self.original_cell = original_cell
        self.hidden_size = original_cell.hidden_size
        self.input_size = original_cell.input_size

        # Store P0 and P0_inv as buffers (non-trainable parameters)
        self.register_buffer("P0", P0)
        self.register_buffer("P0_inv", P0_inv)
        self.register_buffer("eigenvalues", eigenvalues)
   
    def forward(self, input, hx):
        """
        Forward pass in the Z-space with reparameterized parameters.

        Args:
            input (torch.Tensor): Input at the current time step, shape (batch_size, input_size).
            hx (torch.Tensor): Hidden state from the previous time step, shape (batch_size, hidden_size).

        Returns:
            new_hx (torch.Tensor): Updated hidden state, shape (batch_size, hidden_size).
        """
        # Transform the hidden state to Z-space:
        z_hx = torch.matmul(hx, self.P0_inv)

        # use the original cell's weights the output would be
        #if self.complex:
        #    input = input.to(dtype=torch.complex64)
        #    z_hx = z_hx.to(dtype=torch.complex64)

        o_new, z_new = self.original_cell(input, z_hx)        
        #o_old, z_old = self.original_cell(input, hx)

        # Transform back to X-space
        z_new = z_new @ self.P0

        return o_new, z_new

    def diagonal_derivative(self, input, hx):
        """
        Compute the diagonal of the Jacobian in Z-space.

        Args:
            input (torch.Tensor): Input tensor, shape (batch_size, input_size).
            hx (torch.Tensor): Hidden state tensor, shape (batch_size, hidden_size).

        Returns:
            diag_jacobian_z (torch.Tensor): Diagonal of the Jacobian in Z-space, shape (batch_size, hidden_size).
        """
        z_hx = torch.matmul(hx, self.P0_inv)

        # Compute the Jacobian diagonal in the Z-space
        diag_jacobian_z = self.original_cell.diagonal_derivative(input, z_hx)

        #if self.complex:
        #    diag_jacobian_z = diag_jacobian_z.to(dtype=torch.complex64)

        return diag_jacobian_z


def diagonalize_and_reparameterize(model_cell, input_size, hidden_size, x0=None, h0=None, as_real=False):
    """
    Compute the Jacobian at (x0, h0), diagonalize it, and return a reparameterized cell.

    Args:
        model_cell (nn.Module): The original cell to reparameterize.
        input_size (int): Input dimension.
        hidden_size (int): Hidden state dimension.
        x0 (torch.Tensor): Fixed input state for linearization, shape (1, input_size).
        h0 (torch.Tensor): Fixed hidden state for linearization, shape (1, hidden_size).

    Returns:
        reparam_cell (ReparameterizedCell): The reparameterized cell.
    """
    # Compute the Jacobian numerically around x0, h0
    if x0 is None:
        x0 = torch.zeros(1, input_size, requires_grad=True)  # Correct input size 
    if h0 is None:
        h0 = torch.zeros(1, hidden_size, requires_grad=True)  # Correct hidden size

    # Get the Jacobian
    jacobian = torch.autograd.functional.jacobian(lambda h: model_cell(x0, h)[1], h0)

    # Reshape Jacobian to (hidden_size, hidden_size)
    J_f = jacobian.squeeze()

    assert J_f.shape == (hidden_size, hidden_size)
    # Perform eigendecomposition: J_f = P0 @ V0 @ P0_inv
    # Given Jacobian not symmetric, the eigenvectors are imaginary 
    eigenvalues, eigvecs = torch.linalg.eig(J_f)

    P0_inv = torch.linalg.inv(eigvecs.cpu()).to(device=J_f.device)
    # Return the reparameterized cell
    return ReparameterizedCell(model_cell, eigvecs, P0_inv, eigenvalues)


class pRNN(nn.Module):

    """
    An RNN that can be evaluated in parallel using quasi-ELK
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=True,
        dropout=0,
        bidirectional=True,
        num_iters=2,  # number of iterations for quasi-ELK
        k = 0.0, # damping factor
        clip = False, # clip the diagonal Jacobian
        method="minrnn",  # minrnn or gru
        parallel=True,  # parallel implementation
        checkpoint=True,
        learn_diags=False, # learn the diagonal instead of using the nonlinear dynamics
        last_only=False, # only differentiate through the last state
        learn_initial_states=False,
        reparameterize=False, # reparameterize a cell
        while_loop=False, # run to convergence
    ):
        super(pRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.checkpoint = checkpoint
        self.reparameterize = reparameterize

        num_directions = 2 if bidirectional else 1
        self.num_directions = num_directions
        self.parallel = parallel
        self.k = k
        self.clip = clip
        self.learn_diags = learn_diags
        self.last_only = last_only
        self.learn_initial_states = learn_initial_states
        self.num_iters = num_iters  # number of iterations for quasi-DEER
        self.while_loop = while_loop

        method_fn = {"minrnn": MinRNNCell, "gru": AugmentedGRUCell}

        self.forward_cells = nn.ModuleList()

        for i in range(num_layers):
            input_size_ = input_size if i == 0 else hidden_size * num_directions
            cell = method_fn[method](
                input_size_, hidden_size, bias=bias
            )
            if reparameterize:
                cell = diagonalize_and_reparameterize(cell, input_size_, hidden_size)
            self.forward_cells.append(cell)

        if self.learn_diags:
            # print("learning the diagonals")
            self.fwd_diagonals = nn.ModuleList(
                [
                    LearnableDiagonal(
                        input_size if i == 0 else hidden_size * num_directions,
                        self.hidden_size,
                    )
                    for i in range(num_layers)
                ]
            )
        else:
            self.fwd_diagonals = [None] * num_layers
        if self.learn_initial_states:
            self.fwd_state_fns = nn.ModuleList(
                [
                    FeedForward(
                        input_size if i == 0 else hidden_size * num_directions,
                        self.hidden_size,
                    )
                    for i in range(num_layers)
                ]
            )
        else:
            self.fwd_state_fns = [None] * num_layers

        if bidirectional:
            self.reverse_cells = nn.ModuleList()
            for i in range(num_layers):
                input_size_ = input_size if i == 0 else hidden_size * num_directions
                cell = method_fn[method](
                    input_size_, hidden_size, bias=bias
                )
                if reparameterize:
                    cell = diagonalize_and_reparameterize(cell, input_size_, hidden_size)
                self.reverse_cells.append(cell)


            if self.learn_diags:
                self.rev_diagonals = nn.ModuleList(
                    [
                        LearnableDiagonal(
                            input_size if i == 0 else hidden_size * num_directions,
                            self.hidden_size,
                        )
                        for i in range(num_layers)
                    ]
                )
            else:
                self.rev_diagonals = [None] * num_layers
            if self.learn_initial_states:
                self.rev_state_fns = nn.ModuleList(
                    [
                        FeedForward(
                            input_size if i == 0 else hidden_size * num_directions,
                            self.hidden_size,
                        )
                        for i in range(num_layers)
                    ]
                )
            else:
                self.rev_state_fns = [None] * num_layers

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, input, hx=None):
        if not self.batch_first:
            input = input.transpose(0, 1)
    
        batch_size = input.size(0)
        num_directions = 2 if self.bidirectional else 1

        if hx is None:
            hx = torch.zeros(
                self.num_layers * num_directions,
                batch_size,
                self.hidden_size,
                device=input.device,
            )
            # Here is where we should see the layers 
        output = input
        h_n = torch.empty_like(hx)

        for layer in range(self.num_layers):
            forward_layer_output = self._process_layer(
                output, 
                hx[layer * num_directions],
                self.forward_cells[layer],
                self.fwd_diagonals[layer], 
                self.fwd_state_fns[layer]
            )
            if self.bidirectional:
                reverse_layer_output = self._process_layer(
                    output.flip(1),
                    hx[layer * num_directions + 1],
                    self.reverse_cells[layer],
                    self.rev_diagonals[layer],
                    self.rev_state_fns[layer],
                )
                reverse_layer_output = reverse_layer_output.flip(1)
                
                # concatenate outputs
                layer_output = torch.cat(
                    [forward_layer_output, reverse_layer_output], dim=2
                )
            else:
                layer_output = forward_layer_output

            h_n[layer*num_directions] = layer_output[:, -1, : self.hidden_size]
            if self.bidirectional:
                h_n[layer*num_directions + 1] = layer_output[:, 0, self.hidden_size:]

            output = layer_output
            if self.dropout_layer and layer < self.num_layers - 1:
                output = self.dropout_layer(output)

        if not self.batch_first:
            output = output.transpose(0, 1)

        return output, h_n

    @checkpoint
    def _process_layer(self, input, h0, cell, diagonal_fn=None, state_fn=None):
        """
        Function for working through each layer
        Args:
            input: input to the layer
            h0: initial hidden state
            cell: the RNN cell
            diagonal_fn: function to compute the diagonal of the Jacobian
            state_fn: function to compute the initial state
        """
        batch_size, seq_len, _ = input.size()
        h0 = h0.unsqueeze(1).expand(-1, seq_len, -1)

        if self.parallel:
            device = input.device
            if self.learn_initial_states:
                input_time_batch = input.flatten(0,1)
                states_guess = state_fn(input_time_batch) # (B*T, D)
                states_guess = states_guess.view(batch_size, seq_len, -1).permute(0,2,1)
            else:
                states_guess = torch.zeros(batch_size, self.hidden_size, seq_len, device=device)
            def diagonal_derivative(input_t, state_t):
                if diagonal_fn:
                    return diagonal_fn(input_t, state_t)
                else:
                    return cell.diagonal_derivative(input_t, state_t)

            input = input.transpose(1, 2)  # (batch_size, input_size, seq_len)
            output = quasi_deer_torch(
                f=cell,
                diagonal_derivative=diagonal_derivative,
                initial_state=h0[:, 0],
                states_guess=states_guess,
                inputs=input,
                num_iters=self.num_iters,
                k=self.k,
                clip=self.clip,
                last_only=self.last_only,
                while_loop=self.while_loop,
            )  # (B,D,T)
            return output.transpose(1, 2)  # (batch_size, seq_len, hidden_size)
        else:
            output = torch.empty_like(h0)
            h = h0[:, 0]  # Initial hidden state
            for t in range(seq_len):
                h = cell(input[:, t, :], h)
                output[:, t, :] = h
            return output
