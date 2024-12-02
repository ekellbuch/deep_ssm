"""
MinRNN mixer
Using quasi-DEER!
"""

import torch
import torch.nn as nn
import numpy as np
from accelerated_scan.warp import (
    scan,
)  # from https://github.com/proger/accelerated-scan


def quasi_deer_torch(
    f,
    diagonal_derivative,
    initial_state,  # (B,D)
    states_guess,  # (B,D, T)
    inputs,  # (B, d_input, T)
    num_iters=10,  # controls number of newton iterations
    k=0.0,  # controls the amount of damping
):
    """
    Currently is quasi-DEER/ELK
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
        fs = torch.func.vmap(f, in_dims=-1, out_dims=-1)(
            inputs[..., 1:], states[..., :-1]
        )  # (B,D,T-1), in your diagram above, the first interacton is f(e1, h1)

        # Compute the As and bs from fs and Jfs
        As = (1.0 - k) * torch.func.vmap(diagonal_derivative, in_dims=-1, out_dims=-1)(
            inputs[..., 1:], states[..., :-1]
        )  # (B, D, T-1)
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
        new_states.nan_to_num()  # zero out nans, in place modification to be more memory efficient
        return new_states

    deer_traces = []
    for _ in range(num_iters):
        states_guess = step(states_guess)
        deer_traces.append(states_guess)

    return deer_traces[-1]  # (B, D, T)


class MinRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MinRNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.input_weights = nn.Parameter(
            torch.randn(hidden_size, input_size) / (input_size**0.5)
        )
        self.input_bias = nn.Parameter(torch.zeros(hidden_size))
        self.recurrent_weights = nn.Parameter(
            torch.randn(hidden_size, hidden_size) / (hidden_size**0.5)
        )
        self.U_z = nn.Parameter(
            torch.randn(hidden_size, hidden_size) / (hidden_size**0.5)
        )
        self.b_u = nn.Parameter(torch.zeros(hidden_size))

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


class AugmentedGRUCell(nn.GRUCell):
    """
    torch.nn.GRUCell with diagonal derivative
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias)

    def diagonal_derivative(self, input, hidden):
        batch_size, hidden_size = hidden.shape
        input_size = input.shape[1]

        # Extract weights and biases
        w_ir, w_iz, w_in = self.weight_ih.chunk(3, 0)
        w_hr, w_hz, w_hn = self.weight_hh.chunk(3, 0)
        b_ir, b_iz, b_in = (
            self.bias_ih.chunk(3, 0) if self.bias_ih is not None else (0, 0, 0)
        )
        b_hr, b_hz, b_hn = (
            self.bias_hh.chunk(3, 0) if self.bias_hh is not None else (0, 0, 0)
        )

        # Compute intermediate terms
        rcomp = hidden @ w_hn.t() + b_hn
        ract = input @ w_ir.t() + b_ir + hidden @ w_hr.t() + b_hr
        r = torch.sigmoid(ract)

        zact = input @ w_iz.t() + b_iz + hidden @ w_hz.t() + b_hz
        z = torch.sigmoid(zact)

        n_act = input @ w_in.t() + b_in + r * rcomp
        n = torch.tanh(n_act)

        # Compute derivative terms (only diagonal elements)
        dzdh = z * (1 - z) * w_hz.diag()
        drdh = r * (1 - r) * w_hr.diag()
        dndh = (1 - n**2) * (r * w_hn.diag() + drdh * rcomp)

        # Compute the diagonal of the Jacobian
        diag_jacobian = -dzdh * n + (1 - z) * dndh + dzdh * hidden + z

        return diag_jacobian


class pRNN(nn.Module):
    """
    An RNN that can be evaluated in parallel using quasi-DEER
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
        num_iters=2,  # number of iterations for quasi-DEER
        method="minrnn",  # minrrn or gru
        parallel=True,  # parallel implementation
    ):
        super(pRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

        num_directions = 2 if bidirectional else 1
        self.num_directions = num_directions
        self.parallel = parallel

        if method == "minrnn":
            self.forward_cells = nn.ModuleList(
                [
                    MinRNNCell(
                        input_size if i == 0 else hidden_size * num_directions,
                        hidden_size,
                    )
                    for i in range(num_layers)
                ]
            )
        elif method == "gru":
            self.forward_cells = nn.ModuleList(
                [
                    AugmentedGRUCell(
                        input_size if i == 0 else hidden_size * num_directions,
                        hidden_size,
                    )
                    for i in range(num_layers)
                ]
            )
        else:
            raise ValueError(f"Unknown method {method}")

        if bidirectional:
            self.backward_cells = nn.ModuleList(
                [
                    MinRNNCell(
                        input_size if i == 0 else hidden_size * num_directions,
                        hidden_size,
                    )
                    for i in range(num_layers)
                ]
            )

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        self.num_iters = num_iters  # number of iterations for quasi-DEER

    def forward(self, input, hx=None):
        if not self.batch_first:
            input = input.transpose(0, 1)

        batch_size, seq_len, _ = input.size()
        num_directions = 2 if self.bidirectional else 1

        if hx is None:
            hx = torch.zeros(
                self.num_layers * num_directions,
                batch_size,
                self.hidden_size,
                device=input.device,
            )

        output = input
        h_n = []

        for layer in range(self.num_layers):
            forward_layer_output = self._process_layer(
                output, hx[layer * num_directions], self.forward_cells[layer]
            )
            if self.bidirectional:
                backward_layer_output = self._process_layer(
                    output.flip(1),
                    hx[layer * num_directions + 1],
                    self.backward_cells[layer],
                )
                backward_layer_output = backward_layer_output.flip(1)
                layer_output = torch.cat(
                    [forward_layer_output, backward_layer_output], dim=2
                )
            else:
                layer_output = forward_layer_output

            h_n.append(layer_output[:, -1, : self.hidden_size])
            if self.bidirectional:
                h_n.append(layer_output[:, 0, self.hidden_size :])

            output = layer_output
            if self.dropout_layer and layer < self.num_layers - 1:
                output = self.dropout_layer(output)

        h_n = torch.stack(h_n)

        if not self.batch_first:
            output = output.transpose(0, 1)

        return output, h_n

    def _process_layer(self, input, h0, cell):
        batch_size, seq_len, _ = input.size()
        h0 = h0.unsqueeze(1).expand(-1, seq_len, -1)
        input = input.transpose(1, 2)  # (batch_size, hidden_size, seq_len)

        device = input.device
        states_guess = torch.zeros(batch_size, self.hidden_size, seq_len, device=device)
        if self.parallel:
            output = quasi_deer_torch(
                f=cell,
                diagonal_derivative=cell.diagonal_derivative,
                initial_state=h0[:, 0],
                states_guess=states_guess,
                inputs=input,
                num_iters=self.num_iters,
            )  # (B,D,T)
            return output.transpose(1, 2)  # (batch_size, seq_len, hidden_size)
        else:
            output = []
            h = h0[:, 0]  # Initial hidden state
            for t in range(seq_len):
                h = cell(input[:, :, t], h)
                output.append(h)
            output = torch.stack(output, dim=1)  # (batch_size, seq_len, hidden_size)
            return output