import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

class RNNUnit(nn.Module):
    def __init__(
        self, input_size=1, output_size=1, hidden_size=32, nonlinearity="linear"
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # Weight matrices
        self.A = nn.Linear(hidden_size, hidden_size, bias=False)  # hidden-to-hidden
        self.B = nn.Linear(input_size, hidden_size, bias=False)  # input-to-hidden
        self.C = nn.Linear(hidden_size, output_size, bias=False)  # hidden-to-output
        self.D = nn.Linear(
            input_size, output_size, bias=False
        )  # input-to-output (skip connection)

        # Initialize with orthogonal weights for better gradient flow
        nn.init.orthogonal_(self.A.weight)
        nn.init.orthogonal_(self.B.weight)
        nn.init.orthogonal_(self.C.weight)
        nn.init.orthogonal_(self.D.weight)

        # Set nonlinearity function
        if nonlinearity == "identity":
            self.sigma = lambda x: x
        elif nonlinearity == "tanh":
            self.sigma = torch.tanh
        elif nonlinearity == "relu":
            self.sigma = F.relu
        else:
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}.")

    def forward_step(self, u, x):
        """Single step forward.
        Args:
            u: current input [batch_size, input_size]
            x: current hidden state [batch_size, hidden_size]
        Returns:
            y: current output [batch_size, output_size]
            x_next: next hidden state [batch_size, hidden_size]
        """
        # Hidden state update: x_k = σ(Ax_{k-1} + Bu_k)
        x_next = self.sigma(self.A(x) + self.B(u))

        # Output: y_k = Cx_k + Du_k
        y = self.C(x_next) + self.D(u)

        return y, x_next

    def forward(self, u, x_init=None):
        """Forward pass through the sequence.
        Args:
            u: input sequence [batch_size, input_size, seq_length]
               OR [batch_size, seq_length] for scalar inputs
            x_init: initial hidden state (optional) [batch_size, hidden_size]
        Returns:
            outputs: sequence outputs [batch_size, output_size, seq_length]
            OR [batch_size, seq_length] for scalar outputs
        """
        # Handle both scalar and vector inputs
        if u.dim() == 2:  # [batch_size, seq_length]
            u = u.unsqueeze(1)  # Add channel dimension

        batch_size, n_channels, seq_length = u.shape
        assert (
            n_channels == self.input_size
        ), f"Expected input size {self.input_size}, got {n_channels}"

        # Initialize hidden state if not provided
        if x_init is None:
            x_init = torch.zeros(batch_size, self.hidden_size, device=u.device)

        # Transpose to iterate over sequence
        u = u.transpose(1, 2)  # [batch_size, seq_length, input_size]

        outputs = []
        x = x_init

        for t in range(seq_length):
            y, x = self.forward_step(u[:, t], x)
            outputs.append(y)

        # Stack outputs along sequence dimension
        outputs = torch.stack(outputs, dim=1)  # [batch_size, seq_length, output_size]

        # Transpose back to channel format
        outputs = outputs.transpose(1, 2)  # [batch_size, output_size, seq_length]

        # For scalar outputs, squeeze the channel dimension
        if self.output_size == 1:
            outputs = outputs.squeeze(1)

        return outputs


class RNNBlock(nn.Module):
    """Sequence layer with RNNUnit, GLU, dropout and normalization"""

    def __init__(
        self,
        d_model: int,
        hidden_size: int = 32,
        dropout: float = 0.0,
        norm_type: str = "layer",
        nonlinearity: str = "identity",
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.norm_type = norm_type

        # RNN Unit
        self.rnn = RNNUnit(
            input_size=d_model,
            output_size=d_model,
            hidden_size=hidden_size,
            nonlinearity=nonlinearity,
        )

        # Output projections for GLU
        self.out1 = nn.Linear(d_model, d_model)
        self.out2 = nn.Linear(d_model, d_model)

        # Normalization
        if norm_type == "layer":
            self.norm = nn.LayerNorm(d_model)
        else:  # "batch"
            self.norm = nn.BatchNorm1d(d_model)

        # Dropout
        self.drop = nn.Dropout(dropout)

    def forward(self, u, x_init=None):
        """
        Args:
            u: input tensor [batch_size, d_model, seq_length]
            x_init: initial hidden state (optional)
        Returns:
            output tensor [batch_size, d_model, seq_length]
        """
        # Apply normalization (handles different dimensions for layer/batch norm)
        if self.norm_type == "layer":
            # LayerNorm expects [batch, seq, features]
            x_norm = u.transpose(1, 2)
            x_norm = self.norm(x_norm)
            x_norm = x_norm.transpose(1, 2)
        else:
            # BatchNorm expects [batch, features, seq]
            x_norm = self.norm(u)

        # Apply RNN
        rnn_out = self.rnn(x_norm, x_init)

        # Apply GELU activation
        rnn_out = F.gelu(rnn_out)
        rnn_out = self.drop(rnn_out)

        # GLU: out1(x) * sigmoid(out2(x))
        # Transpose to [batch, seq, features] for dense layers
        rnn_out = rnn_out.transpose(1, 2)
        out1 = self.out1(rnn_out)
        out2 = self.out2(rnn_out)
        glu_out = out1 * torch.sigmoid(out2)
        # Back to [batch, features, seq]
        glu_out = glu_out.transpose(1, 2)

        # Final dropout
        glu_out = self.drop(glu_out)

        # Skip connection
        return u + glu_out


class RNNModel(nn.Module):
    """
    Stack of RNN layers with optional input/output projections
    """

    def __init__(
        self,
        d_input: int,
        d_output: int,
        d_model: int,
        n_layers: int,
        hidden_size: int = 32,
        dropout: float = 0.0,
        norm_type: str = "layer",
        layer_norm_first: bool = True,
        layer_norm_last: bool = True,
        nonlinearity: str = "identity",
        **kwargs,
    ):
        """
        Args:
            d_input: Input dimension
            d_output: Output dimension
            d_model: Model dimension (internal representation)
            n_layers: Number of RNN layers
            hidden_size: Hidden state size in each RNN
            dropout: Dropout probability
            norm_type: Type of normalization ("layer" or "batch")
            layer_norm_first: Whether to add LayerNorm after input projection
            layer_norm_last: Whether to add LayerNorm before output projection
            nonlinearity: Activation function for RNN ('identity', 'tanh', or 'relu')
        """
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.d_model = d_model
        self.n_layers = n_layers

        # Input projection if needed
        if d_input != d_model:
            self.input_proj = nn.Linear(d_input, d_model)
            if layer_norm_first:
                if norm_type == "layer":
                    self.input_norm = nn.LayerNorm(d_model)
                else:
                    self.input_norm = nn.BatchNorm1d(d_model)
            else:
                self.input_norm = nn.Identity()
        else:
            self.input_proj = nn.Identity()
            self.input_norm = nn.Identity()

        # Stack of RNN layers
        self.layers = nn.ModuleList(
            [
                RNNBlock(
                    d_model=d_model,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    norm_type=norm_type,
                    nonlinearity=nonlinearity,
                )
                for _ in range(n_layers)
            ]
        )

        # Output projection if needed
        if d_output != d_model:
            if layer_norm_last:
                if norm_type == "layer":
                    self.output_norm = nn.LayerNorm(d_model)
                else:
                    self.output_norm = nn.BatchNorm1d(d_model)
            else:
                self.output_norm = nn.Identity()
            self.output_proj = nn.Linear(d_model, d_output)
        else:
            self.output_norm = nn.Identity()
            self.output_proj = nn.Identity()

    def forward(
        self, x: torch.Tensor, states: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        Args:
            x: Input tensor [batch_size, d_input, seq_length]
            states: Optional list of initial hidden states for each layer
        Returns:
            output tensor [batch_size, d_output, seq_length]
        """
        # Initialize states if not provided
        if states is None:
            states = [None] * self.n_layers

        # Input projection and normalization
        if self.d_input != self.d_model:
            # [batch, d_input, seq] -> [batch, seq, d_input]
            x = x.transpose(1, 2)
            x = self.input_proj(x)
            # [batch, seq, d_model] -> [batch, d_model, seq]
            x = x.transpose(1, 2)

        if isinstance(self.input_norm, nn.LayerNorm):
            x = x.transpose(1, 2)
            x = self.input_norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.input_norm(x)

        # Process through layers
        new_states = []
        for layer, state in zip(self.layers, states):
            x = layer(x, state)
            # Store the final hidden state from each layer
            new_states.append(x)

        # Output normalization and projection
        if isinstance(self.output_norm, nn.LayerNorm):
            x = x.transpose(1, 2)
            x = self.output_norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.output_norm(x)

        if self.d_output != self.d_model:
            x = x.transpose(1, 2)
            x = self.output_proj(x)
            x = x.transpose(1, 2)

        return x

    def get_states(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        """Initialize hidden states for all layers"""
        return [
            torch.zeros(batch_size, self.d_model, device=device)
            for _ in range(self.n_layers)
        ]
