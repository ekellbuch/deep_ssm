"""
Code for LRU (https://proceedings.mlr.press/v202/orvieto23a.html) in torch
written by Jerry W. Liu
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


def theta_init(shape, max_phase):
    """Initialize phases uniformly between 0 and max_phase"""
    u = np.random.uniform(0, 1, shape)
    return np.log(max_phase * u)


def nu_init(shape, r_min, r_max):
    """Initialize magnitudes following the same distribution as JAX implementation"""
    u = np.random.uniform(0, 1, shape)
    return np.log(-0.5 * np.log(u * (r_max**2 - r_min**2) + r_min**2))


def gamma_log_init(nu_log, theta_log):
    """Initialize gamma based on nu and theta"""
    # Compute lambda diagonal elements
    diag_lambda = np.exp(-np.exp(nu_log) + 1j * np.exp(theta_log))
    # Return log(sqrt(1 - |lambda|^2))
    return np.log(np.sqrt(1 - np.abs(diag_lambda) ** 2))


def matrix_init(shape, normalization=1.0):
    """Initialize matrices using scaled normal distribution"""
    return np.random.normal(0, 1, shape) / normalization


class LRU(nn.Module):
    def __init__(
        self,
        d_hidden,  # hidden state dimension
        d_model,  # input and output dimensions
        r_min=0.0,  # smallest lambda norm
        r_max=1.0,  # largest lambda norm
        max_phase=6.28,  # max phase lambda
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.d_model = d_model

        # Initialize parameters
        # Lambda parameters (diagonal matrix)
        self.theta_log = nn.Parameter(
            torch.tensor(theta_init((d_hidden,), max_phase), dtype=torch.float32)
        )
        self.nu_log = nn.Parameter(
            torch.tensor(nu_init((d_hidden,), r_min, r_max), dtype=torch.float32)
        )

        # Initialize gamma_log based on nu and theta
        gamma_log_values = gamma_log_init(
            self.theta_log.detach().numpy(), self.nu_log.detach().numpy()
        )
        self.gamma_log = nn.Parameter(
            torch.tensor(gamma_log_values, dtype=torch.float32)
        )

        # B matrix (input projection)
        self.B_re = nn.Parameter(
            torch.tensor(
                matrix_init((d_hidden, d_model), np.sqrt(2 * d_model)),
                dtype=torch.float32,
            )
        )
        self.B_im = nn.Parameter(
            torch.tensor(
                matrix_init((d_hidden, d_model), np.sqrt(2 * d_model)),
                dtype=torch.float32,
            )
        )

        # C matrix (output projection)
        self.C_re = nn.Parameter(
            torch.tensor(
                matrix_init((d_model, d_hidden), np.sqrt(d_hidden)), dtype=torch.float32
            )
        )
        self.C_im = nn.Parameter(
            torch.tensor(
                matrix_init((d_model, d_hidden), np.sqrt(d_hidden)), dtype=torch.float32
            )
        )

        # D matrix (skip connection)
        self.D = nn.Parameter(
            torch.tensor(matrix_init((d_model,)), dtype=torch.float32)
        )

    def get_lambda(self):
        """Compute diagonal lambda matrix elements"""
        return torch.exp(-torch.exp(self.nu_log) + 1j * torch.exp(self.theta_log))

    def get_B(self):
        """Compute B matrix"""
        B = self.B_re + 1j * self.B_im
        return B * torch.exp(self.gamma_log).unsqueeze(-1)

    def get_C(self):
        """Compute C matrix"""
        return self.C_re + 1j * self.C_im

    def forward(self, inputs):
        """
        Forward pass: h_{t+1} = λh_t + Bx_{t+1}, y_t = Re[Ch_t + Dx_t]
        Args:
            inputs: tensor of shape [batch_size, seq_length, d_model]
        Returns:
            outputs: tensor of shape [batch_size, seq_length, d_model]
        """
        batch_size, seq_length, _ = inputs.shape
        device = inputs.device

        # Get parameters
        lambda_diag = self.get_lambda()  # [d_hidden]
        B = self.get_B()  # [d_hidden, d_model]
        C = self.get_C()  # [d_model, d_hidden]

        # Initialize hidden state
        h = torch.zeros(batch_size, self.d_hidden, dtype=torch.complex64, device=device)

        # Lists to store outputs
        outputs = []

        # Convert inputs to complex (real part only)
        x_complex = torch.complex(inputs, torch.zeros_like(inputs))

        # Recurrent processing
        for t in range(seq_length):
            # Current input
            x = x_complex[:, t, :]  # [batch_size, d_model]

            # Update hidden state: h = λh + Bx
            h = lambda_diag * h + torch.matmul(x, B.T)

            # Compute output: y = Re[Ch + Dx]
            y = torch.matmul(h, C.T).real + self.D * inputs[:, t, :]
            outputs.append(y)

        # Stack outputs
        outputs = torch.stack(outputs, dim=1)  # [batch_size, seq_length, d_model]

        return outputs


class LRUBlock(nn.Module):
    """Sequence layer with LRU, GLU, dropout and normalization"""

    def __init__(
        self,
        d_model: int,
        hidden_size: int = 32,
        dropout: float = 0.0,
        norm_type: str = "layer",
        r_min: float = 0.0,
        r_max: float = 1.0,
        max_phase: float = 6.28,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.norm_type = norm_type

        # LRU Unit
        self.rnn = LRU(
            d_hidden=hidden_size,
            d_model=d_model,
            r_min=r_min,
            r_max=r_max,
            max_phase=max_phase,
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
            x_norm = self.norm(u)
        else:
            # BatchNorm expects [batch, features, seq]
            x_norm = self.norm(u)

        # Apply LRU
        lru_out = self.rnn(x_norm)  # LRU expects [batch, seq, features]

        # Apply GELU activation
        lru_out = F.gelu(lru_out)
        lru_out = self.drop(lru_out)

        # GLU: out1(x) * sigmoid(out2(x))
        # Transpose to [batch, seq, features] for dense layers
        lru_out = lru_out
        out1 = self.out1(lru_out)
        out2 = self.out2(lru_out)
        glu_out = out1 * torch.sigmoid(out2)
        # Back to [batch, features, seq]
        glu_out = glu_out

        # Final dropout
        glu_out = self.drop(glu_out)

        # Skip connection
        return u + glu_out


class LRUModel(nn.Module):
    """Stack of LRU layers with optional input/output projections"""

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
        r_min: float = 0.0,
        r_max: float = 1.0,
        max_phase: float = 6.28,
        **kwargs,
    ):
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

        # Stack of LRU layers
        self.layers = nn.ModuleList(
            [
                LRUBlock(
                    d_model=d_model,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    norm_type=norm_type,
                    r_min=r_min,
                    r_max=r_max,
                    max_phase=max_phase,
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
            x: Input tensor [batch_size, d_input, seq_length] SWITCH TO BTD
            states: Optional list of initial hidden states for each layer
        Returns:
            output tensor [batch_size, d_output, seq_length] SWITCH TO BTD
        """
        # Initialize states if not provided
        if states is None:
            states = [None] * self.n_layers

        # Input projection and normalization
        if self.d_input != self.d_model:
            # [batch, d_input, seq] -> [batch, seq, d_input]
            x = self.input_proj(x)
            # [batch, seq, d_model] -> [batch, d_model, seq]

        if isinstance(self.input_norm, nn.LayerNorm):
            x = self.input_norm(x)
        else:
            x = self.input_norm(x)

        # Process through layers
        for layer, state in zip(self.layers, states):
            x = layer(x, state)

        # Output normalization and projection
        if isinstance(self.output_norm, nn.LayerNorm):
            x = self.output_norm(x)
        else:
            x = self.output_norm(x)

        if self.d_output != self.d_model:
            x = self.output_proj(x)

        return x


if __name__ == "__main__":

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Model parameters
    batch_size = 32
    seq_length = 200
    input_dim = 3
    output_dim = 2
    d_model = 64
    n_layers = 3
    hidden_size = 32

    # Generate random data
    x = torch.randn(batch_size, input_dim, seq_length)
    y_true = torch.randn(batch_size, output_dim, seq_length)

    # Create model
    model = LRUModel(
        d_input=input_dim,
        d_output=output_dim,
        d_model=d_model,
        n_layers=n_layers,
        hidden_size=hidden_size,
        dropout=0.1,
        # LRU specific parameters
        r_min=0.0,
        r_max=1.0,
        max_phase=6.28,
    )

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = x.to(device)
    y_true = y_true.to(device)

    # Print model size
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")

    # Forward pass
    y_pred = model(x)

    # Print shapes
    print("\nTensor shapes:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y_pred.shape}")

    # Print complex parameter shapes
    print("\nComplex parameter shapes:")
    for name, layer in model.named_modules():
        if isinstance(layer, LRU):
            lambda_diag = layer.get_lambda()
            B = layer.get_B()
            C = layer.get_C()
            print(f"Lambda shape: {lambda_diag.shape}")
            print(f"B shape: {B.shape}")
            print(f"C shape: {C.shape}")
            print(f"D shape: {layer.D.shape}")

    # Training loop
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\nTraining for 10 epochs...")
    for epoch in range(10):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    # Generate predictions with trained model
    model.eval()
    with torch.no_grad():
        y_pred = model(x)

    # Print final loss
    final_loss = criterion(y_pred, y_true)
    print(f"\nFinal Loss: {final_loss.item():.4f}")

    # Print statistics about complex parameters after training
    print("\nFinal complex parameter statistics:")
    for name, layer in model.named_modules():
        if isinstance(layer, LRU):
            lambda_diag = layer.get_lambda()
            print(
                f"\nLambda magnitude range: [{torch.abs(lambda_diag).min():.4f}, {torch.abs(lambda_diag).max():.4f}]"
            )
            print(
                f"Lambda phase range: [{torch.angle(lambda_diag).min():.4f}, {torch.angle(lambda_diag).max():.4f}]"
            )
