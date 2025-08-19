# Import necessary libraries
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np # For initial data generation and conversion
from typing import Tuple
import torch
import pytest

# Helper function to convert PyTorch tensors to JAX numpy arrays
# (assuming the user might provide PyTorch snippets)
def to_jax(x):
    # This function would be relevant if input was PyTorch.
    # For a pure JAX Colab, we'll ensure inputs are already JAX arrays or numpy.
    if hasattr(x, 'cpu') and hasattr(x, 'detach') and hasattr(x, 'numpy'):
        return x.cpu().detach().numpy()
    return x # Assume it's already JAX array or numpy if not PyTorch

def discretize_jax(Lambda: jnp.ndarray,
                   B_tilde: jnp.ndarray,
                   Delta: jnp.ndarray,
                   ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """ Discretize a diagonalized, continuous-time linear SSM using JAX.

    Args:
        Lambda (complex64): diagonal state matrix (P,)
        B_tilde (complex64): input matrix (P, H)
        Delta (float32): discretization step sizes (P,)

    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64) (P,), (P,H)
    """
    # Identity vector, same shape as Lambda, with ones
    # This is effectively multiplying by 1 for each element of Lambda
    Identity = jnp.ones(Lambda.shape[0], dtype=Lambda.dtype)

    # Discretized Lambda_bar = exp(Lambda * Delta)
    # This is element-wise exponentiation, as Lambda is diagonal
    Lambda_bar = jnp.exp(Lambda * Delta)

    # Calculate the B_bar term.
    factor = (1 / Lambda * (Lambda_bar - Identity))[..., None]
    B_bar = factor * B_tilde

    return Lambda_bar, B_bar


def discretize_torch(Lambda: torch.Tensor,
                        B_tilde: torch.Tensor,
                        Delta: torch.Tensor,
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
  """Discretize a diagonalized, continuous-time linear SSM using PyTorch.

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
  Identity = torch.ones_like(Lambda)
  Lambda_bar = torch.exp(Lambda * Delta)
  B_bar = (1/ Lambda * (Lambda_bar - Identity))[..., None] * B_tilde
  return Lambda_bar, B_bar

# Helper function to convert JAX arrays to PyTorch tensors
def to_torch(x):
    return torch.from_numpy(np.asarray(x))


def forward_pass_jax(Lambda_bar: jnp.ndarray,
                     B_bar: jnp.ndarray,
                     inputs: jnp.ndarray,
                     initial_state: jnp.ndarray,
                     ) -> jnp.ndarray:
    """ Performs the forward pass of a discretized diagonalized SSM in JAX.

    Args:
        Lambda_bar (complex64): discretized diagonal state matrix (P,)
        B_bar (complex64): discretized input matrix (P, H)
        inputs (complex64): input sequence (T, H)
        initial_state (complex64): initial state (P,)

    Returns:
        outputs (complex64): output sequence (T, P) -
                             Note: This example returns states as outputs for simplicity.
                             In a full SSM, you'd have a C and D matrix.
    """
    num_timesteps = inputs.shape[0]
    state = initial_state
    states_sequence = []

    # Iterate through time steps
    for t in range(num_timesteps):
        # x_k = Lambda_bar * x_{k-1} + B_bar * u_k
        # Note: Lambda_bar is (P,), so it's element-wise multiplied with state (P,)
        # B_bar is (P, H), inputs[t] is (H,)
        # jnp.dot(B_bar, inputs[t]) performs (P, H) @ (H,) -> (P,)
        state = Lambda_bar * state + jnp.dot(B_bar, inputs[t])
        states_sequence.append(state)

    # Stack the states to form a sequence
    # For simplicity, we return the sequence of states.
    # In a full SSM, you would apply C and D matrices to get the actual output y.
    return jnp.stack(states_sequence)
    

def forward_pass_torch(Lambda_bar: torch.Tensor,
                       B_bar: torch.Tensor,
                       inputs: torch.Tensor,
                       initial_state: torch.Tensor,
                       ) -> torch.Tensor:
    """ Performs the forward pass of a discretized diagonalized SSM in PyTorch.

    Args:
        Lambda_bar (complex64): discretized diagonal state matrix (P,)
        B_bar (complex64): discretized input matrix (P, H)
        inputs (complex64): input sequence (T, H)
        initial_state (complex64): initial state (P,)

    Returns:
        outputs (complex64): output sequence (T, P)
    """
    num_timesteps = inputs.shape[0]
    state = initial_state
    states_sequence = []

    for t in range(num_timesteps):
        state = Lambda_bar * state + torch.matmul(B_bar, inputs[t])
        states_sequence.append(state)
    return torch.stack(states_sequence)

# Define system parameters (example values)
# P: number of states
# H: number of input features
P = 4
H = 2

# Continuous-time diagonal state matrix (complex eigenvalues for oscillations)
# Lambda (P,)
# Using complex numbers for illustration of typical SSMs
# Make sure to specify dtype for complex numbers
Lambda_val = np.array([-0.5 + 2j, -0.5 - 2j, -0.1 + 0.5j, -0.1 - 0.5j], dtype=np.complex64)
Lambda_jax = jnp.asarray(Lambda_val)
Lambda_torch = torch.from_numpy(Lambda_val)


# Continuous-time input matrix (P, H)
B_tilde_val = np.random.randn(P, H).astype(np.complex64) + 1j * np.random.randn(P, H).astype(np.complex64)
B_tilde_jax = jnp.asarray(B_tilde_val)
B_tilde_torch = torch.from_numpy(B_tilde_val)


# Discretization step size 
Delta_val = np.random.randn(P).astype(np.float32)
Delta_jax = jnp.asarray(Delta_val)
Delta_torch = torch.from_numpy(Delta_val)


print("--- Original Continuous-Time Parameters ---")
print(f"Lambda:\n{Lambda_val}")
print(f"B_tilde:\n{B_tilde_val}")
print(f"Delta: {Delta_val}\n")

# Discretize the system using JAX
print("--- Discretizing with JAX ---")
Lambda_bar_jax, B_bar_jax = discretize_jax(Lambda_jax, B_tilde_jax, Delta_jax)
print(f"JAX Lambda_bar:\n{Lambda_bar_jax}")
print(f"JAX B_bar:\n{B_bar_jax}\n")

# Discretize the system using PyTorch
print("--- Discretizing with PyTorch ---")
Lambda_bar_torch, B_bar_torch = discretize_torch(Lambda_torch, B_tilde_torch, Delta_torch)
print(f"PyTorch Lambda_bar:\n{Lambda_bar_torch}")
print(f"PyTorch B_bar:\n{B_bar_torch}\n")


# --- Comparison between JAX and PyTorch Discretization ---
print("--- Comparison of Discretized Parameters (JAX vs PyTorch) ---")

# Convert PyTorch tensors to NumPy for comparison
Lambda_bar_torch_np = Lambda_bar_torch.numpy()
B_bar_torch_np = B_bar_torch.numpy()

# Compare Lambda_bar
print("Comparing Lambda_bar:")
all_close_lambda = np.allclose(Lambda_bar_jax, Lambda_bar_torch_np, rtol=1e-5, atol=1e-8)
print(f"Are Lambda_bar (JAX) and Lambda_bar (PyTorch) numerically close? {all_close_lambda}")
if not all_close_lambda:
    print(f"Max absolute difference in Lambda_bar: {np.max(np.abs(Lambda_bar_jax - Lambda_bar_torch_np))}")

# Compare B_bar
print("\nComparing B_bar:")
all_close_b = np.allclose(B_bar_jax, B_bar_torch_np, rtol=1e-5, atol=1e-8)
print(f"Are B_bar (JAX) and B_bar (PyTorch) numerically close? {all_close_b}")
if not all_close_b:
    print(f"Max absolute difference in B_bar: {np.max(np.abs(B_bar_jax - B_bar_torch_np))}")

print("\n--- JAX Forward Pass Simulation ---")
# Generate a dummy input sequence for JAX forward pass
T = 10 # Number of timesteps
inputs_val = np.random.randn(T, H).astype(np.complex64) + 1j * np.random.randn(T, H).astype(np.complex64)
inputs_jax = jnp.asarray(inputs_val)

# Initial state for JAX forward pass
initial_state_val = np.random.randn(P).astype(np.complex64) + 1j * np.random.randn(P).astype(np.complex64)
initial_state_jax = jnp.asarray(initial_state_val)

print(f"Input sequence shape: {inputs_jax.shape}")
print(f"Initial state shape: {initial_state_jax.shape}\n")

# Perform the forward pass with JAX discretized parameters
simulated_states_jax = forward_pass_jax(Lambda_bar_jax, B_bar_jax, inputs_jax, initial_state_jax)

print(f"Simulated states shape (T, P): {simulated_states_jax.shape}")
print(f"First state in sequence:\n{simulated_states_jax[0]}")
print(f"Last state in sequence:\n{simulated_states_jax[-1]}\n")

## 6. Backward Pass (Automatic Differentiation)


### 6.1 JAX Backward Pass

# Define a dummy target for loss calculation
dummy_target_jax = jnp.zeros_like(simulated_states_jax) # Target for the entire state sequence

# Define a loss function for JAX
def loss_fn_jax(Lambda, B_tilde, Delta, inputs, initial_state, target):
    Lambda_bar, B_bar = discretize_jax(Lambda, B_tilde, Delta)
    predictions = forward_pass_jax(Lambda_bar, B_bar, inputs, initial_state)
    # Simple Mean Squared Error loss (complex numbers, so abs(error)^2)
    return jnp.mean(jnp.abs(predictions - target)**2)

# Compute gradients of the loss with respect to Lambda and B_tilde
# We can use jax.value_and_grad to get both the loss value and gradients
loss_value_jax, grads_jax = jax.value_and_grad(loss_fn_jax, argnums=(0, 1))(
    Lambda_jax, B_tilde_jax, Delta_jax, inputs_jax, initial_state_jax, dummy_target_jax
)

print("--- Backward Pass (Gradients with JAX) ---")
print(f"Loss value: {loss_value_jax}")
print(f"Gradient with respect to Lambda:\n{grads_jax[0]}")
print(f"Gradient with respect to B_tilde:\n{grads_jax[1]}")


### 6.2 PyTorch Backward Pass

Lambda_torch_bp = Lambda_torch.clone().detach().requires_grad_(True)
B_tilde_torch_bp = B_tilde_torch.clone().detach().requires_grad_(True)

# Generate a dummy input sequence for PyTorch forward pass
# For simplicity, we'll use the same input values as JAX, converted to torch tensors
inputs_torch = torch.from_numpy(inputs_val)
initial_state_torch = torch.from_numpy(initial_state_val)

# Define a dummy target for loss calculation
dummy_target_torch = torch.zeros_like(initial_state_torch).unsqueeze(0).repeat(T, 1) # (T, P) for states



# Discretize the system with parameters requiring gradients
Lambda_bar_torch_bp, B_bar_torch_bp = discretize_torch(Lambda_torch_bp, B_tilde_torch_bp, Delta_torch)

# Perform forward pass
predictions_torch = forward_pass_torch(Lambda_bar_torch_bp, B_bar_torch_bp, inputs_torch, initial_state_torch)

# Calculate loss
loss_torch = torch.mean(torch.abs(predictions_torch - dummy_target_torch)**2)

# Perform backward pass
loss_torch.backward()

print("--- Backward Pass (Gradients with PyTorch) ---")
print(f"Loss value: {loss_torch.item()}")
print(f"Gradient with respect to Lambda:\n{Lambda_torch_bp.grad}")
print(f"Gradient with respect to B_tilde:\n{B_tilde_torch_bp.grad}")


def loss_fn_torch(Lambda, B_tilde, Delta, inputs, initial_state, target):
    """Loss function for PyTorch implementation."""
    Lambda_bar, B_bar = discretize_torch(Lambda, B_tilde, Delta)
    predictions = forward_pass_torch(Lambda_bar, B_bar, inputs, initial_state)
    return torch.mean(torch.abs(predictions - target)**2)


class TestDiscretizationGradients:
    """Test class for comparing gradients between JAX and PyTorch implementations."""
    
    @pytest.fixture
    def setup_data(self):
        """Setup test data for gradient comparison."""
        # System parameters
        P = 4  # number of states
        H = 2  # number of input features
        T = 10  # number of timesteps
        
        # Continuous-time diagonal state matrix (complex eigenvalues for oscillations)
        Lambda_val = np.array([-0.5 + 2j, -0.5 - 2j, -0.1 + 0.5j, -0.1 - 0.5j], dtype=np.complex64)
        
        # Continuous-time input matrix
        B_tilde_val = np.random.randn(P, H).astype(np.complex64) + 1j * np.random.randn(P, H).astype(np.complex64)
        
        # Discretization step size
        Delta_val = np.random.randn(P).astype(np.float32)
        
        # Input sequence
        inputs_val = np.random.randn(T, H).astype(np.complex64) + 1j * np.random.randn(T, H).astype(np.complex64)
        
        # Initial state
        initial_state_val = np.random.randn(P).astype(np.complex64) + 1j * np.random.randn(P).astype(np.complex64)
        
        # Target (dummy target for loss calculation)
        target_val = np.zeros((T, P), dtype=np.complex64)
        
        return {
            'Lambda_val': Lambda_val,
            'B_tilde_val': B_tilde_val,
            'Delta_val': Delta_val,
            'inputs_val': inputs_val,
            'initial_state_val': initial_state_val,
            'target_val': target_val
        }
    
    def test_discretization_consistency(self, setup_data):
        """Test that discretization produces consistent results between JAX and PyTorch."""
        data = setup_data
        
        # JAX discretization
        Lambda_jax = jnp.asarray(data['Lambda_val'])
        B_tilde_jax = jnp.asarray(data['B_tilde_val'])
        Delta_jax = jnp.asarray(data['Delta_val'])
        
        Lambda_bar_jax, B_bar_jax = discretize_jax(Lambda_jax, B_tilde_jax, Delta_jax)
        
        # PyTorch discretization
        Lambda_torch = torch.from_numpy(data['Lambda_val'])
        B_tilde_torch = torch.from_numpy(data['B_tilde_val'])
        Delta_torch = torch.from_numpy(data['Delta_val'])
        
        Lambda_bar_torch, B_bar_torch = discretize_torch(Lambda_torch, B_tilde_torch, Delta_torch)
        
        # Convert PyTorch tensors to NumPy for comparison
        Lambda_bar_torch_np = Lambda_bar_torch.numpy()
        B_bar_torch_np = B_bar_torch.numpy()
        
        # Assert consistency
        np.testing.assert_allclose(Lambda_bar_jax, Lambda_bar_torch_np, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(B_bar_jax, B_bar_torch_np, rtol=1e-5, atol=1e-8)
    
    def test_forward_pass_consistency(self, setup_data):
        """Test that forward pass produces consistent results between JAX and PyTorch."""
        data = setup_data
        
        # Discretize parameters
        Lambda_jax = jnp.asarray(data['Lambda_val'])
        B_tilde_jax = jnp.asarray(data['B_tilde_val'])
        Delta_jax = jnp.asarray(data['Delta_val'])
        Lambda_bar_jax, B_bar_jax = discretize_jax(Lambda_jax, B_tilde_jax, Delta_jax)
        
        Lambda_torch = torch.from_numpy(data['Lambda_val'])
        B_tilde_torch = torch.from_numpy(data['B_tilde_val'])
        Delta_torch = torch.from_numpy(data['Delta_val'])
        Lambda_bar_torch, B_bar_torch = discretize_torch(Lambda_torch, B_tilde_torch, Delta_torch)
        
        # Forward pass
        inputs_jax = jnp.asarray(data['inputs_val'])
        initial_state_jax = jnp.asarray(data['initial_state_val'])
        states_jax = forward_pass_jax(Lambda_bar_jax, B_bar_jax, inputs_jax, initial_state_jax)
        
        inputs_torch = torch.from_numpy(data['inputs_val'])
        initial_state_torch = torch.from_numpy(data['initial_state_val'])
        states_torch = forward_pass_torch(Lambda_bar_torch, B_bar_torch, inputs_torch, initial_state_torch)
        
        # Assert consistency
        np.testing.assert_allclose(states_jax, states_torch.numpy(), rtol=1e-5, atol=1e-8)
    
    def test_gradient_consistency(self, setup_data):
        """Test that gradients are consistent between JAX and PyTorch implementations."""
        data = setup_data
        
        # JAX gradients
        Lambda_jax = jnp.asarray(data['Lambda_val'])
        B_tilde_jax = jnp.asarray(data['B_tilde_val'])
        Delta_jax = jnp.asarray(data['Delta_val'])
        inputs_jax = jnp.asarray(data['inputs_val'])
        initial_state_jax = jnp.asarray(data['initial_state_val'])
        target_jax = jnp.asarray(data['target_val'])
        
        # Compute JAX gradients
        loss_value_jax, grads_jax = jax.value_and_grad(loss_fn_jax, argnums=(0, 1))(
            Lambda_jax, B_tilde_jax, Delta_jax, inputs_jax, initial_state_jax, target_jax
        )
        
        # PyTorch gradients
        Lambda_torch = torch.from_numpy(data['Lambda_val']).requires_grad_(True)
        B_tilde_torch = torch.from_numpy(data['B_tilde_val']).requires_grad_(True)
        Delta_torch = torch.from_numpy(data['Delta_val'])
        inputs_torch = torch.from_numpy(data['inputs_val'])
        initial_state_torch = torch.from_numpy(data['initial_state_val'])
        target_torch = torch.from_numpy(data['target_val'])
        
        # Compute PyTorch loss and gradients
        loss_torch = loss_fn_torch(Lambda_torch, B_tilde_torch, Delta_torch, 
                                  inputs_torch, initial_state_torch, target_torch)
        loss_torch.backward()
        
        # Extract PyTorch gradients
        grad_lambda_torch = Lambda_torch.grad.numpy()
        grad_b_tilde_torch = B_tilde_torch.grad.numpy()
        
        # Assert loss values are close
        np.testing.assert_allclose(loss_value_jax, loss_torch.item(), rtol=1e-5, atol=1e-8)
        
        # Assert gradients are close
        np.testing.assert_allclose(grads_jax[0].conj(), grad_lambda_torch, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(grads_jax[1].conj(), grad_b_tilde_torch, rtol=1e-5, atol=1e-8)
    
    def test_gradient_numerical_stability(self, setup_data):
        """Test that gradients are numerically stable and reasonable."""
        data = setup_data
        
        # JAX gradients
        Lambda_jax = jnp.asarray(data['Lambda_val'])
        B_tilde_jax = jnp.asarray(data['B_tilde_val'])
        Delta_jax = jnp.asarray(data['Delta_val'])
        inputs_jax = jnp.asarray(data['inputs_val'])
        initial_state_jax = jnp.asarray(data['initial_state_val'])
        target_jax = jnp.asarray(data['target_val'])
        
        loss_value_jax, grads_jax = jax.value_and_grad(loss_fn_jax, argnums=(0, 1))(
            Lambda_jax, B_tilde_jax, Delta_jax, inputs_jax, initial_state_jax, target_jax
        )
        
        # Check that gradients are finite
        assert np.all(np.isfinite(grads_jax[0])), "Lambda gradients contain non-finite values"
        assert np.all(np.isfinite(grads_jax[1])), "B_tilde gradients contain non-finite values"
        
        # Check that loss is finite and positive
        assert np.isfinite(loss_value_jax), "Loss is not finite"
        assert loss_value_jax >= 0, "Loss should be non-negative"
    
    def test_gradient_shapes(self, setup_data):
        """Test that gradients have the correct shapes."""
        data = setup_data
        
        # JAX gradients
        Lambda_jax = jnp.asarray(data['Lambda_val'])
        B_tilde_jax = jnp.asarray(data['B_tilde_val'])
        Delta_jax = jnp.asarray(data['Delta_val'])
        inputs_jax = jnp.asarray(data['inputs_val'])
        initial_state_jax = jnp.asarray(data['initial_state_val'])
        target_jax = jnp.asarray(data['target_val'])
        
        loss_value_jax, grads_jax = jax.value_and_grad(loss_fn_jax, argnums=(0, 1))(
            Lambda_jax, B_tilde_jax, Delta_jax, inputs_jax, initial_state_jax, target_jax
        )
        
        # Check gradient shapes
        assert grads_jax[0].shape == Lambda_jax.shape, f"Lambda gradient shape mismatch: {grads_jax[0].shape} vs {Lambda_jax.shape}"
        assert grads_jax[1].shape == B_tilde_jax.shape, f"B_tilde gradient shape mismatch: {grads_jax[1].shape} vs {B_tilde_jax.shape}"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])


