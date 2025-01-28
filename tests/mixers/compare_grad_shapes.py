import torch
from keras.src.backend.jax.core import while_loop

from deep_ssm.mixers.prnn import quasi_deer_torch
import wandb

wandb.init(mode='disabled')
# Inputs
B, D, T = 32, 128, 256
f = lambda x, h: torch.tanh(x + h)  # Simple nonlinear function
diagonal_derivative = lambda x, h: torch.ones_like(h) * 0.5
initial_state = torch.zeros(B, D)
states_guess = torch.zeros(B, D, T)
inputs = torch.randn(B, D, T)

# Test with last_only=True
last_only=True
clip= False
deer_norm=False
num_iters=10
while_loop=False
states_guess.requires_grad_(True)
output = quasi_deer_torch(f,
                          diagonal_derivative,
                          initial_state,
                          states_guess,
                          inputs,
                          num_iters=num_iters,
                          last_only=last_only,
                          while_loop=while_loop)
loss = output.sum()  # Only the last state contributes
loss.backward()

# Check gradient shape
print("Gradient shape (last_only=True):", states_guess.grad.shape)  # Should be (B, D, T)

# This fails, so should we instead try:
"""
if last_only:
    for _ in range(num_iters - 1):
        with torch.no_grad():
            states_guess, isnan = step(states_guess)  # Detach intermediate iterations
            total_nans += isnan
    # Final iteration retains computation graph
    states_guess, isnan = step(states_guess)
"""
