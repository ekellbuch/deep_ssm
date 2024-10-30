import torch


from deep_ssm.mixers.mamba_simple import Mamba
from deep_ssm.mixers.mamba_vsimple import MambaVsimple


device = 'cuda' if torch.cuda.is_available() else 'cpu'


batch, length, dim = 2, 512, 16

x = torch.randn(batch, length, dim).to(device)
x.requieres_grad = True

#  CUDA Model

torch.manual_seed(1)

configs =  {
  "d_model": dim,  # Model dimension d_model
  "d_state" :16,  # SSM state expansion factor
  "d_conv" : 4,  # Local convolution width
  "expand" :2,  # Block expansion factor
}


model_cuda = Mamba(**configs).to(device)
y_cuda = model_cuda(x)

print(sum([p.numel() for p in model_cuda.parameters()]))
print(y_cuda.shape)

# Test local model

torch.manual_seed(1)

model = MambaVsimple(**configs).to(device)

y_pscan = model(x)

print(sum([p.numel() for p in model.parameters()]))
print(y_pscan.shape)

#  forward #
print("Forward pass check")
print(torch.allclose(y_cuda, y_pscan, rtol=0.1))

#  backward #
print("Backward pass check")
J_cuda = y_cuda.sum()
J_cuda.backward()

J_pscan = y_pscan.sum()
J_pscan.backward()

for (name1, param1), (name2, param2) in zip(model_cuda.named_parameters(), model.named_parameters()):
  if param1.grad is not None and param2.grad is not None:  # Ensure gradients exist
    grad_close = torch.allclose(param1.grad, param2.grad, rtol=0.01)
    print(f"Gradients close for {name1} and {name2}: {grad_close}")
  else:
    print(f"No gradient for {name1} or {name2}")