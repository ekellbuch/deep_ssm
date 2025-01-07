"""
Test the pRNN model implementation, it should be able to replicate the GRU implementation.

One full run can be tested with:
  python run.py --config-name="baseline_gru" trainer_cfg.fast_dev_run=1 model_cfg.configs.layer_dim=1
  python run.py --config-name="prnn" trainer_cfg.fast_dev_run=1 model_cfg.configs.layer_dim=1 model_cfg.configs.parallel=false

Comments:
- tolerance is device dependent
"""
import torch
import torch.nn as nn
from deep_ssm.mixers.prnn import pRNN, diagonalize_and_reparameterize
import pytest
import numpy as np

# Detect the appropriate device for execution
try:
    import torch_xla.core.xla_model as xm
    DEVICE = xm.xla_device()  # TPU
    atol = 1e-5
    rtol = 1e-6
except:
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")  # macOS Metal backend
        atol = 1e-5
        rtol = 1e-6
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")  # GPU
        atol = 1e-3 # 1e-4
        rtol = 1e-5
    else:
        DEVICE = torch.device("cpu")  # CPU fallback
        atol = 1e-5
        rtol = 1e-6

torch.set_default_dtype(torch.float32)

class Arguments:
    def __init__(self, 
                input_size=10, 
                hidden_size=10, 
                num_layers=1, 
                batch_first=True, 
                dropout=0, 
                bidirectional=True, 
                num_iters=2,
                method="gru", 
                parallel=False,
                checkpoint=False,
                reparameterize=False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.method = method
        self.parallel = parallel
        self.num_iters = num_iters
        self.checkpoint = checkpoint
        self.reparameterize = reparameterize



def update_name(param_name, suffix=""):
    "change from {weight_ih_l0} to ''forward_cells.0.weight_ih'"
    if "reverse" in param_name:
        new_name = "reverse_cells"
        param_name, _ = param_name.rsplit("_", 1)
    else:
        new_name = "forward_cells" 

    name, layer = param_name.rsplit("_",1)
    layer_name = layer[1:]   
    new_name = f"{new_name}.{layer_name}.{name}{suffix}"
    return new_name


def assign_parameters(model1, model2, update_name_fn=update_name):
    # assign params in model 1 to model2:
    m2_sd = model2.state_dict()
    for name1, p1 in model1.named_parameters():        
        name2 = update_name_fn(name1)
        assert p1.shape ==  m2_sd[name2].shape
        m2_sd[name2] = p1.clone().detach()


    model2.load_state_dict(m2_sd)
    return model2



def compare_parameters(model1, model2,update_name_fn=update_name, grad=False, atol=1e-6):
    # check that parameters are the same:
    model2_params = dict(model2.named_parameters())

    for name1, p1 in model1.named_parameters():
        name2 = update_name_fn(name1)
        p2 = model2_params[name2]
        try:
            if grad:
                assert torch.allclose(p1.grad, p2.grad, atol=atol), f"{name2} did not match"
            else:
                assert torch.allclose(p1, p2, atol=atol), f"{name2} did not match"
        except:
            breakpoint()
            # would fail for backward if not weights are not reassigned
            return False
    return True


def assign_reparameters(model1, model2, update_name_fn=update_name):
    """
    Assign parameters from model1 (GRU) to model2 (pRNN, including reparameterized version).
    Handles reparameterization by transforming weights and biases into Z-space.
    """
    m2_sd = model2.state_dict()

    for name1, p1 in model1.named_parameters():
        name2 = update_name_fn(name1)
        cell_, layer_id, param_name_ = name2.rsplit(".")
        P0 = model2.forward_cells[int(layer_id)].P0
        P0_inv = model2.forward_cells[int(layer_id)].P0

        if model2.bidirectional and "reverse" in name2:
            P0 = model2.reverse_cells[int(layer_id)].P0
            P0_inv = model2.reverse_cells[int(layer_id)].P0

        # Handle reparameterization if applicable
        if "weight_hh" in name1:  # Hidden-to-hidden weights
            # Transform weights into Z-space
            p2 = m2_sd[name2]
            transformed_weights = torch.cat([
                P0_inv @ p1[i * model2.hidden_size:(i + 1) * model2.hidden_size, :] @ P0
                for i in range(3)
            ], dim=0)
            assert p2.shape == transformed_weights.shape, f"Shape mismatch for {name2}"
            m2_sd[name2] = transformed_weights.clone().detach()

        elif "weight_ih" in name1:  # Input-to-hidden weights
            # Transform weights into Z-space
            breakpoint()
            transformed_weights = (P0_inv @ p1.view(3, model2.hidden_size, model2.input_size)).view_as(p1)
            m2_sd[name2] = transformed_weights.clone().detach()

        elif "bias_hh" in name1:  # Hidden-to-hidden biases
            transformed_bias = (P0_inv @ p1.view(3, model2.hidden_size, 1)).view_as(p1)
            m2_sd[name2] = transformed_bias.clone().detach()

        elif "bias_ih" in name1:  # Input-to-hidden biases
            transformed_bias = (P0_inv @ p1.view(3, model2.hidden_size, 1)).view_as(p1)
            m2_sd[name2] = transformed_bias.clone().detach()
        else:
            breakpoint()

    model2.load_state_dict(m2_sd)
    return model2


def compare_reparameters(model1, model2, update_name_fn=update_name, grad=False, atol=1e-6):
    """
    Compare parameters between model1 (GRU) and model2 (pRNN, including reparameterized version).
    Handles reparameterization by reversing transformations for comparison.
    """
    model2_params = dict(model2.named_parameters())

    #breakpoint()
    for name1, p1 in model1.named_parameters():
        name2 = update_name_fn(name1)
        cell_, layer_id, param_name_ = name2.rsplit(".")
        name2 = f"{cell_}.{layer_id}.original_cell.{param_name_}"
        p2 = model2_params[name2]

        """
        P0 = model2.forward_cells[int(layer_id)].P0
        P0_inv = model2.forward_cells[int(layer_id)].P0

        if model2.bidirectional and "reverse" in name2:
            P0 = model2.reverse_cells[int(layer_id)].P0
            P0_inv = model2.reverse_cells[int(layer_id)].P0

        # Handle reparameterization if applicable
        if "weight_hh" in name1:  # Hidden-to-hidden weights
            # Reverse transformation
            original_term = torch.cat([
               P0 @ p2[i * model2.hidden_size:(i + 1) * model2.hidden_size, :] @ P0_inv
                for i in range(3)
            ], dim=0)

        elif "weight_ih" in name1:  # Input-to-hidden weights
            
            original_term = (P0 @ p2.view(3, model2.hidden_size, model2.input_size)).view_as(p1)

        elif "bias_hh" in name1:  # Hidden-to-hidden biases
            original_term = (P0 @ p2.view(3, model2.hidden_size, 1)).view_as(p1)

        elif "bias_ih" in name1:  # Input-to-hidden biases
            original_term = (P0 @ p2.view(3, model2.hidden_size, 1)).view_as(p1)
        else:
            breakpoint()
        """
        if not torch.allclose(p1, p2, atol=atol):
            raise AssertionError(f"Mismatch in {name2}")

        #else:
        #    # Compare parameters directly for non-reparameterized models
        #    tensor_to_compare = p2.grad if grad else p2
        #    if not torch.allclose(p1, tensor_to_compare, atol=atol):
        #        raise AssertionError(f"{name2} did not match")

    return True

seqlens = [32, 64, 256]
batch_sizes = [1, 2, 8]
dims = [2, 8]
bidirectionality =[False, True]
num_layerss = [1, 2, 3]
num_iterss = [10, 20]

seqlens = [2]
batch_sizes = [1]
dims = [1]
bidirectionality = [False]
num_layerss = [1]
num_iterss = [10]

@pytest.mark.parametrize("seqlen", seqlens)
@pytest.mark.parametrize("batch", batch_sizes)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("bidirectional", bidirectionality)
@pytest.mark.parametrize("num_layers", num_layerss)
@pytest.mark.parametrize("num_iters", num_iterss)
def test_prnn(batch, seqlen, dim, bidirectional,num_layers, num_iters):
    
    # Create an instance of Arguments
    cfg = Arguments(input_size=dim,
                    bidirectional=bidirectional,
                    num_layers=num_layers)

    D = 2 if cfg.bidirectional else 1
    hidden_size = cfg.hidden_size * D
    # Pass arguments to pRNN
    torch.manual_seed(42)
    np.random.seed(42)

    x = torch.randn(batch, seqlen, dim).to(DEVICE)
    h0 = torch.randn(D * num_layers, batch, cfg.hidden_size).to(DEVICE)

    # ---------
    # Get model
    torch.manual_seed(42)
    gru_model = nn.GRU(cfg.input_size,
                       cfg.hidden_size, 
                       cfg.num_layers, 
                       batch_first=cfg.batch_first, 
                       dropout=cfg.dropout, 
                       bidirectional=cfg.bidirectional,
                       ).to(DEVICE)

    output1, hidden1 = gru_model(x, h0)

    assert output1.shape == (batch, seqlen, hidden_size)
    assert hidden1.shape == (D * num_layers, batch, cfg.hidden_size)

    # Compute loss and gradients for GRU
    gru_model.zero_grad()
    gru_loss = output1.mean()
    gru_loss.backward()

    # ----------------
    # Compare with sequential model
    h0.grad = None
    torch.manual_seed(42)
    prnn = pRNN(**vars(cfg)).to(DEVICE)
    
    if bidirectional:
        prnn = assign_parameters(gru_model, prnn)

    assert compare_parameters(gru_model, prnn, atol=atol)

    output, hidden = prnn(x, h0)

    assert output.shape == (batch, seqlen, hidden_size), f"prnn output has incorrect dimensions"
    assert hidden.shape == (D * num_layers, batch, cfg.hidden_size), f"prnn hidden var has incorrect dimensions"

    gru_params = sum(p.numel() for p in gru_model.parameters())
    prnn_params = sum(p.numel() for p in prnn.parameters())
    
    assert gru_params == prnn_params, f"gru and prnn parameters do not match"
    assert torch.allclose(output, output1, atol=atol), f"gru and prnn outputs do not match"
    assert torch.allclose(hidden, hidden1, atol=atol), f"gru and prnn hidden vars do not match"

    # Compare gradients
    # Compute loss and gradients for pRNN
    prnn_loss = output.mean()
    prnn_loss.backward()

    assert compare_parameters(gru_model, prnn, grad=True, atol=atol)

    # ----------------
    # Compare with parallel model
    cfg2 = cfg
    cfg2.parallel = True
    cfg2.num_iters = num_iters

    h0.grad = None
    torch.manual_seed(42)
    prnn2 = pRNN(**vars(cfg2)).to(DEVICE)

    if bidirectional:
        prnn2 = assign_parameters(gru_model, prnn2)
    assert compare_parameters(gru_model, prnn2, atol=atol)

    output2, hidden2 = prnn2(x, h0)

    assert output2.shape == (batch, seqlen, hidden_size),  f"prnn_parallel outputs dimension mismatch"
    assert hidden2.shape == (D *num_layers, batch, cfg.hidden_size),  f"prnn_parallel hidden vars dimension mismatch"

    assert torch.allclose(output, output2, atol=atol),  f"gru and prnn_parallel outputs not match"
    assert torch.allclose(hidden, hidden2, atol=atol),  f"gru and prnn_parallel hidden vars not match"

    # Compute loss and gradients for pRNN2
    prnn2_loss = output2.mean()
    prnn2_loss.backward()

    # Compare gradients
    assert compare_parameters(gru_model, prnn2, grad=True, atol=atol)
    
    # ----------------
    # Compare with reparametrized model
    exit()
    cfg2 = cfg
    cfg2.parallel = False
    cfg2.num_iters = num_iters
    cfg2.reparameterize = True
    
    h0.grad = None
    torch.manual_seed(42)
    prnn3 = pRNN(**vars(cfg2)).to(DEVICE)

    #update_name_fn= lambda x: (update_name(x, suffix="_z"))
    if bidirectional:
        #prnn2 = assign_reparameters(gru_model, prnn2, update_name_fn=update_name_fn)
        prnn3 = assign_parameters(gru_model, prnn3)
    #assert compare_reparameters(gru_model, prnn2, atol=atol, update_name_fn=update_name_fn)
    assert compare_reparameters(gru_model, prnn3, atol=atol)

    output3, hidden3 = prnn3(x, h0)

    assert output3.shape == (batch, seqlen, hidden_size),  f"prnn_reparam outputs dimension mismatch"
    assert hidden3.shape == (D *num_layers, batch, cfg.hidden_size),  f"prnn_reparam hidden vars dimension mismatch"

    assert torch.allclose(output, output3, atol=atol),  f"gru and prnn_reparam outputs not match"
    assert torch.allclose(hidden, hidden3, atol=atol),  f"gru and prnn_reparam hidden vars not match"

    # Compute loss and gradients for pRNN2
    prnn2_loss = output3.mean()
    prnn2_loss.backward()

    # Compare gradients
    #assert compare_parameters(gru_model, prnn3, grad=True, atol=atol, update_name_fn=update_name_fn)
    

seqlens = [32]
batch_sizes = [1]
dims = [2]
bidirectionality = [False]
num_layerss = [1]
num_iterss = [10]

@pytest.mark.parametrize("seqlen", seqlens)
@pytest.mark.parametrize("batch", batch_sizes)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("bidirectional", bidirectionality)
@pytest.mark.parametrize("num_layers", num_layerss)
@pytest.mark.parametrize("num_iters", num_iterss)
def compare_gru_cells(batch, seqlen, dim, bidirectional, num_layers, num_iters):
    """
    Compare Reparametrized and Standard GRU
    TODO: fix
    """

    # Create an instance of Arguments
    cfg = Arguments(input_size=dim,
                    bidirectional=bidirectional,
                    num_layers=num_layers)

    D = 2 if cfg.bidirectional else 1
    hidden_size = cfg.hidden_size * D
    # Pass arguments to pRNN
    torch.manual_seed(42)
    np.random.seed(42)

    x0 = torch.randn(batch, seqlen, dim).to(DEVICE)
    h0 = torch.randn(D * num_layers, batch, cfg.hidden_size).to(DEVICE)

    # ---------
    # Get model
    torch.manual_seed(42)
    standard_gru = nn.GRU(cfg.input_size,
                       cfg.hidden_size, 
                       cfg.num_layers, 
                       batch_first=cfg.batch_first, 
                       dropout=cfg.dropout, 
                       bidirectional=cfg.bidirectional,
                       ).to(DEVICE)


    # Linearize the standard GRU at a specific state x0, h0
    reparam_gru = diagonalize_and_reparameterize(standard_gru, dim, hidden_size, x0, h0)

    # Generate a random input sequence
    inputs = torch.randn(seqlen, batch, dim)

    # Initialize hidden states
    h_standard = torch.zeros(batch, hidden_size)
    h_reparam = torch.zeros(batch, hidden_size)

    # Store hidden states over time
    hidden_states_standard = []
    hidden_states_reparam = []

    # Run both GRU cells over the sequence
    for t in range(seqlen):
        input_t = inputs[t]
        h_standard = standard_gru(input_t, h_standard)
        h_reparam = reparam_gru(input_t, h_reparam)

        print(h_standard)
        print(h_reparam)
        breakpoint()
        #hidden_states_standard.append(h_standard.clone().detach())
        #hidden_states_reparam.append(h_reparam.clone().detach())

    # Convert lists to tensors for easier analysis
    #hidden_states_standard = torch.stack(hidden_states_standard, dim=0)  # (T, B, H)
    #hidden_states_reparam = torch.stack(hidden_states_reparam, dim=0)    # (T, B, H)

    # Plot hidden states for comparison
    """
    plt.figure(figsize=(12, 6))
    for i in range(min(3, batch_size)):  # Plot up to 3 examples
        plt.subplot(2, 1, 1)
        plt.plot(hidden_states_standard[:, i, 0].numpy(), label=f"Standard GRU (Batch {i})")
        plt.subplot(2, 1, 2)
        plt.plot(hidden_states_reparam[:, i, 0].numpy(), label=f"Reparam GRU (Batch {i})")
    plt.subplot(2, 1, 1)
    plt.title("Hidden State Dynamics - Standard GRU")
    plt.subplot(2, 1, 2)
    plt.title("Hidden State Dynamics - Reparametrized GRU")
    plt.legend()
    plt.show()
    """
    # Compare Jacobians at a specific state
    print("Jacobian Comparison:")
    h_test = torch.randn(1, hidden_size)
    J_standard = torch.autograd.functional.jacobian(lambda h: standard_gru(x0, h), h_test)
    J_reparam = torch.autograd.functional.jacobian(lambda h: reparam_gru(x0, h), h_test)

    print("Standard GRU Jacobian:")
    print(J_standard.squeeze())
    print("Reparametrized GRU Jacobian:")
    print(J_reparam.squeeze())


if __name__ == "__main__":
    test_prnn()

    #seqlens = [32]
    #batch_sizes = [1]
    #dims = [2]
    #bidirectionality = [False]
    #num_layerss = [2]
    #num_iterss = [10]
    #for (batch, seqlen, dim, bidirectional, num_layers, num_iters) in zip(batch_sizes, seqlens, dims, bidirectionality, num_layerss, num_iterss):
    #    compare_gru_cells(batch, seqlen, dim, bidirectional, num_layers, num_iters)