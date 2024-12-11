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
from deep_ssm.mixers.prnn import pRNN
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
                checkpoint=False):
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



def update_name(param_name):
    "change from {weight_ih_l0} to ''forward_cells.0.weight_ih'"
    if "reverse" in param_name:
        new_name = "reverse_cells"
        param_name, _ = param_name.rsplit("_", 1)
    else:
        new_name = "forward_cells" 

    name, layer = param_name.rsplit("_",1)
    layer_name = layer[1:]   
    new_name = f"{new_name}.{layer_name}.{name}"
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

    

seqlens = [32, 64, 256]
batch_sizes = [1, 2, 8]
dims = [2, 8]
bidirectionality =[False, True]
num_layerss = [1, 2, 3]


#seqlens = [32]
#batch_sizes = [1]
#dims = [2]
#bidirectionality = [False]
#num_layerss = [2]

@pytest.mark.parametrize("seqlen", seqlens)
@pytest.mark.parametrize("batch", batch_sizes)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("bidirectional", bidirectionality)
@pytest.mark.parametrize("num_layers", num_layerss)
def test_prnn(batch, seqlen, dim, bidirectional,num_layers):
    
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

    # ---------
    # Get model
    torch.manual_seed(42)
    gru_model = nn.GRU(cfg.input_size,
                       cfg.hidden_size, 
                       cfg.num_layers, 
                       batch_first=cfg.batch_first, 
                       dropout=cfg.dropout, 
                       bidirectional=cfg.bidirectional).to(DEVICE)

    output1, hidden1 = gru_model(x)

    assert output1.shape == (batch, seqlen, hidden_size)
    assert hidden1.shape == (D * num_layers, batch, cfg.hidden_size)

    # Compute loss and gradients for GRU
    gru_model.zero_grad()
    gru_loss = output1.mean()
    gru_loss.backward()

    # -------------
    # Get model

    torch.manual_seed(42)
    prnn = pRNN(**vars(cfg)).to(DEVICE)
    
    if bidirectional:
        prnn = assign_parameters(gru_model, prnn)

    assert compare_parameters(gru_model, prnn, atol=atol)

    output, hidden = prnn(x)

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

    cfg2 = cfg
    cfg2.parallel = True
    cfg2.num_iters = 10

    # -----
    torch.manual_seed(42)
    prnn2 = pRNN(**vars(cfg2)).to(DEVICE)

    if bidirectional:
        prnn2 = assign_parameters(gru_model, prnn2)
    assert compare_parameters(gru_model, prnn2, atol=atol)

    output2, hidden2 = prnn2(x)

    assert output2.shape == (batch, seqlen, hidden_size),  f"prnn_parallel outputs dimension mismatch"
    assert hidden2.shape == (D *num_layers, batch, cfg.hidden_size),  f"prnn_parallel hidden vars dimension mismatch"

    assert torch.allclose(output, output2, atol=atol),  f"gru and prnn_parallel outputs not match"
    assert torch.allclose(hidden, hidden2, atol=atol),  f"gru and prnn_parallel hidden vars not match"

    # Compute loss and gradients for pRNN2
    prnn2_loss = output2.mean()
    prnn2_loss.backward()

    # Compare gradients
    assert compare_parameters(gru_model, prnn2, grad=True, atol=atol)



if __name__ == "__main__":
    test_prnn()
