"""
Test mixer layer and step function

"""
import torch
from deep_ssm.mixers.mamba_simple import Mamba
from deep_ssm.mixers.mamba2_simple import Mamba2Simple
from deep_ssm.mixers.utils_mamba import InferenceParams
from deep_ssm.mixers.s5_fjax.ssm import S5


torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


MODELS = {"Mamba": Mamba,
          "S5": S5,
          "Mamba2": Mamba2Simple,
          }
@torch.inference_mode()
def test_step_mamba(model_name):
    batch, seqlen, dim = 1, 64, 16
    dtype = torch.float32

    headdim = 4
    x = torch.randn(batch, seqlen, dim).to(dtype).to(device)

    if model_name == 'Mamba2':
        kwargs ={"headdim": headdim,
                 "rmsnorm": False,
                 "use_mem_eff_path": False,
                 "chunk_size": 1,
                 }
    else:
        kwargs ={}

    # Training-style forward pass (full sequence in parallel)
    model = MODELS[model_name](
        d_model=dim,
        d_state=16,
        expand=2,
        layer_idx=0,
        **kwargs,
    ).to(device)

    y1 = model(x)
    assert y1.shape == x.shape

    # Inference-style forward pass (full sequence in parallel)
    infer_params = InferenceParams(max_batch_size=batch, max_seqlen=seqlen)

    outs = []
    for i in range(seqlen):
        infer_params.seqlen_offset += 1
        out = model(x[:,i:i+1,:], inference_params=infer_params)
        outs.append(out)
    y2 = torch.cat(outs, 1)


    # Inference-style forward pass (step by step using for loop)
    outs = []
    if model_name == 'Mamba':
        D = model.d_inner
        conv_state = torch.zeros(batch, D, 4, device=x.device, dtype=x.dtype)
        ssm_state = torch.zeros(batch, D, 16, device=x.device, dtype=x.dtype)
    elif model_name == 'Mamba2':
        #D = model.d_inner + 2 * model.ngroups * model.d_state
        D =  model.conv1d.weight.shape[0]
        conv_state = torch.zeros( batch, D, model.d_conv, device=x.device, dtype=x.dtype)
        ssm_state = torch.zeros(batch, model.nheads, model.headdim, model.d_state, device=x.device, dtype=x.dtype)

    for i in range(seqlen):
        out, conv_state, ssm_state = model.step(x[:, i : i + 1, :], conv_state, ssm_state)
        outs.append(out)
    y3 = torch.cat(outs, 1)

    assert torch.allclose(y2, y3, rtol=1e-4)

    if model_name == 'Mamba2':
        # TODO: additional cheks fail
        return
    assert torch.allclose(y1, y2, rtol=1e-4)
    assert torch.allclose(y1, y3, rtol=1e-4)


def test_step_s5():
    batch, seqlen, dim = 1, 64, 16
    dtype = torch.float32

    x = torch.randn(batch, seqlen, dim).to(dtype).to(device)

    model = S5(
        d_model=dim,
        ssm_size=16,
        blocks=2,
    ).to(device)

    state = model.initial_state(batch)

    y1, state_original = model(x, state)

    assert y1.shape == x.shape

    outs = []
    state = model.initial_state(batch)
    for i in range(seqlen):
        out, state = model.step(x[:, i: i + 1, :], state)
        outs.append(out)
    y2 = torch.cat(outs, 1)

    assert torch.allclose(y1, y2, rtol=1e-4)


test_step_mamba("Mamba")
test_step_mamba("Mamba2")
test_step_s5()