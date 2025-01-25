"""
Test mixer layer and step function

"""
import torch
from deep_ssm.mixers.mamba_simple import Mamba
from deep_ssm.mixers.mamba2_simple import Mamba2Simple
from deep_ssm.mixers.utils_mamba import InferenceParams
from deep_ssm.mixers.s5_fjax.ssm import S5Layer as S5
from deep_ssm.mixers.s4 import S4Block as S4
from deep_ssm.mixers.mamba_extra import MambaWrapper


from absl.testing import absltest
from absl.testing import parameterized


# Device selection logic
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


torch.manual_seed(42)


MODELS = {"Mamba": Mamba,
        #   "S5": S5,
          "Mamba2": Mamba2Simple,
          # "S4": S4,
          "MambaBi": MambaWrapper,
          }


class TestMixerInference(parameterized.TestCase):

    @parameterized.named_parameters(
        # ("Mamba", "Mamba", False, False),
        # ("Mamba_real", "Mamba", False, False),
        ("Mamba_complex", "Mamba", False, True),
    )
    def test_step_ssm(self, model_name, bidirectional,as_complex):
        """
        Test forward and step in s4/s5 models
        """
        batch, seqlen, dim = 1, 32, 16
        dtype = torch.float32
        dtype_complex = torch.complex64

        ssm_size   = 8
        headdim = 1
        x = torch.randn(batch, seqlen, dim).to(dtype).to(DEVICE)

        if 'Mamba' in model_name:
            #if as_complex:
            #    ssm_size = ssm_size //2
            args ={"d_model": dim,
                    "layer_idx": 0,
                    "expand": 1,
                    "d_state": ssm_size,
                    "as_complex": as_complex,
                    }

        elif model_name == 'S5':
            args ={"d_model": dim,
                    "ssm_size": ssm_size,
                    "blocks": 1,
                    }
            state_dim = (batch, ssm_size)

        elif model_name == 'S4':
            args ={"d_model": dim,
                  "d_state": ssm_size,
                  "transposed": False
                   }
            state_dim = (batch, dim, ssm_size)

        if model_name == 'Mamba2':
            kwargs ={"headdim": headdim,
                     "rmsnorm": False,
                     "use_mem_eff_path": False,
                     "chunk_size": 1,
                     }
        else:
            kwargs ={}

        if bidirectional:
            kwargs['bidirectional'] = bidirectional

        model = MODELS[model_name](
            **args,
            **kwargs,
        ).to(DEVICE)

        model.eval()

        if 'Mamba' in model_name:
            y1 = model(x)
        else:
            y1, new_state = model(x)
            # assert new_state.shape == state_dim

            # if not bidirectional:  assert new_state.shape == state_dim

        assert y1.shape == x.shape

        # Evaluate step unless bidirectional:
        if bidirectional:
            return

        # TODO: (update legacy s4 code) if model is s4, need to setup before step:

        if model_name == 'Mamba':
            D = model.d_inner
            conv_state = torch.zeros(batch, D, model.d_conv, device=x.device, dtype=x.dtype)
            if as_complex:
                ssm_state = torch.zeros(batch, D, ssm_size//2, device=x.device, dtype=x.dtype)
            else:
                ssm_state = torch.zeros(batch, D, ssm_size, device=x.device, dtype=x.dtype)
        elif model_name == 'Mamba2':
            # D = model.d_inner + 2 * model.ngroups * model.d_state
            D =  model.conv1d.weight.shape[0]
            conv_state = torch.zeros( batch, D, model.d_conv, device=x.device, dtype=x.dtype)
            ssm_state = torch.zeros(batch, model.nheads, model.headdim, model.d_state, device=x.device, dtype=x.dtype)
        else:
            if model_name == 'S4':
                # Initialize step module
                mode = 'diagonal'
                assert mode in ['dense', 'diagonal', 'linear']
                for module in model.modules():
                    if hasattr(module, '_setup_step'): module._setup_step(mode=mode)

            new_state = model.initial_state(batch)

        # TODO: check step for mamba2
        outs = []
        for i in range(seqlen):
            if 'Mamba' in model_name:
                out, conv_state, ssm_state = model.step(x[:,i:i+1,:], conv_state, ssm_state)
            else:
                out, new_state = model.step(x[:,i], state=new_state)
            outs.append(out)
        y2 = torch.stack(outs, 1).squeeze(-2)
        assert y2.shape == y1.shape

        # TODO: check tolerance gap for s4
        if model_name == 'S4':
            self.assertTrue(torch.allclose(y2, y1, rtol=rtol, atol=atol))
        else:
            self.assertTrue(
                torch.allclose(y2, y1, rtol=rtol, atol=1e-2),
                f"{model_name} failed" + f"dtype is {x.dtype}"
                f"y2 max: {y2.abs().max()}"
                + f"y1 max: {y1.abs().max()}"
                + f"difference is {(y2 - y1).abs().max()}",
            )

        if 'Mamba' in model_name:
            # Inference-style forward pass (full sequence in parallel)
            infer_params = InferenceParams(max_batch_size=batch, max_seqlen=seqlen)

            outs = []
            for i in range(seqlen):
                infer_params.seqlen_offset += 1
                out = model(x[:, i:i + 1, :], inference_params=infer_params)
                outs.append(out)
            y3 = torch.cat(outs, 1)

            self.assertTrue(torch.allclose(y2, y3, rtol=rtol, atol=atol))


if __name__ == '__main__':
  absltest.main()
