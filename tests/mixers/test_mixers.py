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


torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


MODELS = {"Mamba": Mamba,
          "S5": S5,
          "Mamba2": Mamba2Simple,
          "S4": S4,
          "MambaBi": MambaWrapper,
          }


class TestMixerInference(parameterized.TestCase):

    @parameterized.named_parameters(
        ("S5", "S5", False),
        ("S5 bidirectional", "S5", True),
        ("S4", "S4", False),
        ("S4 bidir#ectional", "S4", True),
        ("Mamba", "Mamba", False),
        ("MambaBi", "MambaBi", True),
        #("Mamba2", "Mamba2", False),
    )
    def test_step_ssm(self, model_name, bidirectional):
        """
        Test forward and step in s4/s5 models
        """
        batch, seqlen, dim = 1, 64, 16
        dtype = torch.float32

        ssm_size   = 8
        headdim = 1
        x = torch.randn(batch, seqlen, dim).to(dtype).to(device)

        if 'Mamba' in model_name:
            args ={"d_model": dim,
                    "layer_idx": 0,
                    "expand": 1,
                    "d_state": ssm_size,
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
        ).to(device)

        model.eval()

        if 'Mamba' in model_name:
            y1 = model(x)
        else:
            y1, new_state = model(x)
            #assert new_state.shape == state_dim

            #if not bidirectional:  assert new_state.shape == state_dim

        assert y1.shape == x.shape

        # Evaluate step unless bidirectional:
        if bidirectional:
            return

        # TODO: (update legacy s4 code) if model is s4, need to setup before step:

        if model_name == 'Mamba':
            D = model.d_inner
            conv_state = torch.zeros(batch, D, model.d_conv, device=x.device, dtype=x.dtype)
            ssm_state = torch.zeros(batch, D, ssm_size, device=x.device, dtype=x.dtype)
        elif model_name == 'Mamba2':
            #D = model.d_inner + 2 * model.ngroups * model.d_state
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
            self.assertTrue(torch.allclose(y2, y1, rtol=1e-0))
        else:
            self.assertTrue(torch.allclose(y2, y1, rtol=1e-5, atol=1e-5))

        if 'Mamba' in model_name:
            # Inference-style forward pass (full sequence in parallel)
            infer_params = InferenceParams(max_batch_size=batch, max_seqlen=seqlen)

            outs = []
            for i in range(seqlen):
                infer_params.seqlen_offset += 1
                out = model(x[:, i:i + 1, :], inference_params=infer_params)
                outs.append(out)
            y3 = torch.cat(outs, 1)

            self.assertTrue(torch.allclose(y2, y3, rtol=1e-5, atol=1e-5))


if __name__ == '__main__':
  absltest.main()