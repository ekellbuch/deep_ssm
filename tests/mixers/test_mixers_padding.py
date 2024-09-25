"""
Test mixer layer padding

"""
import torch
from deep_ssm.mixers.mamba_simple import Mamba
from deep_ssm.mixers.mamba2_simple import Mamba2Simple
from deep_ssm.mixers.s5_fjax.ssm import S5Layer as S5
from deep_ssm.mixers.s4 import S4Block as S4
from torch.nn import functional as F
from deep_ssm.mixers.mamba_extra import BidirectionalMamba, MambaWrapper



from absl.testing import absltest
from absl.testing import parameterized


torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


MODELS = {"Mamba": Mamba,
          "S5": S5,
          "Mamba2": Mamba2Simple,
          "S4": S4,
          "MambaBi": MambaWrapper,
          "MambaBi_v0": BidirectionalMamba
          }



class TestMixerInference(parameterized.TestCase):
    @parameterized.named_parameters(
        ("Mamba", "Mamba", False),
        ("Mamba2", "Mamba2", False),
        ("S4", "S4", False),
        ("S5", "S5", False),
        ("S4 Bidirectional", "S4", True),
        ("S5 Bidirectional", "S5", True),
        #("MambaBi_v0", "MambaBi_v0", True),
        ("MambaBi", "MambaBi", True)
    )
    def test_padded_sequence(self, model_name, bidirectional):
        """
        Test forward step in padded mamba models
        """

        batch, seqlen, dim = 1, 64, 1
        dtype = torch.float32

        headdim = 1
        pad_len = 5
        x = torch.randn(batch, seqlen, dim).to(dtype).to(device)
        padded_x = F.pad(x, (0, 0, 0, pad_len)).to(dtype)

        assert x.shape == padded_x[:,pad_len:].shape
        if 'Mamba' in model_name:
            args ={"d_model": dim,
                    "layer_idx": 0,
                     "expand": 2,
                     "d_state": 16,
                     }

        elif model_name == 'S5':
            args ={"d_model": dim,
                     "ssm_size": 16,
                    "blocks": 1,
                     }
        elif model_name == 'S4':
            args ={"d_model": dim,
                  "d_state": 16,
                  "transposed": False
                   }
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

        # Training-style forward pass (full sequence in parallel)
        model = MODELS[model_name](
            **args,
            **kwargs,
        ).to(device)

        model.eval()

        with torch.no_grad():
          if 'Mamba' in model_name:
            y1 = model(x)
            y1_padded= model(padded_x)

          else:
            y1, _ = model(x)
            y1_padded, _ = model(padded_x)

          assert y1.shape == x.shape
          assert y1_padded.shape == padded_x.shape

          unpadded_y1 = y1_padded[:, :-pad_len, :]

          self.assertTrue(torch.allclose(unpadded_y1, y1, rtol=1e-5, atol=1e-5))



if __name__ == '__main__':
  absltest.main()