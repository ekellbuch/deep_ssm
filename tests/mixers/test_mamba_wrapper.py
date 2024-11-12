"""
Test that embedding mamba in mixer module gives the same results

"""
import torch
from deep_ssm.mixers.mamba_simple import Mamba
from deep_ssm.mixers.mamba2_simple import Mamba2Simple
from deep_ssm.mixers.s5_fjax.ssm import S5Layer as S5
from deep_ssm.mixers.s4 import S4Block as S4
from torch.nn import functional as F
from deep_ssm.mixers.mamba_extra import MambaWrapper, MixerModel



from absl.testing import absltest
from absl.testing import parameterized


torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


MODELS = {"Mamba": Mamba,
          }



class TestMixerInference(parameterized.TestCase):
    @parameterized.named_parameters(
        ("Mamba", "Mamba", False),
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
        means = x.mean(1, keepdim=True).detach()  # B x 1 x D
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)  # B x 1 x D
        x /= stdev

        padded_x = F.pad(x, (0, 0, 0, pad_len)).to(dtype)

        assert x.shape == padded_x[:,pad_len:].shape
        kwargs = {}
        if 'Mamba' in model_name:
            args ={"d_model": dim,
                    "layer_idx": 0,
                     "expand": 2,
                     "d_state": 16,
                     }

        if bidirectional:
          kwargs['bidirectional'] = bidirectional

        # Training-style forward pass (full sequence in parallel)
        torch.manual_seed(42)
        model = MODELS[model_name](**args, **kwargs).to(device)

        model.eval()

        with torch.no_grad():
          y1 = model(x)
          y1_padded= model(padded_x)

        torch.manual_seed(42)
        model2 = MambaWrapper(
            **args,
            **kwargs,
        ).to(device)

        model2.eval()

        with torch.no_grad():
          if 'Mamba' in model_name:
            y2 = model2(x)
            y2_padded= model2(padded_x)

        # Check that two models are the same given same parameters:
        for p1, p2 in zip(model.named_parameters(), model2.mamba_fwd.named_parameters()):
            assert p1[0] == p2[0]
            self.assertTrue(torch.allclose(p1[1], p2[1], rtol=1e-5, atol=1e-5))
        self.assertTrue(torch.allclose(y2, y1, rtol=1e-5, atol=1e-5))
        self.assertTrue(torch.allclose(y2_padded, y1_padded, rtol=1e-5, atol=1e-5))




if __name__ == '__main__':
  absltest.main()