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
        atol = 1e-3
        rtol = 1e-5
    else:
        DEVICE = torch.device("cpu")  # CPU fallback
        atol = 1e-5
        rtol = 1e-6

torch.manual_seed(42)

MODELS = {"Mamba": Mamba,
          "Mamba2": Mamba2Simple,
          "MambaBi": MambaWrapper,
          }

class TestMixerInference(parameterized.TestCase):

    @parameterized.named_parameters(
        ("Mamba_compare_complex", "Mamba", False),
    )
    def test_step_ssm(self, model_name, bidirectional):
        """
        Test forward and step in s4/s5 models, compare Mamba with and without complex.
        """
        batch, seqlen, dim = 1, 32, 16
        dtype = torch.float32
        dtype_complex = torch.complex64

        ssm_size   = 8
        x = torch.randn(batch, seqlen, dim).to(dtype).to(DEVICE)

        outputs = {}
        for as_complex in [False, True]:
            ssm_size_adj = ssm_size #// 2 if as_complex else ssm_size
            args = {
                "d_model": dim,
                "layer_idx": 0,
                "expand": 1,
                "d_state": ssm_size_adj,
                "as_complex": as_complex,
            }

            model = MODELS[model_name](
                **args
            ).to(DEVICE)

            model.eval()

            # Forward pass
            y1 = model(x)

            # Evaluate step unless bidirectional:
            if bidirectional:
                continue

            D = model.d_inner
            conv_state = torch.zeros(batch, D, model.d_conv, device=x.device, dtype=x.dtype)
            ssm_state = torch.zeros(batch, D, ssm_size_adj//2, device=x.device, dtype=x.dtype)


            outs = []
            for i in range(seqlen):
                out, conv_state, ssm_state = model.step(x[:, i:i+1, :], conv_state, ssm_state)
                outs.append(out)
            y2 = torch.stack(outs, 1).squeeze(-2)

            #y2 = torch.zeros_like(y1)
            assert y1.shape == x.shape
            assert y2.shape == y1.shape

            print(as_complex, y1.max())
            outputs["complex" if as_complex else "real"] = y1

        # Compare outputs from real and complex configurations
        y_real = outputs["real"]
        y_complex = outputs["complex"]

        self.assertTrue(
            torch.allclose(y_real, y_complex, rtol=rtol, atol=1e-2),
            f"Outputs differ between real and complex configurations.\n"
            f"Max difference: {(y_real - y_complex).abs().max()}"
        )

if __name__ == '__main__':
    absltest.main()
