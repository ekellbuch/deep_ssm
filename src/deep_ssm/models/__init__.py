from .bci_models import MambaDecoder, GRUDecoder, SashimiDecoder, S5Decoder

all_models = {
  "bci_gru" : GRUDecoder,
  "bci_mamba": MambaDecoder,
  "bci_sashimi": SashimiDecoder,
  "bci_s5": S5Decoder,
}