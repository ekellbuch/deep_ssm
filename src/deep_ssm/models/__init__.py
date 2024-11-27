from .bci_models import MambaDecoder, GRUDecoder, SashimiDecoder, TCNDecoder

all_models = {
  "bci_gru" : GRUDecoder,
  "bci_tcn" : TCNDecoder,
  "bci_mamba": MambaDecoder,
  "bci_sashimi": SashimiDecoder
}