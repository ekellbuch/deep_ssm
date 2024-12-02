from .bci_models import MambaDecoder, GRUDecoder, SashimiDecoder, pRNNDecoder

all_models = {
  "bci_gru" : GRUDecoder,
  "bci_mamba": MambaDecoder,
  "bci_sashimi": SashimiDecoder,
  'bci_prnn': pRNNDecoder}
