from .bci_models import MambaDecoder, GRUDecoder

all_models = {
  "bci_gru" : GRUDecoder,
  "bci_mamba": MambaDecoder,
}