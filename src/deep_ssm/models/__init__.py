from .bci_models import GRUDecoder, MinRNNDecoder

all_models = {
  "bci_gru" : GRUDecoder,
  "bci_minrnn" : MinRNNDecoder
}
