#from .bci_models import MambaDecoder, GRUDecoder, SashimiDecoder, pRNNDecoder
from . import bci_models
from . import parity_models

all_models = {
  "bci_gru" : bci_models.GRUDecoder,
  "bci_mamba": bci_models.MambaDecoder,
  "bci_sashimi": bci_models.SashimiDecoder,
  'bci_prnn': bci_models.pRNNDecoder,
  "parity_gru": parity_models.GRUDecoder
}
