#from .bci_models import MambaDecoder, GRUDecoder, SashimiDecoder, pRNNDecoder
from . import bci_models
from . import state_tracking_models

all_models = {
  "bci_gru" : bci_models.GRUDecoder,
  "bci_mamba": bci_models.MambaDecoder,
  "bci_sashimi": bci_models.SashimiDecoder,
  'bci_prnn': bci_models.pRNNDecoder,
  "state_tracking_gru": state_tracking_models.GRUDecoder,
  "state_tracking_prnn": state_tracking_models.pRNNDecoder,
  "state_tracking_mamba": state_tracking_models.MambaDecoder
}
