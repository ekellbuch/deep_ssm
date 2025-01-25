#from .bci_models import MambaDecoder, GRUDecoder, SashimiDecoder, pRNNDecoder
from . import bci_models
from . import state_tracking_models
from .icl_model import pRNNRegressor

all_models = {
  "bci_gru" : bci_models.GRUDecoder,
  "bci_mamba": bci_models.MambaDecoder,
  "bci_sashimi": bci_models.SashimiDecoder,
  'bci_prnn': bci_models.pRNNDecoder,
  "state_tracking_gru": state_tracking_models.GRUDecoder,
  "state_tracking_prnn": state_tracking_models.pRNNDecoder,
  "state_tracking_mamba": state_tracking_models.MambaDecoder,
  "state_tracking_linear": state_tracking_models.LinearRNNDecoder,
  "state_tracking_lru": state_tracking_models.LRUDecoder,
  "icl_reg_model": pRNNRegressor,
}
