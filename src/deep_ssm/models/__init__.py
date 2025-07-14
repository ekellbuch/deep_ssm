from .bci_models import MambaDecoder, GRUDecoder, SashimiDecoder, S5Decoder, BiTDecoder, BiTMambaDecoder, BiTS5Decoder, BiTMamba2Decoder

all_models = {
  "bci_gru" : GRUDecoder,
  "bci_mamba": MambaDecoder,
  "bci_sashimi": SashimiDecoder,
  "bci_s5": S5Decoder,
  "bci_bit": BiTDecoder,
  "bci_bit_mamba": BiTMambaDecoder,
  "bci_bit_s5": BiTS5Decoder,
  "bci_bit_mamba2": BiTMamba2Decoder,
}