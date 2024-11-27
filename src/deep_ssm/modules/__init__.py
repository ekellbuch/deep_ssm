from .module_bci import BCIDecoder

all_modules = {
    "bci": BCIDecoder,
    "bci_mamba": MambaDecoder,
    "bci_sashimi": SashimiDecoder,
}
