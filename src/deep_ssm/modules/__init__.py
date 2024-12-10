from .module_bci import BCIDecoder
from .module_parity import ParityDecoder
all_modules = {
    "bci": BCIDecoder,
    "parity": ParityDecoder
}
