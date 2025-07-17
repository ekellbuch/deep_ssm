
from .module_bci import BCIDecoder, BCIEncoder
from .module_sequential import SequentialModule

all_modules = {"bci": BCIDecoder,
               "bci_encoder": BCIEncoder,
               "sequential": SequentialModule}