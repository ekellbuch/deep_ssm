from .module_bci import BCIDecoder
from .module_icl import ICLRegression
from .module_state_tracking import StateTrackingDecoder
all_modules = {
    "bci": BCIDecoder,
    "state_tracking": StateTrackingDecoder,
    "icl": ICLRegression,
}