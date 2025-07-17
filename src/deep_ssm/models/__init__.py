import importlib

_all_model_paths = {
    "bci_gru": ("deep_ssm.models.bci_models", "GRUDecoder"),
    "bci_mamba": ("deep_ssm.models.bci_models", "MambaDecoder"),
    "bci_sashimi": ("deep_ssm.models.bci_models", "SashimiDecoder"),
    "bci_s5": ("deep_ssm.models.bci_models", "S5Decoder"),
    "bci_bit": ("deep_ssm.models.bci_models", "BiTDecoder"),
    "bci_bit_mamba": ("deep_ssm.models.bci_models", "BiTMambaDecoder"),
    "bci_bit_s5": ("deep_ssm.models.bci_models", "BiTS5Decoder"),
    "bci_bit_mamba2": ("deep_ssm.models.bci_models", "BiTMamba2Decoder"),
    "bci_vit_sashimi": ("deep_ssm.models.bci_models", "BiTSashimiDecoder"),
    "s5_base": ("deep_ssm.models.baseline_model", "S5Model"),
    "s4_base": ("deep_ssm.models.baseline_model", "S4Model"),
}

class ModelRegistry:
    def __getitem__(self, key):
        if key not in _all_model_paths:
            raise KeyError(f"Model '{key}' not found in registry.")
        module_path, class_name = _all_model_paths[key]
        module = importlib.import_module(module_path)
        return getattr(module, class_name)  # Return class, not instance

all_models = ModelRegistry()
