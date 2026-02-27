from template import Config
from dataclasses import dataclass

@dataclass
class ConceptualDeltaNetConfig(Config):
    model_name: str = "ConceptualDeltaNet"
    conv_size: int = 4
    initial_state: bool = False
    eta: float = 0.1
    use_qk_activation: bool = True
    sync_kv_scale: bool = False