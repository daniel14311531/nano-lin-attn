from template import Config
from dataclasses import dataclass

@dataclass
class OmdDeltaNetConfig(Config):
    model_name: str = "omd_deltanet"
    conv_size: int = 4
    initial_state: bool = False
    eta: float = 0.1
    use_qk_activation: bool = True
    sync_kv_scale: bool = False