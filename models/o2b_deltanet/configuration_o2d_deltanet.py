from template import Config
from dataclasses import dataclass

@dataclass
class O2DDeltaNetConfig(Config):
    model_name: str = "o2d_deltanet"
    conv_size: int = 4
    eta: float = 0.1
    use_qk_activation: bool = True
    sync_kv_scale: bool = False