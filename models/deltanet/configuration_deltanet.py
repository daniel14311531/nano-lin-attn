from template import Config
from dataclasses import dataclass

@dataclass
class DeltaNetConfig(Config):
    model_name: str = "deltanet"
    conv_size: int = 4
    initial_state: bool = False
    eta: float = 1.0
    use_qk_activation: bool = False