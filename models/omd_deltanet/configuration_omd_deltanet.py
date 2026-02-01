from template import Config
from dataclasses import dataclass

@dataclass
class OmdDeltaNetConfig(Config):
    model_name: str = "omd_deltanet"
    conv_size: int = 4
    initial_state: bool = False