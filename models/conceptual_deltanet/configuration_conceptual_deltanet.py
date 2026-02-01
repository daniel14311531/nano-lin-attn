from template import Config
from dataclasses import dataclass

@dataclass
class ConceptualDeltaNetConfig(Config):
    model_name: str = "ConceptualDeltaNet"
    conv_size: int = 4
    initial_state: bool = False
    eta: float = 1e5