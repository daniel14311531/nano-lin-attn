from template import Config
from dataclasses import dataclass

@dataclass
class GPT2Config(Config):
    model_name: str = "gpt2"