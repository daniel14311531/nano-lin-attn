from models.gpt2 import GPT, GPT2Config
from models.deltanet import DeltaNetLM, DeltaNetConfig
from models.omd_deltanet import OmdDeltaNetLM, OmdDeltaNetConfig
from models.conceptual_deltanet import ConceptualDeltaNetLM, ConceptualDeltaNetConfig
from models.o2b_deltanet import O2DDeltaNetLM, O2DDeltaNetConfig

model_map = {
    "gpt2": GPT,
    "deltanet": DeltaNetLM,
    "omd_deltanet": OmdDeltaNetLM,
    "conceptual_deltanet": ConceptualDeltaNetLM,
    "o2d_deltanet": O2DDeltaNetLM,
}

config_map = {
    "gpt2": GPT2Config,
    "deltanet": DeltaNetConfig,
    "omd_deltanet": OmdDeltaNetConfig,
    "conceptual_deltanet": ConceptualDeltaNetConfig,
    "o2d_deltanet": O2DDeltaNetConfig,
}

def get_model(model_name):
    return model_map.get(model_name, None)

def get_config(model_name):
    return config_map.get(model_name, None)