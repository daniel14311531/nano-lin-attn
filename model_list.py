from models.gpt2 import GPT
from models.deltanet import DeltaNetLM
from models.omd_deltanet import OmdDeltaNetLM

model_map = {
    "gpt2": GPT,
    "deltanet": DeltaNetLM,
    "omd_deltanet": OmdDeltaNetLM,
}

def get_model(model_name):
    return model_map.get(model_name, None)