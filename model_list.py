from models.gpt2 import GPT
from models.deltanet import DeltaNetLM

model_map = {
    "gpt2": GPT,
    "deltanet": DeltaNetLM,
}

def get_model(model_name):
    return model_map.get(model_name, None)