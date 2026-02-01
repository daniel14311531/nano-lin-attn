from models.gpt2 import GPT
from models.deltanet import DeltaNetLM
from models.omd_deltanet import OmdDeltaNetLM
from models.conceptual_deltanet import ConceptualDeltaNetLM

model_map = {
    "gpt2": GPT,
    "deltanet": DeltaNetLM,
    "omd_deltanet": OmdDeltaNetLM,
    "conceptual_deltanet": ConceptualDeltaNetLM,
}

def get_model(model_name):
    return model_map.get(model_name, None)