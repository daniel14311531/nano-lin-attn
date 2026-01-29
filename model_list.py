from models.gpt2 import GPT2Config, GPT

model_map = {
    "gpt2": (GPT2Config, GPT),
}

def get_model(model_name):
    return model_map.get(model_name, (None, None))