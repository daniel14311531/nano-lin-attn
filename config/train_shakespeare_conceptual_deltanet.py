# train a miniature shakespeare model
# good for debugging and playing on macbooks and such
from utils.config import *
from models import *

# I/O
io_config = IOConfig(
    out_dir = 'out-shakespeare',
    eval_interval = 50, # keep frequent because we'll overfit
    eval_iters = 50,
    log_interval = 5, # don't print too too often
    always_save_checkpoint = False, # we expect to overfit on this small dataset, so only save when val improves
    init_from = 'scratch',
)

# wandb
wandb_config = WandbConfig(
    wandb_log= True,
    wandb_project= 'shakespeare',
    wandb_run_name= 'conceptual_deltanet', # [TODO]
)

# data
data_config = DataConfig(
    dataset = 'shakespeare',
    gradient_accumulation_steps = 1,
    batch_size = 8,
    block_size = 256, # context of up to 256 previous characters
)

# baby GPT model :)
model_config_instance = ConceptualDeltaNetConfig(
    model_name = 'conceptual_deltanet',
    n_layer = 6,
    n_head = 6,
    n_embd = 384,
    dropout = 0.2,
    block_size = 256,
    bias = False,

    conv_size=4,
    initial_state=False,
    eta=0.0001,
)

# optimizer
optimizer_config = OptimizerConfig(
    learning_rate = 5e-4, # with baby networks can afford to go a bit higher
    max_iters = 300,
    lr_decay_iters = 300, # make equal to max_iters usually
    min_lr = 1e-4, # learning_rate / 10 usually
    beta2 = 0.99, # make a bit bigger because number of tokens per iter is small
    warmup_iters = 25, # not super necessary potentially
)

# system
system_config = SystemConfig(
    backend = 'gloo',  # gloo backend works better for small data
    device = 'cpu',  # run on cpu only
    compile = True # do not torch compile the model
)

# config = {
#     "io": asdict(io_config),
#     "wandb": asdict(wandb_config),
#     "data": asdict(data_config),
#     "model": asdict(model_config_instance),
#     "optimizer": asdict(optimizer_config),
#     "system": asdict(system_config),
# }
config = {
    "io": io_config,
    "wandb": wandb_config,
    "data": data_config,
    "model": model_config_instance,
    "optimizer": optimizer_config,
    "system": system_config,
}