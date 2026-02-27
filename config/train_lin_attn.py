# train a miniature shakespeare model
# good for debugging and playing on macbooks and such
from utils.config import *
from models import *
from model_list import get_config

model_name = 'deltanet' # [TODO]
model_config = get_config(model_name)

# I/O
io_config = IOConfig(
    out_dir = 'out-shakespeare',
    eval_interval = 25, # keep frequent because we'll overfit
    eval_iters = 100,
    log_interval = 5, # don't print too too often
    always_save_checkpoint = False, # we expect to overfit on this small dataset, so only save when val improves
    init_from = 'scratch', # 'scratch' or 'resume' from previous checkpoint in out_dir
)

# wandb
wandb_config = WandbConfig(
    wandb_log = False,
    wandb_project = 'shakespeare',
    wandb_run_name = model_name, # [TODO]
)

# data
data_config = DataConfig(
    dataset = 'shakespeare',
    gradient_accumulation_steps = 4,
    batch_size = 8,
    block_size = 256, # context of up to 256 previous characters
)

# baby GPT model :)
model_config_instance = model_config(
    model_name = model_name,
    n_layer = 6,
    n_head = 6,
    n_embd = 384,
    dropout = 0.2,
    block_size = 256,
    bias = False,

    conv_size = 4,
    initial_state = False,
    eta = 0.1,
)

# optimizer
optimizer_config = OptimizerConfig(
    learning_rate = 5e-4, # with baby networks can afford to go a bit higher
    max_iters = 800,
    lr_decay_iters = 1000, # make equal to max_iters usually
    min_lr = 1e-4, # learning_rate / 10 usually
    beta2 = 0.99, # make a bit bigger because number of tokens per iter is small
    warmup_iters = 20, # not super necessary potentially
)

# system (cpu)
system_config = SystemConfig(
    backend = 'gloo',  # gloo backend works better for small data
    device = 'cpu',  # run on cpu only
    compile = True # do not torch compile the model
)
# system (cuda)
# system_config = SystemConfig(
#     backend = 'nccl',  # gloo backend works better for small data
#     device = 'cuda',  # run on cpu only
#     compile = False # do not torch compile the model
# )

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