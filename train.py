"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import sys
import time
import math
import pickle
from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
from datetime import datetime

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from utils.config import *
from models import *
from model_list import get_model, model_map
from data.dataloader import BatchIterator

os.environ["TORCHDYNAMO_VERBOSE"] = "1"

# -----------------------------------------------------------------------------

# load the config values from config file and command line overrides
config = get_default_config()
exec(open('configurator.py').read()) # overrides from command line or config file
io_config: IOConfig = config['io']
wandb_config: WandbConfig = config['wandb']
data_config: DataConfig = config['data']
model_config_instance = config['model']
optimizer_config: OptimizerConfig = config['optimizer']
system_config: SystemConfig = config['system']
print("training config:")
print(io_config)
print(wandb_config)
print(data_config)
print(model_config_instance)
print(optimizer_config)
print(system_config)

run_name = model_config_instance.model_name + time.strftime("-%Y-%m-%d-%H-%M-%S")

# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    print("setting up ddp...")
    init_process_group(backend=system_config.backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert data_config.gradient_accumulation_steps % ddp_world_size == 0
    data_config.gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    device = system_config.device
tokens_per_iter = data_config.gradient_accumulation_steps * ddp_world_size * data_config.batch_size * data_config.block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(io_config.out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[system_config.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', data_config.dataset)

train_dataset = BatchIterator(data_config, 'train', device=torch.device(device), device_type=device_type)
val_dataset = BatchIterator(data_config, 'val', device=torch.device(device), device_type=device_type)

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        return train_dataset.get_batch()
    else:
        return val_dataset.get_batch()

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# get the model class and config based on model_name
model_class = get_model(model_config_instance.model_name)
if model_class is None:
    raise ValueError(f"Unknown model name '{model_config_instance.model_name}'. Available models: {list(model_map.keys())}")

# model init
model_args = dict(n_layer=model_config_instance.n_layer, n_head=model_config_instance.n_head, n_embd=model_config_instance.n_embd, block_size=model_config_instance.block_size,
                  bias=model_config_instance.bias, vocab_size=None, dropout=model_config_instance.dropout) # start with model_args from command line
if io_config.init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    for k, v in model_args.items():
        setattr(model_config_instance, k, v)
    model = model_class(model_config_instance)
elif io_config.init_from == 'resume':
    print(f"Resuming training from {io_config.out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(io_config.out_dir, f'ckpt_{model_config_instance.model_name}.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    for k, v in model_args.items():
        setattr(model_config_instance, k, v)
    model = model_class(model_config_instance)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
else:
    raise ValueError(f"Unknown init_from value '{io_config.init_from}'")
# crop down the model block size if desired, using model surgery
if data_config.block_size < model.config.block_size:
    model.crop_block_size(data_config.block_size)
    model_args['block_size'] = data_config.block_size # so that the checkpoint will have the right value
model = model.to(device)
model.show_number_of_parameters()

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(device=device, enabled=(system_config.dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(optimizer_config.weight_decay, optimizer_config.learning_rate, (optimizer_config.beta1, optimizer_config.beta2), device_type)
if io_config.init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if system_config.compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(splits: list = ['train', 'val']):
    out = {}
    model.eval()
    for split in splits:
        losses = torch.zeros(io_config.eval_iters)
        for k in range(io_config.eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < optimizer_config.warmup_iters:
        return optimizer_config.learning_rate * (it + 1) / (optimizer_config.warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > optimizer_config.lr_decay_iters:
        return optimizer_config.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - optimizer_config.warmup_iters) / (optimizer_config.lr_decay_iters - optimizer_config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return optimizer_config.min_lr + coeff * (optimizer_config.learning_rate - optimizer_config.min_lr)

# logging
if wandb_config.wandb_log and master_process:
    import wandb
    # flatten config to avoid pydantic warnings about tuples
    flat_config = {}
    for category, category_config in config.items():
        if is_dataclass(category_config):
            for k, v in asdict(category_config).items():
                flat_config[f"{category}/{k}"] = v
        elif isinstance(category_config, dict):
                for k, v in category_config.items():
                    flat_config[f"{category}/{k}"] = v
        else:
            flat_config[category] = category_config
    wandb.init(
        project=wandb_config.wandb_project,
        name=wandb_config.wandb_run_name + f"-{model.get_num_params()/1e6:.1f}M" + f"-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
        config=flat_config
    )

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

training_start_time = time.time()

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if optimizer_config.decay_lr else optimizer_config.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % io_config.eval_interval == 0 and master_process:
        losses = estimate_loss(['val'])
        # print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print(f"step {iter_num}: val loss {losses['val']:.4f}")
        if wandb_config.wandb_log:
            wandb.log({
                "iter": iter_num,
                # "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or io_config.always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'model_name': model_config_instance.model_name,
                }
                print(f"saving checkpoint to {io_config.out_dir}")
                torch.save(checkpoint, os.path.join(io_config.out_dir, f'ckpt_{model_config_instance.model_name}.pt'))
                with open(os.path.join(io_config.out_dir, f'result_{run_name}.txt'), 'w') as f:
                    f.write(f"{model_config_instance.model_name} {model_config_instance.eta} {best_val_loss:.4f}\n")
    if iter_num == 0 and io_config.eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(data_config.gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == data_config.gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / data_config.gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if optimizer_config.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), optimizer_config.grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % io_config.log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * data_config.gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(data_config.batch_size * data_config.gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        if wandb_config.wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": lossf,
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > optimizer_config.max_iters:
        break

training_end_time = time.time()
total_training_time = training_end_time - training_start_time
print(f"Training completed in {total_training_time/60:.2f} minutes.")

if wandb_config.wandb_log and master_process:
    wandb.finish()

if ddp:
    destroy_process_group()
