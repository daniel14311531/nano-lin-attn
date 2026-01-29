from template import CausalLM
from .configuration_gpt2 import GPT2Config
from .attention import CausalSelfAttention
import torch.nn as nn
import torch
from torch.nn import functional as F
from template import MLP, LayerNorm

class Block(nn.Module):

	def __init__(self, config: GPT2Config):
		super().__init__()
		self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
		self.attn = CausalSelfAttention(config)
		self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
		self.mlp = MLP(config)

	def forward(self, x):
		x = x + self.attn(self.ln_1(x))
		x = x + self.mlp(self.ln_2(x))
		return x


class GPT(CausalLM):

	def __init__(self, config: GPT2Config):
		super().__init__(config)
		assert config.vocab_size is not None
		assert config.block_size is not None
		self.config = config

		self.transformer = nn.ModuleDict(dict(
			h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
			ln_f = LayerNorm(config.n_embd, bias=config.bias),
		))

		# init all weights
		self.apply(self._init_weights)

	def forward(self, idx, targets=None):
		device = idx.device
		b, t = idx.size()
		assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

		# forward the GPT model itself
		tok_emb = self.input_embed.wte(idx) # token embeddings of shape (b, t, n_embd)
		x = self.input_embed.drop(tok_emb)
		for block in self.transformer.h:
			x = block(x)
		x = self.transformer.ln_f(x)

		if targets is not None:
			# if we are given some desired targets also calculate the loss
			logits = self.lm_head(x)
			loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
		else:
			# inference-time mini-optimization: only forward the lm_head on the very last position
			logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
			loss = None

		return logits, loss

	@classmethod
	def from_pretrained(cls, model_type, override_args=None):
		raise NotImplementedError("loading from pretrained GPT-2 is currently disabled.")
		assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
		override_args = override_args or {} # default to empty dict
		# only dropout can be overridden see more notes below
		assert all(k == 'dropout' for k in override_args)
		from transformers import GPT2LMHeadModel
		print("loading weights from pretrained gpt: %s" % model_type)

		# n_layer, n_head and n_embd are determined from model_type
		config_args = {
			'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
			'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
			'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
			'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
		}[model_type]
		print("forcing vocab_size=50257, block_size=1024, bias=True")
		config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
		config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
		config_args['bias'] = True # always True for GPT model checkpoints
		# we can override the dropout rate, if desired
		if 'dropout' in override_args:
			print(f"overriding dropout rate to {override_args['dropout']}")
			config_args['dropout'] = override_args['dropout']
		# create a from-scratch initialized minGPT model
		config = Config(**config_args)
		model = GPT(config)
		sd = model.state_dict()
		sd_keys = sd.keys()
		sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

		# init a huggingface/transformers model
		model_hf = GPT2LMHeadModel.from_pretrained(model_type)
		sd_hf = model_hf.state_dict()

		# copy while ensuring all of the parameters are aligned and match in names and shapes
		sd_keys_hf = sd_hf.keys()
		sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
		sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
		transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
		# basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
		# this means that we have to transpose these weights when we import them
		assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
		for k in sd_keys_hf:
			if any(k.endswith(w) for w in transposed):
				# special treatment for the Conv1D weights we need to transpose
				assert sd_hf[k].shape[::-1] == sd[k].shape
				with torch.no_grad():
					sd[k].copy_(sd_hf[k].t())
			else:
				# vanilla copy over the other parameters
				assert sd_hf[k].shape == sd[k].shape
				with torch.no_grad():
					sd[k].copy_(sd_hf[k])

		return model