from template import CausalLM
from .configuration_gpt2 import GPT2Config
from .attention import CausalSelfAttention
import torch.nn as nn
import torch
from torch.nn import functional as F
from template import MLP, LayerNorm
from template.cache import AttnCache

class Block(nn.Module):

	def __init__(self, config: GPT2Config):
		super().__init__()
		self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
		self.attn = CausalSelfAttention(config)
		self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
		self.mlp = MLP(config)

	def forward(
		self,
		x: torch.Tensor,
		use_cache: bool = False,
		cache: AttnCache = None,
		cache_index: int = None
	):
		new_x, cache = self.attn(self.ln_1(x), use_cache=use_cache, attn_cache=cache, cache_index=cache_index)
		x = x + new_x
		x = x + self.mlp(self.ln_2(x))
		return x, cache


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

	def forward(
		self,
		idx,
		targets=None,
		use_cache: bool = False,
		cache: AttnCache = None,
	):
		device = idx.device
		b, t = idx.size()
		assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
		assert (not use_cache) or (cache is not None), "cache must be provided when use_cache is True"

		# forward the GPT model itself
		tok_emb = self.input_embed.wte(idx) # token embeddings of shape (b, t, n_embd)
		x = self.input_embed.drop(tok_emb)
		for i, block in enumerate(self.transformer.h):
			x, cache = block(x, use_cache=use_cache, cache=cache, cache_index=i)
		x = self.transformer.ln_f(x)

		if targets is not None:
			# if we are given some desired targets also calculate the loss
			logits = self.lm_head(x)
			loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
		else:
			# inference-time mini-optimization: only forward the lm_head on the very last position
			logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
			loss = None

		return logits, loss, cache

	@classmethod
	def from_pretrained(cls, ckpt_path):
		# TODO
		# load the model
		checkpoint = torch.load(ckpt_path, map_location='cpu')
		model_config = GPT2Config(**checkpoint['model_args'])
		model = cls(model_config)
		state_dict = checkpoint['model']
		unwanted_prefix = '_orig_mod.'
		for k, v in list(state_dict.items()):
			if k.startswith(unwanted_prefix):
				state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
		model.load_state_dict(state_dict)
		return model