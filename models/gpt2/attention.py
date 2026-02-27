"""
Attention-related building blocks extracted from model.py.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from template import RotaryEmbedding
from .configuration_gpt2 import GPT2Config
from template.cache import AttnCache

class CausalSelfAttention(nn.Module):

	def __init__(self, config: GPT2Config):
		super().__init__()
		assert config.n_embd % config.n_head == 0
		# key, query, value projections for all heads, but in a batch
		self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
		# output projection
		self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
		# ROPE embeddings
		self.rotary_emb = RotaryEmbedding(
			dim = (config.n_embd // config.n_head),
			max_position_embeddings = config.block_size,
		)
		# regularization
		self.attn_dropout = nn.Dropout(config.dropout)
		self.resid_dropout = nn.Dropout(config.dropout)
		self.n_head = config.n_head
		self.n_embd = config.n_embd
		self.dropout = config.dropout
		# flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
		self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
		if not self.flash:
			print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
			# causal mask to ensure that attention is only applied to the left in the input sequence
			self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
										.view(1, 1, config.block_size, config.block_size))

	def forward(self,
		x: torch.Tensor,
		use_cache: bool = False,
		cache_index: int = None,
		attn_cache: AttnCache = None,
	):
		B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

		assert (not use_cache) or (attn_cache is not None and cache_index is not None), "attn_cache and cache_index must be provided when use_cache is True"

		if use_cache:
			cache = attn_cache.get(cache_index)
			prev_k = cache.get('k')
			prev_v = cache.get('v')
			if prev_k is not None:
				offset = prev_k.size(2)
			else:
				prev_k = torch.zeros(B, self.n_head, 0, C // self.n_head, device=x.device, dtype=x.dtype)
				prev_v = torch.zeros(B, self.n_head, 0, C // self.n_head, device=x.device, dtype=x.dtype)
				offset = 0
		else:
			offset = 0

		# calculate query, key, values for all heads in batch and move head forward to be the batch dim
		q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
		k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
		q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
		v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

		# apply rotary embeddings
		k = self.rotary_emb(k, offset=offset, seq_len=T)
		q = self.rotary_emb(q, offset=offset, seq_len=T)
		if use_cache:
			k = torch.cat([prev_k, k], dim=2)
			v = torch.cat([prev_v, v], dim=2)

		# causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
		if self.flash:
			# efficient attention using Flash Attention CUDA kernels
			y = torch.nn.functional.scaled_dot_product_attention(
				q, k, v,
				attn_mask=None,
				dropout_p=self.dropout if self.training else 0,
				is_causal=True,
			)
		else:
			# manual implementation of attention
			att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
			att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
			att = F.softmax(att, dim=-1)
			att = self.attn_dropout(att)
			y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
		y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

		# output projection
		y = self.resid_dropout(self.c_proj(y))

		if use_cache:
			attn_cache.update(
				cache_index,
				k = k,
				v = v
			)
		
		return y, attn_cache