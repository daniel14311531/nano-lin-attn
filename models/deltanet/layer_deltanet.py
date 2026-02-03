import torch
import torch.nn as nn
import torch.nn.functional as F
from .configuration_deltanet import DeltaNetConfig
from template.shortconvolution import ShortConv
from template.norm import RMSNorm
from template.delta_rule import delta_rule
from template.cache import AttnCache

class DeltaNetLayer(nn.Module):
    def __init__(self, config: DeltaNetConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.beta_proj = nn.Linear(config.n_embd, config.n_head, bias=config.bias)
        
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_norm = RMSNorm(self.head_dim, eps=1e-6)
        
        self.k_conv1d = ShortConv(config.conv_size, config.n_embd)
        self.q_conv1d = ShortConv(config.conv_size, config.n_embd)
        self.v_conv1d = ShortConv(config.conv_size, config.n_embd)
        
        self.initial_state = config.initial_state
        if config.initial_state:
            self.init_state = nn.Parameter(torch.zeros(1, config.n_head, self.head_dim, self.head_dim))
        self.eta = config.eta

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool=False,
        cache_index: int=None,
        attn_cache: AttnCache = None,
    ):
        B, L, D = x.size()

        assert (not use_cache) or (attn_cache is not None and cache_index is not None), "attn_cache and cache_index must be provided when use_cache is True"

        prev_k, prev_q, prev_v = None, None, None
        if use_cache:
            cache = attn_cache.get(cache_index)
            prev_k = cache.get('k')
            prev_q = cache.get('q')
            prev_v = cache.get('v')

        k, prev_k = self.k_conv1d(self.k_proj(x), prev_k)
        q, prev_q = self.q_conv1d(self.q_proj(x), prev_q)
        v, prev_v = self.v_conv1d(self.v_proj(x), prev_v)
        beta = torch.sigmoid(self.beta_proj(x))
        k, q = F.silu(k), F.silu(q)

        k = k.view(B, L, self.n_head, self.head_dim)
        q = q.view(B, L, self.n_head, self.head_dim)
        v = v.view(B, L, self.n_head, self.head_dim)
        beta = beta.view(B, L, self.n_head)

        k = k / torch.norm(k, p=2, dim=-1, keepdim=True)
        q = q / torch.norm(q, p=2, dim=-1, keepdim=True)

        cur_state = None
        if use_cache:
            cur_state = cache.get('state')
        if cur_state is None:
            if self.initial_state:
                cur_state = self.init_state.repeat(B, 1, 1, 1)
            else:
                cur_state = torch.zeros(B, self.n_head, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype)

        o, cur_state = delta_rule(
            k = k,
            q = q,
            v = v,
            beta = self.eta * beta,
            init_state = cur_state
        )

        o = self.out_norm(o)
        o = o.contiguous().view(B, L, D)
        o = self.out_proj(o)

        if use_cache:
            attn_cache.update(
                cache_index,
                k = prev_k,
                q = prev_q,
                v = prev_v,
                state = cur_state
            )

        return o, attn_cache