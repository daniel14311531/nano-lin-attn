import torch
import torch.nn as nn
import torch.nn.functional as F
from .configuration_o2d_deltanet import O2DDeltaNetConfig
from template.shortconvolution import ShortConv
from template.norm import RMSNorm
from ops.o2b_deltanet import o2b_delta_rule
from template.cache import AttnCache

class O2DDeltaNetLayer(nn.Module):
    def __init__(self, config: O2DDeltaNetConfig):
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
        
        self.eta = config.eta
        self.use_qk_activation = config.use_qk_activation
        self.sync_kv_scale = config.sync_kv_scale

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
        if self.use_qk_activation:
            k, q = F.silu(k), F.silu(q)

        k = k.view(B, L, self.n_head, self.head_dim)
        q = q.view(B, L, self.n_head, self.head_dim)
        v = v.view(B, L, self.n_head, self.head_dim)
        beta = beta.view(B, L, self.n_head)
        v = F.silu(v)

        knorm = torch.norm(k, dim=-1, keepdim=True)  # (B, L, n_head, 1)
        qnorm = torch.norm(q, dim=-1, keepdim=True)  # (B, L, n_head, 1)
        k = k / (knorm + 1e-6)
        if self.sync_kv_scale:
            v = v / (knorm + 1e-6)
        q = q / (qnorm + 1e-6)

        cur_state = None
        if use_cache:
            cur_state = cache.get('state')
        if cur_state is None:
            cur_state = (
                torch.zeros(B, self.n_head, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype),
                torch.zeros(B, self.n_head, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype),
                torch.tensor(0, dtype=torch.long, device=x.device),
            )

        b = self.eta * beta.view(B, L, self.n_head, 1) * k

        o, cur_state = o2b_delta_rule(
            k = k,
            q = q,
            v = v,
            b = b,
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