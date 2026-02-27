import torch
import torch.nn as nn
import torch.nn.functional as F
from .configuration_omd_deltanet import OmdDeltaNetConfig
from template.shortconvolution import ShortConv
from template.norm import RMSNorm
from template.delta_rule import delta_rule

class OmdDeltaNetLayer(nn.Module):
    def __init__(self, config: OmdDeltaNetConfig):
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
        self.use_qk_activation = config.use_qk_activation
        self.sync_kv_scale = config.sync_kv_scale

    def forward(self, x):
        B, L, D = x.size()
        k = self.k_conv1d(self.k_proj(x))
        q = self.q_conv1d(self.q_proj(x))
        v = self.v_conv1d(self.v_proj(x))
        beta = torch.sigmoid(self.beta_proj(x))
        if self.use_qk_activation:
            k = F.silu(k)
            q = F.silu(q)

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

        k = torch.concat([torch.zeros_like(k[:, :1, :, :]), k[:, :-1, :, :]], dim=1)
        v = torch.concat([torch.zeros_like(v[:, :1, :, :]), v[:, :-1, :, :]], dim=1)
        beta = torch.concat([torch.zeros_like(beta[:, :1, :]), beta[:, :-1, :]], dim=1)

        if self.initial_state:
            init_state = self.init_state.repeat(B, 1, 1, 1)
        else:
            init_state = torch.zeros(B, self.n_head, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype)

        o = delta_rule(
            k = k,
            q = q,
            v = v,
            beta = self.eta * beta,
            init_state = init_state
        )

        o = self.out_norm(o)
        o = o.contiguous().view(B, L, D)
        o = self.out_proj(o)

        return o