import torch
import torch.nn as nn
import torch.nn.functional as F
from .configuration_deltanet import DeltaNetConfig
from template.shortconvolution import ShortConv
from template.norm import RMSNorm
from template.delta_rule import delta_rule

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

    def forward(self, x):
        B, L, D = x.size()
        k = F.silu(self.k_conv1d(self.k_proj(x)))
        q = F.silu(self.q_conv1d(self.q_proj(x)))
        v = self.v_conv1d(self.v_proj(x))
        beta = torch.sigmoid(self.beta_proj(x))

        k = k.view(B, L, self.n_head, self.head_dim)
        q = q.view(B, L, self.n_head, self.head_dim)
        v = v.view(B, L, self.n_head, self.head_dim)
        beta = beta.view(B, L, self.n_head)

        k = k / torch.norm(k, p=2, dim=-1, keepdim=True)
        q = q / torch.norm(q, p=2, dim=-1, keepdim=True)

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