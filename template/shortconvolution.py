import torch.nn as nn
import torch
from torch.nn import functional as F


class ShortConv(nn.Module):
    def __init__(self, kernel_size: int, n_embd: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=n_embd,
            out_channels=n_embd,
            kernel_size=kernel_size,
            padding=0,
            groups=n_embd,
            bias=False,
        )
        self.n_embd = n_embd

    def forward(
        self,
        x: torch.Tensor,
        prev_x: torch.Tensor = None,
    ):
        # x: (B, T, C)
        x = x.transpose(1, 2)  # (B, C, T)
        if prev_x is None:
            prev_x = torch.zeros(
                x.size(0),
                x.size(1),
                self.conv.kernel_size[0] - 1,
                device=x.device,
                dtype=x.dtype,
            )
        x = torch.cat([prev_x, x], dim=2)  # (B, C, T + K - 1)
        new_x = self.conv(x)  # (B, C, T)
        new_x = new_x.transpose(1, 2)  # (B, T, C)
        return new_x, x[:, :, - (self.conv.kernel_size[0] - 1):]  # return output and new prev_x