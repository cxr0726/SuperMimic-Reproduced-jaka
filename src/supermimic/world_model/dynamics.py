"""Dynamics MLP f_omega (paper Table A.4: MLP (2048, 2048, 2048))."""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...], out_dim: int, act=nn.SiLU, zero_last: bool = False, last_gain: float | None = None, layer_norm: bool = False):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            if layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(act())
            d = h
        last = nn.Linear(d, out_dim)
        if zero_last:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        elif last_gain is not None:
            nn.init.orthogonal_(last.weight, gain=last_gain)
            nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DynamicsMLP(nn.Module):
    """delta = f(canonical(s), a, xi_norm) -> (…, 70). Zero-initialised last layer (identity dynamics at init)."""

    def __init__(self, feat_dim: int = 71, act_dim: int = 29, xi_dim: int = 33, hidden=(2048, 2048, 2048), out_dim: int = 70, *, layer_norm: bool = False, activation: str = "silu"):
        super().__init__()
        activations = {"silu": nn.SiLU, "gelu": nn.GELU}
        if activation not in activations:
            raise ValueError(f"Unknown dynamics activation: {activation}")
        self.mlp = MLP(feat_dim + act_dim + xi_dim, tuple(hidden), out_dim,
                       zero_last=True, layer_norm=layer_norm, act=activations[activation])

    def forward(self, feat: torch.Tensor, action: torch.Tensor, xi_norm: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([feat, action, xi_norm], dim=-1))
