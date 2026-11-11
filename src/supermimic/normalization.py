"""Explicit, checkpointed empirical input normalization.

The update/forward split follows SuperMimic's RSL-RL 5.4 normalizer. Only
fresh physical observations update moments; replay and imagined forwards never
do. Buffers keep their storage so compiled/CUDA-graph readers remain valid.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch import nn


class RunningNormalizer(nn.Module):
    """Cumulative mean/variance, with the official ``(x-mean)/(std+.01)`` rule.

    Statistics are FP32 regardless of input precision. Batch sufficient
    statistics are accumulated in FP64; distributed training combines them
    before every update, so all ranks retain identical inference transforms.
    Call ``update`` explicitly outside inference_mode and compiled forwards.
    """

    def __init__(self, num_features: int, eps: float = 0.01):
        super().__init__()
        if isinstance(num_features, bool) or not isinstance(num_features, int) or num_features < 1:
            raise ValueError("Normalizer num_features must be a positive integer")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("Normalizer eps must be finite and positive")
        self.num_features = num_features
        self.eps = float(eps)
        self.register_buffer("_mean", torch.zeros(num_features, dtype=torch.float32))
        self.register_buffer("_var", torch.ones(num_features, dtype=torch.float32))
        self.register_buffer("_std", torch.ones(num_features, dtype=torch.float32))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std + self.eps)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if not self.training:
            return
        if x.shape[-1] != self.num_features or not x.is_floating_point():
            raise ValueError("Normalizer update needs floating samples with the configured feature width")
        if x.device != self._mean.device:
            raise ValueError("Normalizer samples and buffers must be on the same device")
        samples = x.detach().reshape(-1, self.num_features).double()
        finite = torch.isfinite(samples)
        # Invalid input must neither poison buffers nor deadlock other ranks.
        safe = torch.where(finite, samples, 0.0)
        totals = torch.cat([
            samples.new_full((1,), samples.shape[0]), safe.sum(0), safe.square().sum(0),
            finite.all().to(torch.float64).reshape(1),
        ])
        world = 1
        if dist.is_available() and dist.is_initialized():
            world = dist.get_world_size()
            dist.all_reduce(totals)
        valid = totals[-1] == world
        torch._assert_async(valid, "Normalizer received nonfinite fresh observations")
        n = totals[0]
        nonempty = n > 0
        batch_mean = totals[1:1 + self.num_features] / n.clamp_min(1)
        batch_var = (totals[1 + self.num_features:-1] / n.clamp_min(1) - batch_mean.square()).clamp_min(0)
        count = self.count.double()
        new_count = count + n
        rate = n / new_count.clamp_min(1)
        delta = batch_mean - self._mean.double()
        mean = self._mean.double() + rate * delta
        var = self._var.double() + rate * (
            batch_var - self._var.double() + delta * (batch_mean - mean)
        )
        mean, var = mean.float(), var.clamp_min(0).float()
        finite_moments = torch.isfinite(mean).all() & torch.isfinite(var).all()
        torch._assert_async(finite_moments, "Normalizer moments exceed the finite FP32 range")
        apply = valid & nonempty & finite_moments
        self._mean.copy_(torch.where(apply, mean, self._mean))
        self._var.copy_(torch.where(apply, var, self._var))
        self._std.copy_(self._var.sqrt())
        self.count.copy_(torch.where(apply, new_count.long(), self.count))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Reject corrupt moment buffers instead of silently corrupting inference."""
        errors = []
        expected = {"_mean": self._mean, "_var": self._var,
                    "_std": self._std, "count": self.count}
        for name, target in expected.items():
            value = state_dict.get(prefix + name)
            if value is None:
                continue  # Let nn.Module report missing keys normally.
            if (not isinstance(value, torch.Tensor) or value.shape != target.shape
                    or value.dtype != target.dtype):
                errors.append(f"{prefix}{name}: normalizer shape/dtype differs")
            elif not torch.isfinite(value).all():
                errors.append(f"{prefix}{name}: normalizer buffer is nonfinite")
            elif name != "_mean" and (value < 0).any():
                errors.append(f"{prefix}{name}: normalizer count/variance/std must be nonnegative")
        if not errors and prefix + "_var" in state_dict and prefix + "_std" in state_dict:
            var, std = state_dict[prefix + "_var"], state_dict[prefix + "_std"]
            if not torch.allclose(std, var.sqrt(), atol=1e-6, rtol=1e-5):
                errors.append(f"{prefix}: normalizer std does not match variance")
        if errors:
            error_msgs.extend(errors)
            return
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    @property
    def mean(self) -> torch.Tensor:
        return self._mean

    @property
    def std(self) -> torch.Tensor:
        return self._std
