"""A frozen deployment candidate averaged after successful actor optimizer steps."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping

import torch
from torch import nn


class ActorEMA:
    """Match InstantTrack's first-update copy, then average parameters only.

    Buffers describe the policy's physical input/action mapping, so they are
    copied exactly. The live actor and its optimizer remain the training model.
    """

    def __init__(self, policy: nn.Module, decay: float = 0.995):
        if not math.isfinite(decay) or not 0 < decay < 1:
            raise ValueError("Actor EMA decay must be finite and strictly between 0 and 1")
        self.decay = float(decay)
        self._num_updates = 0
        self._device_updates: torch.Tensor | None = None
        self.module = copy.deepcopy(policy).eval().requires_grad_(False)

    @property
    def num_updates(self) -> int:
        # Read only at log/checkpoint boundaries, never inside a captured step.
        return self._num_updates if self._device_updates is None else int(self._device_updates.item())

    @num_updates.setter
    def num_updates(self, value: int) -> None:
        self._num_updates = value
        if self._device_updates is not None:
            self._device_updates.fill_(value)

    def enable_cuda_graph(self) -> None:
        """Track the first-copy/EMA boundary on device during graph replay."""
        device = next(self.module.parameters()).device
        if device.type != "cuda":
            raise ValueError("CUDA Graph EMA requires a CUDA policy")
        if self._device_updates is None:
            self._device_updates = torch.tensor(self._num_updates, dtype=torch.long, device=device)

    @torch.no_grad()
    def update(self, policy: nn.Module) -> None:
        if self._device_updates is not None:
            first = self._device_updates == 0
            for average, live in zip(self.module.parameters(), policy.parameters(), strict=True):
                average.copy_(torch.where(first, live, average.lerp(live, 1.0 - self.decay)))
            self.sync_buffers(policy)
            self._device_updates.add_(1)
            return
        for average, live in zip(self.module.parameters(), policy.parameters(), strict=True):
            if self._num_updates == 0:
                average.copy_(live)
            else:
                average.lerp_(live, 1.0 - self.decay)
        self.sync_buffers(policy)
        self._num_updates += 1

    @torch.no_grad()
    def sync_buffers(self, policy: nn.Module) -> None:
        """Refresh running statistics after collection without averaging weights."""
        for average, live in zip(self.module.buffers(), policy.buffers(), strict=True):
            average.copy_(live)

    def state_dict(self) -> Mapping[str, torch.Tensor]:
        return self.module.state_dict()

    def validate_checkpoint(self, checkpoint: Mapping) -> bool:
        """Validate before mutating any model; False denotes no historical EMA."""
        state = checkpoint.get("policy_ema")
        if state is None:
            return False
        if not isinstance(state, Mapping) or not state:
            raise ValueError("Checkpoint policy_ema must be a nonempty state_dict")
        decay = checkpoint.get("policy_ema_decay")
        if (isinstance(decay, bool) or not isinstance(decay, (int, float))
                or not math.isfinite(decay) or float(decay) != self.decay):
            raise ValueError("Checkpoint actor EMA decay differs; use matching --policy-trainer.ema-decay")
        count = checkpoint.get("policy_ema_updates")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Checkpoint policy_ema_updates must be a nonnegative integer")
        expected = self.module.state_dict()
        if state.keys() != expected.keys():
            raise ValueError("Checkpoint policy_ema keys differ from the current actor")
        for name, target in expected.items():
            value = state[name]
            if (not isinstance(value, torch.Tensor) or value.shape != target.shape
                    or value.dtype != target.dtype):
                raise ValueError(f"Checkpoint policy_ema tensor {name} differs in shape or dtype")
            if not torch.isfinite(value).all():
                raise ValueError(f"Checkpoint policy_ema tensor {name} is nonfinite")
        # These buffers must never become a second, incompatible PD mapping.
        for name in ("reference_q_default", "reference_action_scale"):
            if name in expected and not torch.equal(state[name].to(expected[name]), expected[name]):
                raise ValueError(f"Checkpoint policy_ema {name} differs from the physical action mapping")
        return True

    @torch.no_grad()
    def restore(self, policy: nn.Module, checkpoint: Mapping) -> None:
        if self.validate_checkpoint(checkpoint):
            self.module.load_state_dict(checkpoint["policy_ema"], strict=True)
            self.num_updates = checkpoint["policy_ema_updates"]
        else:
            # Call only after loading live weights, never from random init.
            self.module.load_state_dict(policy.state_dict(), strict=True)
            for average, live in zip(self.module.buffers(), policy.buffers(), strict=True):
                average.copy_(live)
            self.num_updates = 0
        self.module.eval().requires_grad_(False)
