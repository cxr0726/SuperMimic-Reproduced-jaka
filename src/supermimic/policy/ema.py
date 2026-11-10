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
        self.num_updates = 0
        self.module = copy.deepcopy(policy).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, policy: nn.Module) -> None:
        for average, live in zip(self.module.parameters(), policy.parameters(), strict=True):
            if self.num_updates == 0:
                average.copy_(live)
            else:
                average.lerp_(live, 1.0 - self.decay)
        for average, live in zip(self.module.buffers(), policy.buffers(), strict=True):
            average.copy_(live)
        self.num_updates += 1

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
