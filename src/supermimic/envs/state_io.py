"""Read / write an origin-relative floating-base robot state (qpos | qvel)."""

from __future__ import annotations

import torch

STATE_DIM = 36 + 35
NQ = 36
NV = 35


class StateIO:
    def __init__(self, env):
        self.env = env
        self.sim = env.sim
        # States are stored relative to each env's origin so that they are directly comparable
        # with the (origin-free) motion library and translation-invariant across envs.
        self.origins = env.scene.env_origins.clone()  # (N, 3)
        self.nq = int(self.sim.data.qpos.shape[-1])
        self.nv = int(self.sim.data.qvel.shape[-1])
        self.num_joints = self.nq - 7
        self.state_dim = self.nq + self.nv
        if self.num_joints < 1 or self.nv != self.num_joints + 6:
            raise ValueError("StateIO requires one free root and scalar robot joints")

    def read(self) -> torch.Tensor:
        """(N, nq + nv) copy of [qpos | qvel]."""
        # Index with [:] to get plain tensors out of mjlab's TorchArray proxies (torch.cat on the
        # proxies recurses inside their __torch_function__).
        s = torch.cat([self.sim.data.qpos[:], self.sim.data.qvel[:]], dim=-1).clone()
        s[:, 0:3] -= self.origins
        return s

    @torch.no_grad()
    def write(self, env_ids: torch.Tensor, state: torch.Tensor, forward: bool = True) -> None:
        """Overwrite qpos/qvel of ``env_ids`` and refresh derived quantities."""
        state = state.to(self.sim.data.qpos.dtype).clone()
        state[:, 0:3] += self.origins[env_ids]
        if state.shape != (env_ids.numel(), self.state_dim):
            raise ValueError(f"Expected state shape {(env_ids.numel(), self.state_dim)}, got {state.shape}")
        self.sim.data.qpos[env_ids] = state[:, :self.nq]
        self.sim.data.qvel[env_ids] = state[:, self.nq:]
        if hasattr(self.sim.data, "qacc_warmstart"):
            self.sim.data.qacc_warmstart[env_ids] = 0.0
        if forward:
            self.sim.forward()
