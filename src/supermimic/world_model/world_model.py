"""Learned dynamics, configurable termination, and analytic observation/reward heads."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from supermimic.normalization import RunningNormalizer
from supermimic.world_model.dynamics import MLP, DynamicsMLP
from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg


@dataclass
class WorldModelCfg:
    hidden: tuple[int, ...] = (2048, 2048, 2048)
    term_hidden: tuple[int, ...] = (64, 64)
    state_spec: StateSpecCfg = field(default_factory=StateSpecCfg)
    term_loss_weight: float = 1.0
    height_threshold: float = 0.25
    tilt_threshold: float = 0.8
    # Legacy standalone/checkpoint defaults; TrainConfig enables the official bundle.
    input_normalization: bool = False
    hidden_layer_norm: bool = False
    hidden_activation: Literal["silu", "gelu"] = "silu"
    termination_mode: Literal["learned", "analytic"] = "learned"
    termination_softness: float = 0.05


class TerminationHead(nn.Module):
    """P(terminal | state, reference). Input: analytic termination features and their magnitudes."""

    def __init__(self, hidden=(64, 64), height_thr: float = 0.25, tilt_thr: float = 0.8):
        super().__init__()
        self.mlp = MLP(4, tuple(hidden), 1)
        self.register_buffer("scale", torch.tensor([1.0 / height_thr, 1.0 / tilt_thr]))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats (…, 2) = [height error, tilt error] -> logits (…,)"""
        x = feats * self.scale
        return self.mlp(torch.cat([x, x.abs()], dim=-1)).squeeze(-1)


class WorldModel(nn.Module):
    def __init__(self, cfg: WorldModelCfg, heads: AnalyticHeads, device, *,
                 action_scale: torch.Tensor | None = None, action_offset: torch.Tensor | None = None):
        super().__init__()
        self.cfg = cfg
        if cfg.termination_mode not in ("learned", "analytic"):
            raise ValueError("termination_mode must be learned or analytic")
        if any(not math.isfinite(x) or x <= 0 for x in
               (cfg.height_threshold, cfg.tilt_threshold, cfg.termination_softness)):
            raise ValueError("Termination thresholds and softness must be finite and positive")
        self.spec = StateSpec(cfg.state_spec, device, num_joints=heads.num_joints)
        self.dynamics = DynamicsMLP(feat_dim=self.spec.feat_dim, act_dim=self.spec.num_joints,
                                    xi_dim=self.spec.xi_dim, hidden=cfg.hidden,
                                    out_dim=self.spec.delta_dim, layer_norm=cfg.hidden_layer_norm,
                                    activation=cfg.hidden_activation)
        self.term_head = (TerminationHead(cfg.term_hidden, cfg.height_threshold, cfg.tilt_threshold)
                          if cfg.termination_mode == "learned" else None)
        if cfg.input_normalization:
            if action_scale is None:
                raise ValueError("WM input normalization requires the simulator's action_scale")
            action_scale = torch.as_tensor(action_scale, dtype=torch.float32, device=device)
            action_offset = torch.as_tensor(
                heads.default_joint_pos if action_offset is None else action_offset,
                dtype=torch.float32, device=device)
            if action_scale.shape != (self.spec.num_joints,) or action_offset.shape != action_scale.shape:
                raise ValueError("WM action affine mapping must match the robot's joint order")
            if not torch.isfinite(action_scale).all() or not torch.isfinite(action_offset).all():
                raise ValueError("WM action affine mapping must be finite")
            self.register_buffer("action_scale", action_scale.clone())
            self.register_buffer("action_offset", action_offset.clone())
            self.state_normalizer = RunningNormalizer(self.spec.feat_dim)
            self.action_normalizer = RunningNormalizer(self.spec.num_joints)
        self.heads = heads  # analytic observation / reward / FK (no parameters)
        self.to(device)

    # ------------------------------------------------------------------ one step

    def step(self, s: torch.Tensor, action: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """s_{t+1} = integrate(s_t, f(s_t, a_t, xi))."""
        feat = self.spec.canonical(s)
        if self.cfg.input_normalization:
            # Preserve this repository's heading representation, but replace its
            # fixed feature divisors with empirical statistics in physical units.
            feat = self.state_normalizer(feat * self.spec.in_scale)
            action = self.action_normalizer(self.proc_action(action))
        delta = self.dynamics(feat, action, self.spec.normalize_xi(xi))
        return self.spec.integrate(s, delta)

    def proc_action(self, action: torch.Tensor) -> torch.Tensor:
        """Actual affine PD target, before encoder bias/optional actuator delay."""
        return action * self.action_scale + self.action_offset

    @torch.no_grad()
    def update_normalization(self, states: torch.Tensor, actions: torch.Tensor) -> None:
        """One update from fresh physical transitions, never from replay/imagination."""
        if not self.cfg.input_normalization:
            return
        features = self.spec.canonical(states) * self.spec.in_scale
        for normalizer, values in ((self.state_normalizer, features),
                                   (self.action_normalizer, self.proc_action(actions))):
            normalizer.train()
            normalizer.update(values.reshape(-1, values.shape[-1]))
            normalizer.eval()

    def _termination_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        if self.cfg.termination_mode == "analytic":
            error = torch.maximum(feats[..., 0].abs() / self.cfg.height_threshold,
                                  feats[..., 1].abs() / self.cfg.tilt_threshold)
            return (error - 1.0) / self.cfg.termination_softness
        return self.term_head(feats)

    def termination_logits(self, s_next: torch.Tensor, ref: ReferenceFrame) -> torch.Tensor:
        # The pelvis is the free-joint body, so its pose is the state's root pose: no FK needed.
        feats = self.heads.termination_features(s_next[..., 0:3], s_next[..., 3:7], ref)
        return self._termination_from_features(feats)

    # ------------------------------------------------------------------ Eq. 1

    def _loss_step(self, s, a, xi, s_true, done, ref_root_pos, ref_root_quat):
        """One autoregressive step of Eq. 1: returns (s_next, sq_err (n,70), term_bce (n,), logits (n,))."""
        s_next = self.step(s, a, xi)
        e = self.spec.error_vector(s_next, s_true)
        feats = self.heads.termination_features(s_next[..., 0:3], s_next[..., 3:7], ReferenceFrame(None, None, None, None, ref_root_pos, ref_root_quat))
        logits = self._termination_from_features(feats)
        # Analytic termination is a known rule, not another supervised loss on
        # the dynamics. Its gradients still flow through actor imagination.
        bce = (F.binary_cross_entropy_with_logits(logits, done, reduction="none")
               if self.cfg.termination_mode == "learned" else torch.zeros_like(done))
        return s_next, e * e, bce, logits

    def enable_compile(self, mode: str | None = "default") -> None:
        self._loss_step_fn = torch.compile(self._loss_step, mode=None if mode == "default" else mode, dynamic=False) if mode else self._loss_step

    def _window_rollout(self, initial_state: torch.Tensor, actions: torch.Tensor,
                        xi: torch.Tensor) -> torch.Tensor:
        """Predict every transition in a replay window without teacher forcing.

        Keeping this complete recurrence in one tensor-only function lets
        Inductor optimize its forward and backward together. The replay schema
        contains k+1 states and k actions: all k transitions are retained.
        """
        state = initial_state
        predictions = []
        for action in actions.unbind(0):
            state = self.step(state, action, xi)
            predictions.append(state)
        return torch.stack(predictions)

    def _window_losses(self, predictions: torch.Tensor, targets: torch.Tensor,
                       done: torch.Tensor, ref_root_pos: torch.Tensor,
                       ref_root_quat: torch.Tensor):
        """Batch analytic errors/termination over time, preserving Eq. 1.

        Leading dimensions are (k, batch). Only the dynamics recurrence is
        sequential; target errors and termination features have no dependence
        between time indices and can be evaluated in one batch.
        """
        error = self.spec.error_vector(predictions, targets)
        squared_error = error.square()
        reference = ReferenceFrame(None, None, None, None, ref_root_pos, ref_root_quat)
        features = self.heads.termination_features(
            predictions[..., :3], predictions[..., 3:7], reference,
        )
        logits = self._termination_from_features(features)
        term_bce = (F.binary_cross_entropy_with_logits(logits, done, reduction="none")
                    if self.cfg.termination_mode == "learned" else torch.zeros_like(done))
        # Equal weighting of every transition and sample, exactly as in the
        # stepwise sum of per-step batch means (up to FP reduction order).
        state_loss = squared_error.sum(-1).mean()
        term_loss = term_bce.mean()
        loss = state_loss + self.cfg.term_loss_weight * term_loss
        with torch.no_grad():
            metrics = {
                "state_loss": state_loss.detach(),
                "term_bce": term_loss.detach(),
                "nse_1step": squared_error[0].mean(),
                "nse_kstep": squared_error[-1].mean(),
                "term_acc": ((logits > 0).float() == done).float().mean(),
            }
        return loss, metrics

    def window_loss_full(self, states: torch.Tensor, a_exec: torch.Tensor,
                         xi: torch.Tensor, done: torch.Tensor,
                         ref_root_pos: torch.Tensor, ref_root_quat: torch.Tensor):
        """Tensor-only complete rollout plus batched loss, eager or compiled.

        Execution wrappers are installed by WorldModelTrainer and do not add
        parameters/buffers or update normalization statistics.
        """
        rollout = getattr(self, "_window_rollout_fn", None) or self._window_rollout
        losses = getattr(self, "_window_losses_fn", None) or self._window_losses
        predictions = rollout(states[0], a_exec, xi)
        return losses(predictions, states[1:], done, ref_root_pos, ref_root_quat)

    def window_loss(self, states: torch.Tensor, a_exec: torch.Tensor, xi: torch.Tensor, done: torch.Tensor, refs: list[ReferenceFrame]):
        """k-step autoregressive loss.

        states: (k+1, n, 71) logged states; a_exec: (k, n, 29); xi: (n, 33);
        done: (k, n) termination label of states 1..k; refs[j]: reference of transition j (for the
        termination features of state j+1).
        """
        k = a_exec.shape[0]
        s = states[0]
        total_state = 0.0
        total_term = 0.0
        metrics = {}
        err_first = None
        err_last = None
        acc_num = 0.0
        step_fn = getattr(self, "_loss_step_fn", None) or self._loss_step
        for j in range(k):
            s, sq, bce, logits = step_fn(s, a_exec[j], xi, states[j + 1], done[j], refs[j].root_pos_w, refs[j].root_quat_w)
            total_state = total_state + sq.sum(-1).mean()
            total_term = total_term + bce.mean()
            with torch.no_grad():
                nse = sq.mean(-1).mean()
                if j == 0:
                    err_first = nse
                err_last = nse
                acc_num += ((logits > 0).float() == done[j]).float().mean()
        loss = total_state / k + self.cfg.term_loss_weight * total_term / k
        # Metrics stay on-device (detached tensors): a float() here would stall the CPU on the
        # forward pass every minibatch, which in data-parallel runs turns into lockstep waiting.
        metrics.update(
            state_loss=(total_state / k).detach(),
            term_bce=(total_term / k).detach(),
            nse_1step=err_first.detach(),
            nse_kstep=err_last.detach(),
            term_acc=(acc_num / k).detach(),
        )
        return loss, metrics

    # ------------------------------------------------------------------ diagnostics

    @torch.no_grad()
    def rollout_error(self, states: torch.Tensor, a_exec: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """Open-loop k-step normalised squared error per step (k,), averaged over the batch."""
        k = a_exec.shape[0]
        s = states[0]
        out = []
        for j in range(k):
            s = self.step(s, a_exec[j], xi)
            out.append(self.spec.normalized_sq_error(s, states[j + 1]).mean())
        return torch.stack(out)
