"""Short-horizon actor–critic through the frozen world model (paper §3.3, Table A.5)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.objective import gae_targets, policy_objective
from supermimic.imagination.rollout import ImagRollout, imagine
from supermimic.policy.ema import ActorEMA
from supermimic.policy.nets import Critic, Policy
from supermimic.training.cuda_graph import CapturedUpdate, graph_state, prepare_adam_for_capture
from supermimic.training.distributed import allreduce_grads
from supermimic.world_model.heads import ReferenceFrame
from supermimic.world_model.world_model import WorldModel


@dataclass
class PolicyTrainerCfg:
    horizon: int = 32
    batch_size: int = 4096
    history: int = 10
    future_frames: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip_norm: float = 1.0
    grad_clip_value: float = 0.1
    critic_lr: float = 2e-4
    critic_grad_clip: float = 1.0
    critic_epochs: int = 5
    critic_minibatches: int = 4
    use_critic: bool = True
    obs_noise: bool = True
    compile: str | None = "default"
    anchor_obs: bool = False  # None | "default" | "reduce-overhead" | "max-autotune-no-cudagraphs"
    activation_checkpoint: bool = False  # recompute activations; preserve the full effective batch
    observation_layout: str = "supermimic"
    ema_decay: float = 0.995  # Post-actor-update deployment average; zero disables it.
    critic_value_bound: bool = False  # Historical configs had no bootstrap/target upper bound.
    batch_reference_encoder: bool = True
    cuda_graph: bool = False  # Execution-only: full H32 actor + critic update.
    cuda_graph_warmup: int = 3  # Counted training updates, not extra optimizer steps.


def critic_value_ceiling(max_step_reward: float, gamma: float) -> float:
    """Analytic upper bound; negative rewards/values deliberately have no floor."""
    if not math.isfinite(gamma) or not 0 <= gamma < 1:
        raise ValueError("A critic value ceiling requires 0 <= gamma < 1")
    if not math.isfinite(max_step_reward) or max_step_reward <= 0:
        raise ValueError("A critic value ceiling requires a positive finite max step reward")
    return max_step_reward / (1.0 - gamma)


@torch.no_grad()
def critic_bound_diagnostics(ceiling: float, *, tensor_metrics: bool = False,
                             **raw_values: torch.Tensor) -> dict:
    """Summarize unclipped predictions/targets with a single device-to-host copy.

    P99 uses the empirical nearest-rank percentile and ``kthvalue`` rather than
    sorting an entire H*batch array. Diagnostics never enter either loss graph.
    """
    names, statistics = [], []
    for name, value in raw_values.items():
        flat = value.detach().reshape(-1).float()
        if not flat.numel():
            continue
        rank = max(1, math.ceil(.99 * flat.numel()))
        statistics.extend(((flat > ceiling).float().mean(), flat.max(),
                           flat.kthvalue(rank).values))
        names.extend((f"critic_{name}_raw_exceed_fraction", f"critic_{name}_raw_max",
                      f"critic_{name}_raw_p99"))
    if not statistics:
        return {}
    values = torch.stack(statistics)
    return dict(zip(names, values.unbind() if tensor_metrics else values.cpu().tolist(), strict=True))


class PolicyTrainer:
    def __init__(self, wm: WorldModel, policy: Policy, critic: Critic, lib: MotionLibrary, cfg: PolicyTrainerCfg):
        self.wm, self.policy, self.critic, self.lib, self.cfg = wm, policy, critic, lib, cfg
        if not math.isfinite(cfg.ema_decay) or not 0 <= cfg.ema_decay < 1:
            raise ValueError("ema_decay must lie in [0, 1); zero disables actor EMA")
        # The reward ceiling is part of the actual robot's analytic reward cfg,
        # so adding Khan neck tracking gives 11.2 instead of G1's 10.0. Keep
        # legacy lightweight/mock trainers working when the bound is disabled.
        reward_cfg = getattr(getattr(wm, "heads", None), "reward_cfg", None)
        self.value_ceiling = None
        if cfg.use_critic and reward_cfg is not None and 0 <= cfg.gamma < 1:
            self.value_ceiling = critic_value_ceiling(reward_cfg.max_step_reward, cfg.gamma)
        if cfg.use_critic and cfg.critic_value_bound and self.value_ceiling is None:
            raise ValueError("Critic value bounds need the world model's reward ceiling and gamma < 1")
        self.ema = ActorEMA(policy, cfg.ema_decay) if cfg.ema_decay > 0 else None
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, betas=cfg.betas)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
        from supermimic.imagination.compiled import ImagStep

        self._active_buffer: ReplayBuffer | None = None
        self._prepared_starts = None
        device = next(policy.parameters()).device
        self._capture_enabled = bool(cfg.cuda_graph and device.type == "cuda" and not (
            torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
        ))
        self.step_fn = ImagStep(
            wm, lib.default_joint_pos, compile_mode=None if self._capture_enabled else cfg.compile,
            observation_layout=cfg.observation_layout,
            return_body_poses=cfg.use_critic and cfg.observation_layout == "isaac_khan",
        )
        if self._capture_enabled and cfg.compile:
            # The outer recording owns graph replay. Keep step fusion without
            # a second, nested cudagraph that could overwrite BPTT activations.
            self.step_fn._fn = torch.compile(
                self.step_fn._step, dynamic=False, options={"triton.cudagraphs": False},
            )
        if self._capture_enabled:
            prepare_adam_for_capture(self.opt, device)
            prepare_adam_for_capture(self.critic_opt, device)
            if self.ema is not None:
                self.ema.enable_cuda_graph()
        elif cfg.cuda_graph:
            print("[INFO] Actor/Critic CUDA Graph: eager fallback (CPU or distributed gradient reduction)")
        self._captured_update = CapturedUpdate(
            self._update_tensors, self._graph_state, enabled=self._capture_enabled,
            warmup=cfg.cuda_graph_warmup, label="Actor/Critic",
        )

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            if isinstance(g["lr"], torch.Tensor):
                g["lr"].fill_(lr)
            else:
                g["lr"] = lr
        for g in self.critic_opt.param_groups:
            if isinstance(g["lr"], torch.Tensor):
                g["lr"].fill_(lr)
            else:
                g["lr"] = lr

    def _graph_state(self):
        modules = [self.wm, self.policy, self.critic]
        extra = []
        if self.ema is not None:
            modules.append(self.ema.module)
            if self.ema._device_updates is not None:
                extra.append(self.ema._device_updates)
        if self._active_buffer is not None:
            extra.extend(v for v in vars(self._active_buffer).values() if isinstance(v, torch.Tensor))
        extra.extend(v for v in vars(self.lib).values() if isinstance(v, torch.Tensor))
        if self._prepared_starts is not None:
            extra.append(self._prepared_starts.weights)
        return graph_state(modules, (self.opt, self.critic_opt), extra)

    def reset_capture(self) -> None:
        """Optimizer restore may replace step/LR storage; record a fresh graph."""
        self._captured_update.invalidate()
        if self._capture_enabled:
            device = next(self.policy.parameters()).device
            prepare_adam_for_capture(self.opt, device)
            prepare_adam_for_capture(self.critic_opt, device)

    def _critic_inputs(self, ro: ImagRollout):
        H1, n, _ = ro.states.shape
        states = ro.states.reshape(H1 * n, -1)
        feat = self.wm.spec.canonical(states).reshape(H1, n, -1)
        goals = ro.critic_goals if ro.critic_goals is not None else ro.goals
        errors = None
        if ro.critic_reference is not None:
            reference = ro.critic_reference
            if (ro.critic_body_pos is None) != (ro.critic_body_quat is None):
                raise ValueError("Critic body position and quaternion caches must both be provided")
            body_pos = ro.critic_body_pos.flatten(0, 1) if ro.critic_body_pos is not None else None
            body_quat = ro.critic_body_quat.flatten(0, 1) if ro.critic_body_quat is not None else None

            def ref_slice(sl):
                return ReferenceFrame(
                    reference.body_pos_w[sl], reference.body_quat_w[sl],
                    reference.body_lin_vel_w[sl], reference.body_ang_vel_w[sl],
                    reference.root_pos_w[sl], reference.root_quat_w[sl],
                    reference.joint_pos[sl] if reference.joint_pos is not None else None,
                    reference.joint_vel[sl] if reference.joint_vel is not None else None,
                )

            def error_slice(sl):
                if body_pos is None:
                    # Custom six-output steps retain the original FK path.
                    return self.wm.heads.critic_error_features(states[sl], ref_slice(sl))
                return self.wm.heads.critic_error_features(
                    states[sl], ref_slice(sl), {"pos": body_pos[sl], "quat": body_quat[sl]},
                )

            # Only V(s_H) contributes to actor BPTT. Earlier errors are used
            # exclusively by the detached critic fit; do not retain H redundant
            # FK graphs for those states in a full-size training batch.
            with torch.no_grad():
                prefix = error_slice(slice(None, -n))
            bootstrap = error_slice(slice(-n, None))
            errors = torch.cat([prefix, bootstrap], dim=0).reshape(H1, n, -1)
        return feat, goals, ro.e_root, errors

    def update(self, buffer: ReplayBuffer, xi_norm_fn=None) -> dict[str, float]:
        self._active_buffer = buffer
        self._prepared_starts = (buffer.prepare_starts(
            self.cfg.history, horizon=self.cfg.horizon, clip_last=self.lib.clip_last,
        ) if hasattr(buffer, "prepare_starts") else None)
        if self._capture_enabled and self._prepared_starts is None:
            raise TypeError("Captured policy updates require prepared GPU replay sampling")
        metrics = self._captured_update()
        # Exactly one device-to-host metric copy outside the update graph.
        names = list(metrics)
        values = torch.stack([torch.as_tensor(metrics[k], device=next(self.policy.parameters()).device)
                              for k in names]).detach().cpu().tolist()
        return dict(zip(names, values, strict=True))

    def _update_tensors(self) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        buffer = self._active_buffer
        sampling = {"prepared": self._prepared_starts} if self._prepared_starts is not None else {}
        start = buffer.sample_starts(
            cfg.batch_size, cfg.history, horizon=cfg.horizon,
            clip_last=self.lib.clip_last,
            **sampling,
        )
        xi_norm = self.wm.spec.normalize_xi(start.xi)

        # ---- actor: BPTT through the frozen world model ------------------------------------
        self.wm.requires_grad_(False)
        self.critic.requires_grad_(False)
        ro = imagine(
            self.wm, self.policy, self.lib, start, cfg.horizon, cfg.future_frames,
            obs_noise=cfg.obs_noise, step_fn=self.step_fn, anchor_obs=cfg.anchor_obs,
            activation_checkpoint=cfg.activation_checkpoint,
            batch_reference_encoder=cfg.batch_reference_encoder, tensor_metrics=True,
            validate_horizon=self._prepared_starts is None,
            checkpoint_preserve_rng_state=not self._capture_enabled,
        )
        feat, goals, e_root, errors = self._critic_inputs(ro)
        raw_bootstrap = None
        if cfg.use_critic:
            raw_bootstrap = self.critic(feat[-1], goals[-1], xi_norm, e_root[-1],
                                        errors[-1] if errors is not None else None)
            v_H = (raw_bootstrap.clamp(max=self.value_ceiling)
                   if cfg.critic_value_bound else raw_bootstrap)
        else:
            v_H = None
        J = policy_objective(ro, v_H, cfg.gamma)
        loss = -J / cfg.horizon
        self.opt.zero_grad(set_to_none=not self._capture_enabled)
        loss.backward()
        allreduce_grads(self.policy)
        gn = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip_norm)
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), cfg.grad_clip_value)
        self.opt.step()
        if self.ema is not None:
            self.ema.update(self.policy)
        self.wm.requires_grad_(True)
        self.critic.requires_grad_(True)

        metrics = {
            "imag_objective": J.detach(),
            "imag_return_undisc": (ro.survival[:-1] * ro.rewards).sum(0).mean().detach(),
            "imag_reward_mean": ro.rewards.mean().detach(),
            "imag_survival_H": ro.survival[-1].mean().detach(),
            "imag_dhat_mean": ro.dhat[1:].mean().detach(),
            "policy_grad_norm": gn.detach(),
            "action_abs_mean": ro.actions.abs().mean().detach(),
        }
        for k, v in ro.reward_terms.items():
            metrics[f"imag_r/{k}"] = v

        # ---- critic: fit to GAE targets of this rollout (targets fixed) ---------------------
        if cfg.use_critic:
            with torch.no_grad():
                feat_d, goals_d, e_root_d = feat.detach(), goals.detach(), e_root.detach()
                errors_d = errors.detach() if errors is not None else None
                H1, n, _ = feat_d.shape
                xi_rep = xi_norm[None].expand(H1, n, -1)
                values = self.critic(feat_d.reshape(H1 * n, -1), goals_d.reshape(H1 * n, -1),
                                     xi_rep.reshape(H1 * n, -1), e_root_d.reshape(H1 * n, -1),
                                     errors_d.reshape(H1 * n, -1) if errors_d is not None else None).reshape(H1, n)
                raw_targets = gae_targets(ro.rewards.detach(), ro.dhat[1:].detach(), values, cfg.gamma, cfg.gae_lambda)  # (H, n)
                targets = (raw_targets.clamp(max=self.value_ceiling)
                           if cfg.critic_value_bound else raw_targets)
                X_feat = feat_d[:-1].reshape(-1, feat_d.shape[-1])
                X_goal = goals_d[:-1].reshape(-1, goals_d.shape[-1])
                X_xi = xi_rep[:-1].reshape(-1, xi_rep.shape[-1])
                X_er = e_root_d[:-1].reshape(-1, 4)
                X_errors = errors_d[:-1].reshape(-1, errors_d.shape[-1]) if errors_d is not None else None
                Y = targets.reshape(-1)
            N = Y.shape[0]
            mb = N // cfg.critic_minibatches
            last = torch.zeros((), device=Y.device)
            for _ in range(cfg.critic_epochs):
                perm = torch.randperm(N, device=Y.device)
                for i in range(cfg.critic_minibatches):
                    idx = perm[i * mb : (i + 1) * mb]
                    pred = self.critic(X_feat[idx], X_goal[idx], X_xi[idx], X_er[idx],
                                       X_errors[idx] if X_errors is not None else None)
                    closs = torch.nn.functional.mse_loss(pred, Y[idx])
                    self.critic_opt.zero_grad(set_to_none=not self._capture_enabled)
                    closs.backward()
                    allreduce_grads(self.critic)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.critic_grad_clip)
                    self.critic_opt.step()
                    last = closs.detach()
            metrics["critic_loss"] = last
            metrics["critic_value_mean"] = values.mean()
            metrics["critic_target_mean"] = targets.mean()
            if self.value_ceiling is not None:
                metrics["critic_value_ceiling"] = values.new_full((), self.value_ceiling)
                metrics["critic_value_bound_enabled"] = values.new_full((), float(cfg.critic_value_bound))
                metrics.update(critic_bound_diagnostics(
                    self.value_ceiling, tensor_metrics=True,
                    bootstrap=raw_bootstrap, value=values, target=raw_targets,
                ))
        return metrics
