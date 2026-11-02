"""Short-horizon actor–critic through the frozen world model (paper §3.3, Table A.5)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.objective import gae_targets, policy_objective
from supermimic.imagination.rollout import ImagRollout, imagine
from supermimic.policy.nets import Critic, Policy
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


class PolicyTrainer:
    def __init__(self, wm: WorldModel, policy: Policy, critic: Critic, lib: MotionLibrary, cfg: PolicyTrainerCfg):
        self.wm, self.policy, self.critic, self.lib, self.cfg = wm, policy, critic, lib, cfg
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, betas=cfg.betas)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
        from supermimic.imagination.compiled import ImagStep

        self.step_fn = ImagStep(wm, lib.default_joint_pos, compile_mode=cfg.compile,
                               observation_layout=cfg.observation_layout)

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr
        for g in self.critic_opt.param_groups:
            g["lr"] = lr

    def _critic_inputs(self, ro: ImagRollout):
        H1, n, _ = ro.states.shape
        states = ro.states.reshape(H1 * n, -1)
        feat = self.wm.spec.canonical(states).reshape(H1, n, -1)
        goals = ro.critic_goals if ro.critic_goals is not None else ro.goals
        errors = None
        if ro.critic_reference is not None:
            reference = ro.critic_reference

            def ref_slice(sl):
                return ReferenceFrame(
                    reference.body_pos_w[sl], reference.body_quat_w[sl],
                    reference.body_lin_vel_w[sl], reference.body_ang_vel_w[sl],
                    reference.root_pos_w[sl], reference.root_quat_w[sl],
                    reference.joint_pos[sl] if reference.joint_pos is not None else None,
                    reference.joint_vel[sl] if reference.joint_vel is not None else None,
                )

            # Only V(s_H) contributes to actor BPTT. Earlier errors are used
            # exclusively by the detached critic fit; do not retain H redundant
            # FK graphs for those states in a full-size training batch.
            with torch.no_grad():
                prefix = self.wm.heads.critic_error_features(states[:-n], ref_slice(slice(None, -n)))
            bootstrap = self.wm.heads.critic_error_features(states[-n:], ref_slice(slice(-n, None)))
            errors = torch.cat([prefix, bootstrap], dim=0).reshape(H1, n, -1)
        return feat, goals, ro.e_root, errors

    def update(self, buffer: ReplayBuffer, xi_norm_fn=None) -> dict[str, float]:
        cfg = self.cfg
        start = buffer.sample_starts(
            cfg.batch_size, cfg.history, horizon=cfg.horizon,
            clip_last=self.lib.clip_last,
        )
        xi_norm = self.wm.spec.normalize_xi(start.xi)

        # ---- actor: BPTT through the frozen world model ------------------------------------
        self.wm.requires_grad_(False)
        self.critic.requires_grad_(False)
        ro = imagine(self.wm, self.policy, self.lib, start, cfg.horizon, cfg.future_frames, obs_noise=cfg.obs_noise, step_fn=self.step_fn, anchor_obs=cfg.anchor_obs, activation_checkpoint=cfg.activation_checkpoint)
        feat, goals, e_root, errors = self._critic_inputs(ro)
        if cfg.use_critic:
            v_H = self.critic(feat[-1], goals[-1], xi_norm, e_root[-1],
                              errors[-1] if errors is not None else None)
        else:
            v_H = None
        J = policy_objective(ro, v_H, cfg.gamma)
        loss = -J / cfg.horizon
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        allreduce_grads(self.policy)
        gn = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip_norm)
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), cfg.grad_clip_value)
        self.opt.step()
        self.wm.requires_grad_(True)
        self.critic.requires_grad_(True)

        metrics = {
            "imag_objective": float(J.detach()),
            "imag_return_undisc": float((ro.survival[:-1] * ro.rewards).sum(0).mean().detach()),
            "imag_reward_mean": float(ro.rewards.mean().detach()),
            "imag_survival_H": float(ro.survival[-1].mean().detach()),
            "imag_dhat_mean": float(ro.dhat[1:].mean().detach()),
            "policy_grad_norm": float(gn),
            "action_abs_mean": float(ro.actions.abs().mean().detach()),
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
                targets = gae_targets(ro.rewards.detach(), ro.dhat[1:].detach(), values, cfg.gamma, cfg.gae_lambda)  # (H, n)
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
                    self.critic_opt.zero_grad(set_to_none=True)
                    closs.backward()
                    allreduce_grads(self.critic)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.critic_grad_clip)
                    self.critic_opt.step()
                    last = closs.detach()
            metrics["critic_loss"] = float(last)
            metrics["critic_value_mean"] = float(values.mean())
            metrics["critic_target_mean"] = float(targets.mean())
        return metrics
