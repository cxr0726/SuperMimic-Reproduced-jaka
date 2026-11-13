"""Imagined rollouts through the world model (paper §3.3)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import torch
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from supermimic.data.replay_buffer import StartBatch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.compiled import REWARD_TERM_NAMES, ImagStep
from supermimic.observation_layout import (
    critic_reference_extras,
    isaac_anchor_orientation,
    isaac_goal_command,
    resolve_observation_sizes,
)
from supermimic.policy.nets import Policy
from supermimic.world_model.heads import (
    ReferenceFrame,
    analytic_observation,
    anchor_relative_obs,
    flatten_history_term_major,
)
from supermimic.world_model.wm_trainer import reference_at
from supermimic.world_model.world_model import WorldModel

# Paper Table A.1 sensor noise (uniform half-widths) in the 93-d observation order.
OBS_NOISE_HALF = torch.tensor([0.01] * 29 + [0.5] * 29 + [0.2] * 3 + [0.05] * 3 + [0.0] * 29)
ANCHOR_NOISE_HALF = torch.tensor([0.25] * 3 + [0.05] * 6)


def observation_noise_half(
    num_joints: int, *, device=None, dtype=torch.float32, observation_layout: str = "supermimic",
) -> torch.Tensor:
    """Sensor noise in the chosen layout's units (Khan observations are pre-scaled)."""
    if observation_layout == "isaac_khan":
        # Noise is added after analytic scaling here, equivalent to the
        # environment's noise-before-scale pipeline: angular .2*.5, qd .5*.2.
        return torch.tensor(
            [0.07] * 3 + [0.1] * 3 + [0.01] * num_joints
            + [0.1] * num_joints + [0.0] * num_joints, device=device, dtype=dtype,
        )
    return torch.tensor(
        [0.01] * num_joints + [0.5] * num_joints + [0.2] * 3
        + [0.05] * 3 + [0.0] * num_joints,
        device=device, dtype=dtype,
    )


@cache
def _noise_constants(
    num_joints: int, device: torch.device, dtype: torch.dtype, observation_layout: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read-only device constants, allocated once by the eager graph warmup.

    Constructing a tensor from a Python list, or transferring the CPU anchor
    constant on each call, performs a host-to-device copy that CUDA capture
    rejects. Cache only this internal use; the public noise-vector helper keeps
    returning an independent tensor and consumes no random draws. Do not evict
    these tiny constants while a graph can still hold their device addresses.
    """
    return (
        observation_noise_half(num_joints, device=device, dtype=dtype,
                               observation_layout=observation_layout),
        ANCHOR_NOISE_HALF.to(device),
    )


@dataclass
class ImagRollout:
    states: torch.Tensor  # (H+1, n, 71)
    actions: torch.Tensor  # (H, n, 29)
    rewards: torch.Tensor  # (H, n)      r_{t+1}
    dhat: torch.Tensor  # (H+1, n)    predicted termination prob of s_t (d_0 = 0)
    survival: torch.Tensor  # (H+1, n)    c_t
    goals: torch.Tensor  # (H+1, n, 680)
    e_root: torch.Tensor  # (H+1, n, 4)
    frames: torch.Tensor  # (H+1, n)
    reward_terms: dict  # mean of each reward term over the rollout (diagnostics)
    critic_goals: torch.Tensor | None = None  # Khan: command155 + orientation30 + velocity/gravity180
    critic_reference: ReferenceFrame | None = None  # Khan current frames, flattened (H+1)*n
    critic_body_pos: torch.Tensor | None = None  # (H+1, n, tracked bodies, 3), actual world poses
    critic_body_quat: torch.Tensor | None = None  # (H+1, n, tracked bodies, 4), only final poses retain grad


def imagine(
    wm: WorldModel,
    policy: Policy,
    lib: MotionLibrary,
    start: StartBatch,
    horizon: int,
    future_frames: int | None = None,
    obs_noise: bool = True,
    step_fn: ImagStep | None = None,
    anchor_obs: bool = False,
    activation_checkpoint: bool = False,
    batch_reference_encoder: bool = False,
    tensor_metrics: bool = False,
    validate_horizon: bool = True,
    checkpoint_preserve_rng_state: bool = True,
    encode_goal_fn=None,
    decode_from_latent_fn=None,
    policy_forward_fn=None,
) -> ImagRollout:
    """Imagine H transitions, optionally sharing one reference-encoder call.

    ``tensor_metrics`` avoids diagnostic host reads inside a captured update.
    ``validate_horizon=False`` is reserved for starts already checked by the
    replay sampler. State-dependent legacy anchor goals stay stepwise.
    """
    device = start.state.device
    # Trainer-owned compiled callables keep collection, registered modules and
    # deployment state_dicts unchanged. Standalone callers retain eager methods.
    encode_goal_fn = getattr(policy, "encode_goal", None) if encode_goal_fn is None else encode_goal_fn
    decode_from_latent_fn = getattr(policy, "decode_from_latent", None) if decode_from_latent_fn is None else decode_from_latent_fn
    policy_forward_fn = policy if policy_forward_fn is None else policy_forward_fn
    n = start.state.shape[0]
    if horizon < 1:
        raise ValueError("imagination horizon must be positive")
    if validate_horizon:
        last = lib.last_frame_of(start.frame)
        if bool((start.frame + horizon > last).any()):
            raise ValueError("an imagined start must have a full reference horizon within its clip")
    q0 = lib.default_joint_pos
    layout = getattr(policy.cfg, "observation_layout", "supermimic")
    if future_frames is None:
        future_frames = getattr(policy.cfg, "future_frames", 10)
    if layout == "isaac_khan":
        resolve_observation_sizes(layout, future_frames, start.hist_states.shape[0], anchor_obs)
    step_fn = step_fn or ImagStep(wm, q0, observation_layout=layout,
                                 return_body_poses=layout == "isaac_khan")
    if getattr(step_fn, "observation_layout", layout) != layout:
        raise ValueError("Imagined observation layout differs from the policy")
    returns_body_poses = getattr(step_fn, "return_body_poses", False)
    cache_body_poses = layout == "isaac_khan" and returns_body_poses
    term_names = getattr(step_fn, "term_names", REWARD_TERM_NAMES)
    noise_half, anchor_noise_half = _noise_constants(q0.numel(), device, start.state.dtype, layout)
    ai = wm.heads.anchor_index
    batch_encoder = (batch_reference_encoder and isinstance(policy, Policy)
                     and (layout == "isaac_khan" or not anchor_obs))
    reference_goals = reference_orientations = reference_extras = None
    orientation_noise = encoder_goals = sensor_noise = None

    def add_noise(o):
        if obs_noise:
            return o + (torch.rand_like(o) * 2 - 1) * noise_half
        return o

    def make_inputs(frame_t, ref_t, anchor_pose, time):
        if layout == "isaac_khan":
            if batch_encoder:
                g = reference_goals[time]
                orientation = isaac_anchor_orientation(anchor_pose[..., 3:7], reference_orientations[time])
                extras = reference_extras[time]
            else:
                window = lib.reference_window(frame_t, future_frames)
                g = isaac_goal_command(window.root_pos_w, window.root_quat_w, window.joint_pos)
                orientation = isaac_anchor_orientation(anchor_pose[..., 3:7], window.root_quat_w)
                extras = critic_reference_extras(
                    window.joint_vel, window.root_quat_w,
                    window.root_ang_vel_w, window.root_lin_vel_w,
                )
            # Moving future orientation from actor goal to proprioception must
            # not remove it from value estimation. Like Isaac's privileged
            # observations, this critic reference contains no sensor noise.
            critic_goal = torch.cat([g, orientation, extras], dim=-1)
            if obs_noise:
                delta = (orientation_noise[time] if batch_encoder else
                         (torch.rand_like(orientation) * 2 - 1) * 0.05)
                orientation = orientation + delta
            return g, orientation, critic_goal
        g = reference_goals[time] if batch_encoder else lib.goal_window(frame_t, future_frames)
        if not anchor_obs:
            return g, None, g
        rel = anchor_relative_obs(anchor_pose[..., 0:3], anchor_pose[..., 3:7], ref_t.body_pos_w[..., ai, :], ref_t.body_quat_w[..., ai, :])
        if obs_noise:
            rel = rel + (torch.rand_like(rel) * 2 - 1) * anchor_noise_half
        g = torch.cat([g, rel], dim=-1)
        return g, None, g

    # Initial observation history from the logged states/actions (noise re-sampled per frame).
    hist = [add_noise(analytic_observation(start.hist_states[i], start.hist_actions[i], start.xi,
                                         q0, layout)) for i in range(start.hist_states.shape[0])]
    if layout == "isaac_khan" and start.history_valid is not None:
        # A reset fills each history buffer with the same first noisy observation.
        # Preserve that correlation when reconstructing history from replay rows.
        history_tensor = torch.stack(hist)
        first = start.history_valid.long().argmax(dim=0)
        initial = history_tensor[first, torch.arange(n, device=device)]
        hist = list(torch.where(start.history_valid[..., None], history_tensor,
                                initial[None]).unbind(0))
    s = start.state
    frame = start.frame.clone()
    a_prev = start.prev_action
    c = torch.ones(n, device=device)

    states = [s]
    actions, rewards = [], []
    dhats, survivals, frames = [torch.zeros(n, device=device)], [c], [frame]
    # Starts have H real reference transitions. Future goal windows retain
    # their independent last-frame padding, including at the bootstrap state.
    frames_all = frame[None, :] + torch.arange(horizon + 1, device=device)[:, None]  # (H+1, n)
    ref_all = reference_at(lib, frames_all.reshape(-1))
    latents = None
    if batch_encoder:
        flat_frames = frames_all.flatten()
        if layout == "isaac_khan":
            window = lib.reference_window(flat_frames, future_frames)
            reference_goals = isaac_goal_command(
                window.root_pos_w, window.root_quat_w, window.joint_pos,
            ).reshape(horizon + 1, n, -1)
            reference_orientations = window.root_quat_w.reshape(horizon + 1, n, future_frames, 4)
            reference_extras = critic_reference_extras(
                window.joint_vel, window.root_quat_w,
                window.root_ang_vel_w, window.root_lin_vel_w,
            ).reshape(horizon + 1, n, -1)
        else:
            reference_goals = lib.goal_window(flat_frames, future_frames).reshape(horizon + 1, n, -1)

        # Reproduce the old per-step RNG order: orientation(t), encoder(t),
        # sensor(t+1), then orientation(t+1). Only sampling moves earlier;
        # deterministic Policy/ImagStep evaluation consumes no random draws.
        # Keeping these individual draws also preserves reset-backfill noise.
        orientation_noise, encoder_goals, sensor_noise = [], [], []
        for time in range(horizon + 1):
            if layout == "isaac_khan" and obs_noise:
                orientation_noise.append((torch.rand(
                    (n, 6 * future_frames), device=device, dtype=start.state.dtype,
                ) * 2 - 1) * 0.05)
            if time < horizon:
                encoder_goals.append(policy.make_encoder_goal(reference_goals[time], enabled=obs_noise))
                if obs_noise:
                    sensor_noise.append((torch.rand_like(hist[-1]) * 2 - 1) * noise_half)
        encoder_goals = torch.stack(encoder_goals).flatten(0, 1)
        clean_goals = reference_goals[:-1].flatten(0, 1)
        if activation_checkpoint:
            with set_checkpoint_early_stop(False):
                latents = checkpoint(
                    encode_goal_fn, clean_goals, encoder_goal=encoder_goals,
                    use_reentrant=False, preserve_rng_state=checkpoint_preserve_rng_state,
                )
        else:
            latents = encode_goal_fn(clean_goals, encoder_goal=encoder_goals)
        latents = latents.reshape(horizon, n, -1)

    def ref_at(t):
        sl = slice(t * n, (t + 1) * n)
        return ReferenceFrame(ref_all.body_pos_w[sl], ref_all.body_quat_w[sl],
                              ref_all.body_lin_vel_w[sl], ref_all.body_ang_vel_w[sl],
                              ref_all.root_pos_w[sl], ref_all.root_quat_w[sl],
                              ref_all.joint_pos[sl] if ref_all.joint_pos is not None else None,
                              ref_all.joint_vel[sl] if ref_all.joint_vel is not None else None)

    ref = ref_at(0)
    if anchor_obs or layout == "isaac_khan":
        b0 = wm.heads.robot_bodies(s)
        anchor_pose = torch.cat([b0["pos"][..., ai, :], b0["quat"][..., ai, :]], dim=-1)
    else:
        anchor_pose = None
    # The initial FK is already needed for actor anchor observations. Store
    # only actual geometry, never reward-specific reanchoring or ref_t errors.
    critic_body_pos = [b0["pos"].detach()] if cache_body_poses else None
    critic_body_quat = [b0["quat"].detach()] if cache_body_poses else None
    goal, future_orientation, critic_goal = make_inputs(frame, ref, anchor_pose, 0)
    goals = [goal]
    critic_goals = [critic_goal] if layout == "isaac_khan" else None
    e_roots = [wm.heads.root_error_features(s, ref)]
    term_sum = torch.zeros(len(term_names), device=device)

    for t in range(horizon):
        obs_flat = flatten_history_term_major(torch.stack(hist, 0), observation_layout=layout)
        if layout == "isaac_khan":
            obs_flat = torch.cat([future_orientation, obs_flat], dim=-1)
        # Sample outside activation recomputation. goals/critic_goals remain
        # clean; only the encoder consumes this independently corrupted copy.
        encoder_kwargs = ({"encoder_goal": policy.make_encoder_goal(goals[-1])}
                          if not batch_encoder and obs_noise and getattr(policy, "has_encoder_goal_noise", False)
                          else {})
        if activation_checkpoint:
            # Recompute deterministic activations during backward; the complete batch,
            # objective and update schedule are unchanged. Sensor noise stays outside.
            # TorchScript quaternion helpers wrap the checkpoint early-stop signal
            # as RuntimeError. Complete recomputation preserves eager compatibility.
            with set_checkpoint_early_stop(False):
                if batch_encoder:
                    a = checkpoint(decode_from_latent_fn, obs_flat, goals[-1], latents[t],
                                   use_reentrant=False, preserve_rng_state=checkpoint_preserve_rng_state)
                else:
                    a = checkpoint(policy_forward_fn, obs_flat, goals[-1], **encoder_kwargs,
                                   use_reentrant=False, preserve_rng_state=checkpoint_preserve_rng_state)
                step_result = checkpoint(
                    step_fn, s, a, a_prev, start.xi, ref,
                    use_reentrant=False, preserve_rng_state=checkpoint_preserve_rng_state,
                )
        else:
            a = (decode_from_latent_fn(obs_flat, goals[-1], latents[t]) if batch_encoder else
                 policy_forward_fn(obs_flat, goals[-1], **encoder_kwargs))
            step_result = step_fn(s, a, a_prev, start.xi, ref)
        if returns_body_poses:
            s_next, r, term_vec, logits, obs_clean, anchor_pose, body_pos, body_quat = step_result
            if cache_body_poses:
                # Earlier geometry is used only by detached critic fitting. Only
                # s_H's geometry needs BPTT for V(s_H); do not retain prefix FK graphs.
                critic_body_pos.append(body_pos if t == horizon - 1 else body_pos.detach())
                critic_body_quat.append(body_quat if t == horizon - 1 else body_quat.detach())
        else:
            # Preserve the six-output custom-step/G1 interface and FK fallback.
            s_next, r, term_vec, logits, obs_clean, anchor_pose = step_result
        d = torch.sigmoid(logits)
        c = c * (1.0 - d)
        term_sum += term_vec.detach().mean(-1)

        frame = frames_all[t + 1]
        ref = ref_at(t + 1)
        hist = hist[1:] + [(obs_clean + sensor_noise[t] if obs_noise else obs_clean)
                          if batch_encoder else add_noise(obs_clean)]
        a_prev = a
        s = s_next
        states.append(s)
        actions.append(a)
        rewards.append(r)
        dhats.append(d)
        survivals.append(c)
        goal, future_orientation, critic_goal = make_inputs(frame, ref, anchor_pose, t + 1)
        goals.append(goal)
        if critic_goals is not None:
            critic_goals.append(critic_goal)
        e_roots.append(wm.heads.root_error_features(s, ref))
        frames.append(frame)

    return ImagRollout(
        states=torch.stack(states),
        actions=torch.stack(actions),
        rewards=torch.stack(rewards),
        dhat=torch.stack(dhats),
        survival=torch.stack(survivals),
        goals=torch.stack(goals),
        e_root=torch.stack(e_roots),
        frames=torch.stack(frames),
        reward_terms=({k: v for k, v in zip(term_names, (term_sum / horizon).unbind())}
                      if tensor_metrics else
                      {k: float(v / horizon) for k, v in zip(term_names, term_sum.tolist())}),
        critic_goals=torch.stack(critic_goals) if critic_goals is not None else None,
        critic_reference=ref_all if layout == "isaac_khan" else None,
        critic_body_pos=torch.stack(critic_body_pos) if critic_body_pos is not None else None,
        critic_body_quat=torch.stack(critic_body_quat) if critic_body_quat is not None else None,
    )
