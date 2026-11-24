"""World-model state and transition semantics for a free-root robot with J joints.

States retain the packed order [position, wxyz quaternion, joints, root linear
velocity, root angular velocity, joint velocities]. PartyMimic stores both root
velocities in world coordinates. Its WM sees both velocities in the current
body frame and predicts their world-vector changes in that same body frame.
Pose corrections are in the current heading, body and joint frames respectively.

The legacy representation stores MuJoCo free-joint velocities (linear in world,
angular in body), uses heading-frame linear inputs/deltas, and drifts at the
predicted endpoint velocity. Both modes use the configured physical scales.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from supermimic.kinematics.rotations import (
    heading_quat,
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_canonical,
    quat_from_rotvec,
    quat_mul,
    quat_normalize,
    rot6d_from_quat,
    rotvec_from_quat,
)
from supermimic.sim_context import CONTROL_SUBSTEPS, ContextKind, SimContextLayout

STATE_DIM = 71
FEAT_DIM = 71
DELTA_DIM = 70
NJ = 29
WMRepresentation = Literal["partymimic", "legacy"]
# Mean of the n semi-implicit substep velocities under constant acceleration.
SUBSTEP_COEFF = (CONTROL_SUBSTEPS + 1) / (2 * CONTROL_SUBSTEPS)
DEFAULT_XI_OFFSET = (0.75,) + (0.0,) * NJ + (0.0,) * 3
DEFAULT_XI_SCALE = (0.45,) + (0.01,) * NJ + (0.05,) * 3


def _party_heading_xy(q: torch.Tensor) -> torch.Tensor:
    """PartyMimic yaw_mat's regularized projection of the unit root's forward axis."""
    w, x, y, z = q.unbind(-1)
    forward = torch.stack((1 - 2 * (y.square() + z.square()), 2 * (x * y + w * z)), -1)
    return forward / torch.sqrt(forward.square().sum(-1, keepdim=True) + 1e-9)


def _apply_heading(heading_xy: torch.Tensor, vec: torch.Tensor, *, transpose: bool = False) -> torch.Tensor:
    """Apply H or Hᵀ without a batched matrix multiplication."""
    c, s = heading_xy.unbind(-1)
    if transpose:
        s = -s
    x, y, z = vec.unbind(-1)
    return torch.stack((c * x - s * y, s * x + c * y, z), -1)


@dataclass
class StateSpecCfg:
    dt: float = 0.02
    # Nominal error scales (D3) for [root pos m, root rotvec rad, root lin vel m/s, root ang vel rad/s, joint pos rad, joint vel rad/s].
    err_scale: tuple[float, ...] = (0.05, 0.1, 0.5, 1.0, 0.1, 2.0)
    # Output scales of the dynamics head for [dv, dw, dqd, dp_h, dth_b, dq].
    out_scale: tuple[float, ...] = (0.3, 0.5, 2.0, 0.01, 0.02, 0.02)
    # Input feature scales for [root_z, rot6d, lin_vel, ang_vel_b, joint_pos, joint_vel] (divide by these).
    in_scale: tuple[float, ...] = (1.0, 1.0, 1.0, 2.0, 1.0, 5.0)
    explicit_integration: bool = True
    # Training resolves the robot's layout; base preserves legacy standalone models.
    context_layout: ContextKind = "base"
    # Default vectors expand through the selected context layout.
    xi_offset: tuple[float, ...] = field(default_factory=lambda: DEFAULT_XI_OFFSET)
    xi_scale: tuple[float, ...] = field(default_factory=lambda: DEFAULT_XI_SCALE)
    representation: WMRepresentation | None = "partymimic"  # Training resolves None before use.


class StateSpec:
    def __init__(self, cfg: StateSpecCfg, device, dtype=torch.float32, num_joints: int = NJ):
        self.cfg = cfg
        self.dt = cfg.dt
        if cfg.representation not in ("partymimic", "legacy"):
            raise ValueError(f"Unknown WM representation: {cfg.representation!r}")
        if num_joints < 1:
            raise ValueError("num_joints must be positive")
        self.num_joints = int(num_joints)
        self.nq = 7 + self.num_joints
        self.nv = 6 + self.num_joints
        self.state_dim = self.feat_dim = self.nq + self.nv
        self.delta_dim = 12 + 2 * self.num_joints
        self.context = SimContextLayout(self.num_joints, cfg.context_layout)
        self.xi_dim = self.context.dim
        nj = self.num_joints
        rep = lambda scales, sizes: torch.cat([torch.full((n,), float(s)) for s, n in zip(scales, sizes)])
        self.err_scale = rep(cfg.err_scale, (3, 3, 3, 3, nj, nj)).to(device=device, dtype=dtype)
        self.out_scale = rep(cfg.out_scale, (3, 3, nj, 3, 3, nj)).to(device=device, dtype=dtype)
        self.in_scale = rep(cfg.in_scale, (1, 6, 3, 3, nj, nj)).to(device=device, dtype=dtype)
        # Expand the existing per-quantity defaults to this robot's joint count.
        # An explicitly customised vector must already have the correct width.
        offset = (self.context.offset
                  if cfg.xi_offset == DEFAULT_XI_OFFSET else cfg.xi_offset)
        scale = (self.context.scale
                 if cfg.xi_scale == DEFAULT_XI_SCALE else cfg.xi_scale)
        if len(offset) != self.xi_dim or len(scale) != self.xi_dim:
            raise ValueError(f"xi normalisation needs {self.xi_dim} values for {nj} joints")
        self.xi_offset = torch.tensor(offset, device=device, dtype=dtype)
        self.xi_scale = torch.tensor(scale, device=device, dtype=dtype)
        if not torch.isfinite(self.xi_offset).all() or not torch.isfinite(self.xi_scale).all() or (self.xi_scale <= 0).any():
            raise ValueError("xi normalisation must be finite with positive scales")

    # ------------------------------------------------------------------ split / features

    @staticmethod
    def split(s: torch.Tensor):
        """Split a free-root humanoid state while preserving the G1 static API."""
        if s.shape[-1] < 15 or (s.shape[-1] - 13) % 2:
            raise ValueError(f"Invalid humanoid state width: {s.shape[-1]}")
        nj = (s.shape[-1] - 13) // 2
        nq = 7 + nj
        return (s[..., 0:3], s[..., 3:7], s[..., 7:nq], s[..., nq:nq + 3],
                s[..., nq + 3:nq + 6], s[..., nq + 6:])

    def canonical(self, s: torch.Tensor) -> torch.Tensor:
        """Scaled WM input with both velocities in the body frame for PartyMimic."""
        return self._features(s, body_linear=self.cfg.representation == "partymimic")

    def critic_features(self, s: torch.Tensor) -> torch.Tensor:
        """Critic observations keep heading-frame linear and body-frame angular velocity."""
        return self._features(s, body_linear=False)

    def _features(self, s: torch.Tensor, *, body_linear: bool) -> torch.Tensor:
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        if body_linear:
            heading = _party_heading_xy(q)
            columns = matrix_from_quat(q)[..., :, :2].transpose(-1, -2)
            tilt_columns = _apply_heading(heading.unsqueeze(-2), columns, transpose=True)
            tilt = tilt_columns.transpose(-1, -2).flatten(-2)
            v_local = quat_apply_inverse(q, v)
        else:
            hq = heading_quat(q)
            q_local = quat_canonical(quat_mul(quat_apply_inverse_quat(hq), q))
            tilt = rot6d_from_quat(q_local)
            v_local = quat_apply_inverse(hq, v)
        w_body = quat_apply_inverse(q, w) if self.cfg.representation == "partymimic" else w
        feat = torch.cat([p[..., 2:3], tilt, v_local, w_body, qj, qd], dim=-1)
        return feat / self.in_scale

    def normalize_xi(self, xi: torch.Tensor) -> torch.Tensor:
        return (xi - self.xi_offset) / self.xi_scale

    # ------------------------------------------------------------------ integration

    def integrate(self, s: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Scale the six head outputs to physical units and advance one control step."""
        d = delta * self.out_scale
        nj = self.num_joints
        dv, dw, dqd, dp_h, dth_b, dq = d.split((3, 3, nj, 3, 3, nj), dim=-1)
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        dt = self.dt if self.cfg.explicit_integration else 0.0
        qd_new = qd + dqd
        if self.cfg.representation == "partymimic":
            dv_w = quat_apply(q, dv)
            v_new = v + dv_w
            w_new = w + quat_apply(q, dw)
            # Mix endpoints in a common frame. The body-frame rotation drift
            # equals R^T times the mixed world angular velocity.
            v_pose = v + SUBSTEP_COEFF * dv_w
            w_pose_b = quat_apply_inverse(q, w) + SUBSTEP_COEFF * dw
            qd_pose = qd + SUBSTEP_COEFF * dqd
            # Near a vertical root forward axis, the regularized H Hᵀ is
            # not the identity. Retain both transforms from PartyMimic.
            heading = _party_heading_xy(q)
            drift_h = _apply_heading(heading, dt * v_pose, transpose=True)
            p_new = p + _apply_heading(heading, dp_h + drift_h)
        else:
            hq = heading_quat(q)
            v_new = v + quat_apply(hq, dv)
            w_new = w + dw
            w_pose_b, qd_pose = w_new, qd_new
            p_new = p + dt * v_new + quat_apply(hq, dp_h)

        q_new = quat_normalize(quat_mul(q, quat_from_rotvec(dt * w_pose_b + dth_b)))
        qj_new = qj + dt * qd_pose + dq
        return torch.cat([p_new, q_new, qj_new, v_new, w_new, qd_new], dim=-1)

    # ------------------------------------------------------------------ errors

    def error_vector(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """Return [dp, rotvec, dv, dw, dq, dqd] divided by the fixed nominal scales."""
        p1, q1, qj1, v1, w1, qd1 = self.split(s_pred)
        p2, q2, qj2, v2, w2, qd2 = self.split(s_true)
        q1 = quat_normalize(q1)
        q2 = quat_normalize(q2)
        rot_err = rotvec_from_quat(quat_mul(quat_apply_inverse_quat(q2), q1))
        dp, dv, dw = p1 - p2, v1 - v2, w1 - w2
        if self.cfg.representation == "partymimic":
            dp = _apply_heading(_party_heading_xy(q2), dp, transpose=True)
            dv = quat_apply_inverse(q2, dv)
            dw = quat_apply_inverse(q2, dw)
        err = torch.cat([dp, rot_err, dv, dw, qj1 - qj2, qd1 - qd2], dim=-1)
        return err / self.err_scale

    def normalized_sq_error(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """Mean over dims of the squared normalised error (the reported 'prediction error')."""
        e = self.error_vector(s_pred, s_true)
        return (e * e).mean(-1)


def quat_apply_inverse_quat(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a unit quaternion (inverse)."""
    # A fresh CPU literal copied to CUDA is not legal inside graph capture.
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def _convert_root_velocity(
    s: torch.Tensor, representation: WMRepresentation, *, to_mujoco: bool,
) -> torch.Tensor:
    if representation == "legacy":
        return s
    if representation != "partymimic":
        raise ValueError(f"Unknown WM representation: {representation!r}")
    p, q, qj, v, w, qd = StateSpec.split(s)
    rotation = quat_normalize(q)
    angular = quat_apply_inverse(rotation, w) if to_mujoco else quat_apply(rotation, w)
    return torch.cat((p, q, qj, v, angular, qd), dim=-1)


def from_mujoco_state(s: torch.Tensor, representation: WMRepresentation) -> torch.Tensor:
    """Convert a packed [qpos | qvel] snapshot to the configured WM replay format."""
    return _convert_root_velocity(s, representation, to_mujoco=False)


def to_mujoco_state(s: torch.Tensor, representation: WMRepresentation) -> torch.Tensor:
    """Recover the packed [qpos | qvel] expected by MuJoCo and FK."""
    return _convert_root_velocity(s, representation, to_mujoco=True)
