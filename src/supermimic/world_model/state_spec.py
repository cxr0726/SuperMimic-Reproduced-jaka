"""State representation for the world model (plan D1–D3).

The dimensions below describe the default 29-joint G1; the same layout and
per-quantity scales apply to any single-free-root model, including 27-joint Khan.

Simulator state  s = [root_pos(3), root_quat wxyz(4), joint_pos(29), root_lin_vel_w(3),
                      root_ang_vel_b(3), joint_vel(29)]            (71, MuJoCo layout, origin-free)

Canonical features (network input, heading-invariant, 71):
    [root_z(1), rot6d(heading^-1 ⊗ root_quat)(6), root_lin_vel in heading frame(3),
     root_ang_vel_b(3), joint_pos(29), joint_vel(29)]

Network output ``delta`` (70) and the integration step (dt = 0.02 s):
    dv_h(3), dw_b(3), dqd(29)            velocity increments, heading / body frame
    dp_h(3), dth_b(3), dq(29)            position corrections on top of explicit integration
    v'  = v + R_h dv_h               w' = w + dw_b          qd' = qd + dqd
    p'  = p + dt v' + R_h dp_h       q' = q ⊗ exp(dt w' + dth_b)    qj' = qj + dt qd' + dq
With zero output the model extrapolates at constant velocity (a physical prior); the
last layer of the dynamics MLP is zero-initialised so training starts from it.

Loss / error normalisation (D3): per-quantity nominal scales sigma applied to the
70-d error vector [dp(3), rotvec(3), dv(3), dw(3), dq(29), dqd(29)].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from supermimic.kinematics.rotations import (
    heading_quat,
    quat_apply,
    quat_apply_inverse,
    quat_canonical,
    quat_from_rotvec,
    quat_mul,
    quat_normalize,
    rot6d_from_quat,
    rotvec_from_quat,
)
from supermimic.sim_context import ContextKind, SimContextLayout

STATE_DIM = 71
FEAT_DIM = 71
DELTA_DIM = 70
NJ = 29
DEFAULT_XI_OFFSET = (0.75,) + (0.0,) * NJ + (0.0,) * 3
DEFAULT_XI_SCALE = (0.45,) + (0.01,) * NJ + (0.05,) * 3


@dataclass
class StateSpecCfg:
    dt: float = 0.02
    # Nominal error scales (D3) for [root pos m, root rotvec rad, root lin vel m/s, root ang vel rad/s, joint pos rad, joint vel rad/s].
    err_scale: tuple[float, ...] = (0.05, 0.1, 0.5, 1.0, 0.1, 2.0)
    # Output scales of the dynamics head for [dv_h, dw_b, dqd, dp_h, dth_b, dq].
    out_scale: tuple[float, ...] = (0.3, 0.5, 2.0, 0.01, 0.02, 0.02)
    # Input feature scales for [root_z, rot6d, lin_vel_h, ang_vel_b, joint_pos, joint_vel] (divide by these).
    in_scale: tuple[float, ...] = (1.0, 1.0, 1.0, 2.0, 1.0, 5.0)
    explicit_integration: bool = True
    # Training resolves the robot's layout; base preserves legacy standalone models.
    context_layout: ContextKind = "base"
    # Default vectors expand through the selected context layout.
    xi_offset: tuple[float, ...] = field(default_factory=lambda: DEFAULT_XI_OFFSET)
    xi_scale: tuple[float, ...] = field(default_factory=lambda: DEFAULT_XI_SCALE)


class StateSpec:
    def __init__(self, cfg: StateSpecCfg, device, dtype=torch.float32, num_joints: int = NJ):
        self.cfg = cfg
        self.dt = cfg.dt
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
        """Heading-invariant, scaled network input features (…, 71)."""
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        hq = heading_quat(q)
        q_local = quat_canonical(quat_mul(quat_apply_inverse_quat(hq), q))
        v_h = quat_apply_inverse(hq, v)
        feat = torch.cat([p[..., 2:3], rot6d_from_quat(q_local), v_h, w, qj, qd], dim=-1)
        return feat / self.in_scale

    def normalize_xi(self, xi: torch.Tensor) -> torch.Tensor:
        return (xi - self.xi_offset) / self.xi_scale

    # ------------------------------------------------------------------ integration

    def integrate(self, s: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Apply the (unscaled) network output ``delta`` (…, 70) to state ``s`` (…, 71)."""
        d = delta * self.out_scale
        nj = self.num_joints
        dv_h, dw_b, dqd, dp_h, dth_b, dq = (d[..., 0:3], d[..., 3:6], d[..., 6:6 + nj],
                                           d[..., 6 + nj:9 + nj], d[..., 9 + nj:12 + nj],
                                           d[..., 12 + nj:12 + 2 * nj])
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        hq = heading_quat(q)
        v_new = v + quat_apply(hq, dv_h)
        w_new = w + dw_b
        qd_new = qd + dqd
        dt = self.dt if self.cfg.explicit_integration else 0.0
        p_new = p + dt * v_new + quat_apply(hq, dp_h)
        q_new = quat_normalize(quat_mul(q, quat_from_rotvec(dt * w_new + dth_b)))
        qj_new = qj + dt * qd_new + dq
        return torch.cat([p_new, q_new, qj_new, v_new, w_new, qd_new], dim=-1)

    # ------------------------------------------------------------------ errors

    def error_vector(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """Normalised 70-d error [dp, rotvec, dv, dw, dq, dqd] / sigma."""
        p1, q1, qj1, v1, w1, qd1 = self.split(s_pred)
        p2, q2, qj2, v2, w2, qd2 = self.split(s_true)
        q1 = quat_normalize(q1)
        q2 = quat_normalize(q2)
        rot_err = rotvec_from_quat(quat_mul(quat_apply_inverse_quat(q2), q1))
        err = torch.cat([p1 - p2, rot_err, v1 - v2, w1 - w2, qj1 - qj2, qd1 - qd2], dim=-1)
        return err / self.err_scale

    def normalized_sq_error(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """Mean over dims of the squared normalised error (the reported 'prediction error')."""
        e = self.error_vector(s_pred, s_true)
        return (e * e).mean(-1)


def quat_apply_inverse_quat(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a unit quaternion (inverse)."""
    # A fresh CPU literal copied to CUDA is not legal inside graph capture.
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)
