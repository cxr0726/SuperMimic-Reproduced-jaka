"""Rotation helpers (torch, batched, quaternions in **wxyz**).

Uses the unchanged Python bodies of mjlab's ``lab_api.math`` primitives so the
project shares its rotation conventions without a TorchScript profiling
executor inside checkpoint recomputation/CUDA Graph capture. Global mjlab/JIT
settings remain untouched. Adds rotation-vector exp/log, 6D representation and
heading (yaw) frames.

All functions are differentiable and work on arbitrary leading batch dims.
"""

from __future__ import annotations

import math
from types import FunctionType

import torch
from mjlab.utils.lab_api import math as _mjlab_math


def _python_math_body(function, cache):
    """Clone a primitive and its scripted dependencies into private globals.

    TorchDynamo already uses ``_torchdynamo_inline`` to access these same
    bodies. Eager autograd needs them too: TorchScript can change its saved
    tensor plan between a checkpoint's forward and recomputation, or start
    profiling an optimized plan during an outer CUDA Graph capture.
    """
    if id(function) in cache:
        return cache[id(function)]
    body = function if isinstance(function, FunctionType) else getattr(function, "_torchdynamo_inline", None)
    if not isinstance(body, FunctionType):
        raise RuntimeError(  # noqa: TRY004 -- dependency compatibility, not a caller's argument type
            "mjlab rotation primitives must expose their Python body for checkpoint/CUDA Graph safety; "
            "the installed TorchScript/mjlab API is incompatible"
        )
    namespace = dict(body.__globals__)
    result = FunctionType(body.__code__, namespace, body.__name__, body.__defaults__, body.__closure__)
    cache[id(function)] = result  # Register before following dependencies/cycles.
    result.__qualname__, result.__doc__ = body.__qualname__, body.__doc__
    result.__annotations__ = dict(body.__annotations__)
    result.__kwdefaults__ = None if body.__kwdefaults__ is None else dict(body.__kwdefaults__)
    for name in body.__code__.co_names:
        dependency = namespace.get(name)
        if (isinstance(dependency, torch.jit.ScriptFunction)
                or isinstance(dependency, FunctionType) and dependency.__globals__ is body.__globals__):
            namespace[name] = _python_math_body(dependency, cache)
    return result


_PRIMITIVE_NAMES = (
    "matrix_from_quat", "quat_apply", "quat_apply_inverse", "quat_conjugate",
    "quat_error_magnitude", "quat_from_angle_axis", "quat_inv", "quat_mul", "yaw_quat", "quat_box_minus",
)
_primitive_cache = {}
(matrix_from_quat, quat_apply, quat_apply_inverse, quat_conjugate, quat_error_magnitude,
 quat_from_angle_axis, quat_inv, quat_mul, yaw_quat, quat_box_minus) = (
    _python_math_body(getattr(_mjlab_math, name), _primitive_cache) for name in _PRIMITIVE_NAMES
)

_EPS = 1.0e-8


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(_EPS)


def quat_canonical(q: torch.Tensor) -> torch.Tensor:
    """Flip sign so that w >= 0 (removes the double cover ambiguity)."""
    return torch.where(q[..., :1] < 0, -q, q)


def quat_from_rotvec(v: torch.Tensor) -> torch.Tensor:
    """Exponential map: rotation vector (angle * axis) -> unit quaternion (wxyz).

    Uses a Taylor expansion of sin(a/2)/a near zero so gradients stay finite at v = 0.
    """
    angle_sq = (v * v).sum(-1, keepdim=True)
    angle = torch.sqrt(angle_sq.clamp_min(1e-30))
    half = 0.5 * angle
    small = angle < 1e-4
    # sin(half)/angle with series fallback: 1/2 - angle^2/48
    k = torch.where(small, 0.5 - angle_sq / 48.0, torch.sin(half) / angle)
    w = torch.where(small, 1.0 - angle_sq / 8.0, torch.cos(half))
    return torch.cat([w, v * k], dim=-1)


def rotvec_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Logarithmic map: unit quaternion (wxyz) -> rotation vector in (-pi, pi]."""
    q = quat_canonical(q)
    w = q[..., :1]
    xyz = q[..., 1:]
    s = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(s, w)
    small = s < 1e-6
    # angle / sin(angle/2) == angle / s ; series: 2 + angle^2/12 ... (with s ~ angle/2)
    k = torch.where(small, 2.0 + (angle * angle) / 12.0, angle / torch.where(small, torch.ones_like(s), s))
    return xyz * k


def quat_integrate(q: torch.Tensor, dtheta_body: torch.Tensor) -> torch.Tensor:
    """Apply a body-frame rotation increment: q ⊗ exp(dtheta)."""
    return quat_normalize(quat_mul(q, quat_from_rotvec(dtheta_body)))


def rot6d_from_quat(q: torch.Tensor) -> torch.Tensor:
    """First two columns of the rotation matrix, flattened (..., 6)."""
    m = matrix_from_quat(q)
    return m[..., :, :2].reshape(*q.shape[:-1], 6)


def heading_yaw(q: torch.Tensor) -> torch.Tensor:
    """Yaw angle of the heading frame of q (rotation of the body x-axis about world z)."""
    w, x, y, z = q.unbind(-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw
    zeros = torch.zeros_like(half)
    return torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)


def heading_quat(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion of the pure-yaw heading frame (same as mjlab yaw_quat, but via atan2)."""
    return quat_from_yaw(heading_yaw(q))


def wrap_to_pi(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def projected_gravity(q_body_w: torch.Tensor) -> torch.Tensor:
    """Gravity direction (0,0,-1) expressed in the body frame."""
    # Device-native construction also works inside a complete learner graph.
    g = q_body_w.new_zeros((*q_body_w.shape[:-1], 3))
    g[..., 2] = -1.0
    return quat_apply_inverse(q_body_w, g)
