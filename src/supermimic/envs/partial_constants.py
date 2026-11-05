"""Reset-only Khan armature refresh using the stock Warp math on selected worlds.

For partial resets, a compact, isolated workspace runs set_const_0; only selected
model rows return to the live simulation, without touching live Data. Full resets
retain the stock path. Normal mjlab reset/step performs forward afterwards.
"""

from __future__ import annotations

import gc
from dataclasses import fields, is_dataclass, replace
from math import prod
from typing import Any

import mujoco_warp as mjwarp
import torch
import warp as wp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import requires_model_fields

# All Model outputs of the pinned backend's set_const_0, including outputs not
# listed by mjlab's requires_model_fields decorator. Empty Khan fields are kept
# here so they cannot accidentally alias a writable live buffer.
_OUTPUTS = frozenset({
    "dof_invweight0", "body_invweight0", "tendon_length0", "tendon_invweight0",
    "actuator_acc0", "actuator_biasprm", "eq_data", "cam_pos0", "cam_poscom0",
    "cam_mat0", "light_pos0", "light_poscom0", "light_dir0", "stat.meaninertia",
})
_DERIVED = (
    "dof_invweight0", "body_invweight0", "tendon_length0",
    "tendon_invweight0", "actuator_acc0",
)


@wp.kernel
def _gather_worlds(src: wp.array(dtype=Any), dst: wp.array(dtype=Any),
                   ids: wp.array(dtype=int), width: int, shared: int):
    world, item = wp.tid()
    source_world = ids[world]
    if shared != 0:
        source_world = 0
    dst[world * width + item] = src[source_world * width + item]


@wp.kernel
def _scatter_worlds(src: wp.array(dtype=Any), dst: wp.array(dtype=Any),
                    ids: wp.array(dtype=int), count: wp.array(dtype=int), width: int):
    world, item = wp.tid()
    if world < count[0]:
        dst[ids[world] * width + item] = src[world * width + item]


@wp.kernel
def _scatter_shared_world0(src: wp.array(dtype=Any), dst: wp.array(dtype=Any),
                           ids: wp.array(dtype=int), count: wp.array(dtype=int), width: int):
    world, item = wp.tid()
    # Stock full-world set_const_0 uses original world 0 when meaninertia is
    # shared. Never substitute the first compacted world or a padded slot.
    if world < count[0] and ids[world] == 0:
        dst[item] = src[world * width + item]


class _Bucket:
    def __init__(self, sim, capacity: int):
        self.capacity = capacity
        self.ids = wp.zeros(capacity, dtype=int, device=sim.wp_device)
        self.count = wp.zeros(1, dtype=int, device=sim.wp_device)
        self.gathers = []
        self.scatters = []
        self.shared_scatters = []
        nworld = sim.num_envs

        def compact(obj, prefix=""):
            result = replace(obj)
            for field in fields(obj):
                source = getattr(obj, field.name)
                name = f"{prefix}.{field.name}" if prefix else field.name
                if is_dataclass(source):
                    setattr(result, field.name, compact(source, name))
                    continue
                shape = getattr(field.type, "shape", ())
                if not (isinstance(source, wp.array) and shape
                        and shape[0] in ("*", "nworld")):
                    continue
                if source.shape[0] not in (1, nworld):
                    raise ValueError(f"Unsupported world layout for {name}: {source.shape}")
                if source.shape[0] == 1 and name not in _OUTPUTS:
                    continue  # Shared, read-only parameter/topology data.
                target = wp.empty((capacity, *source.shape[1:]), dtype=source.dtype,
                                  device=sim.wp_device)
                setattr(result, field.name, target)
                width = prod(source.shape[1:])
                if source.size == 0:
                    continue
                self.gathers.append((source.flatten(), target.flatten(), width,
                                     int(source.shape[0] == 1)))
                if name in _OUTPUTS:
                    if source.shape[0] == 1:
                        self.shared_scatters.append((target.flatten(), source.flatten(), width))
                    elif source.shape[0] == nworld:
                        self.scatters.append((target.flatten(), source.flatten(), width))
                    else:
                        raise ValueError(f"Writable field {name} must be expanded per environment")
            return result

        with wp.ScopedDevice(sim.wp_device):
            self.model = compact(sim.wp_model)
            # This helper is deliberately limited to the rigid Khan reset path.
            # set_const_0 needs only reference geometry, not live velocities,
            # contacts, solver warmstart, time, or actuator history.
            self.data = mjwarp.make_data(sim.mj_model, nworld=capacity,
                                        nconmax=0, nccdmax=0, njmax=0)
            self.graph = None
            self._run()  # Compile all kernels before capture; count=0 writes no live rows.
            if sim.use_cuda_graph:
                enabled = gc.isenabled()
                gc.disable()
                try:
                    with wp.ScopedCapture() as capture:
                        self._run()
                    self.graph = capture.graph
                finally:
                    if enabled:
                        gc.enable()

    def _run(self):
        for source, target, width, shared in self.gathers:
            wp.launch(_gather_worlds, dim=(self.capacity, width),
                      inputs=[source, target, self.ids, width, shared])
        mjwarp.set_const_0(self.model, self.data, restore=False)
        for source, target, width in self.scatters:
            wp.launch(_scatter_worlds, dim=(self.capacity, width),
                      inputs=[source, target, self.ids, self.count, width])
        for source, target, width in self.shared_scatters:
            wp.launch(_scatter_shared_world0, dim=(self.capacity, width),
                      inputs=[source, target, self.ids, self.count, width])

    def refresh(self, env_ids):
        n = env_ids.numel()
        # Padding repeats a selected world, never an unrelated running world;
        # count prevents those duplicate slots from scattering anything back.
        target_ids = wp.to_torch(self.ids)
        target_ids.copy_(env_ids[:1].expand(self.capacity))
        target_ids[:n].copy_(env_ids)
        wp.to_torch(self.count).fill_(n)
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._run()


class KhanArmatureConstants:
    """Preserve full-refresh model semantics; partial resets leave Data untouched.

    Call only from a reset event followed by mjlab's normal sim.forward(). The
    buckets cap padding at <2x the selected count and cache a CUDA graph, avoiding
    repeated Python dispatch of the backend's hundreds of matrix solves.
    """

    def __init__(self, sim):
        model = sim.mj_model
        if (model.nv != 33 or model.nu != 27 or
                any(getattr(model, name) for name in
                    ("ntendon", "neq", "nflex", "nmocap"))):
            raise ValueError("Partial armature refresh supports the rigid 27-joint Khan model only")
        self.sim = sim
        self.buckets: dict[int, _Bucket] = {}
        self.last_capacity = 0

    def refresh(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.sim.num_envs, device=self.sim.device, dtype=torch.int)
        if env_ids.ndim != 1:
            raise ValueError("env_ids must be a vector of reset environments")
        n = env_ids.numel()
        if n == 0:
            self.last_capacity = 0
            return
        if n == self.sim.num_envs:
            # All worlds really reset: retain the original path, without keeping
            # an otherwise unused 4096-world scratch allocation/graph resident.
            with wp.ScopedDevice(self.sim.wp_device):
                mjwarp.set_const_0(self.sim.wp_model, self.sim.wp_data)
            self.last_capacity = n
            return
        capacity = 1 << (n - 1).bit_length()
        with wp.ScopedDevice(self.sim.wp_device):
            if capacity not in self.buckets:
                self.buckets[capacity] = _Bucket(self.sim, capacity)
            self.buckets[capacity].refresh(env_ids)
        self.last_capacity = capacity


@requires_model_fields("dof_armature", "actuator_biasprm", *_DERIVED)
def khan_joint_armature(env, env_ids, **kwargs):
    """Same armature samples as mjlab; refresh only the affected model rows."""
    if env_ids is not None and env_ids.numel() == 0:
        return
    refresher = getattr(env, "_khan_armature_constants", None)
    if refresher is None:
        refresher = env._khan_armature_constants = KhanArmatureConstants(env.sim)
    dr.joint_armature(env, env_ids, **kwargs)
    refresher.refresh(env_ids)
