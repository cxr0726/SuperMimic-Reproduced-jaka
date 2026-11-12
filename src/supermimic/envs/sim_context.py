"""Read applied simulator parameters into the shared, versioned xi layout."""

from __future__ import annotations

import numpy as np
import torch

from supermimic.sim_context import (
    KHAN_ANKLE_JOINTS,
    KHAN_FOOT_BODIES,
    KHAN_LOWER_BODY_JOINTS,
    ContextKind,
    SimContextLayout,
)

XI_DIM = 33


class SimContext:
    def __init__(self, env, foot_geom_regex: str, com_body_name: str = "torso_link",
                 entity_name: str = "robot", *, context_layout: ContextKind = "base"):
        self.env = env
        self.robot = env.scene[entity_name]
        self.num_joints = int(self.robot.data.encoder_bias.shape[-1])
        self.layout = SimContextLayout(self.num_joints, context_layout)
        self.xi_dim = self.layout.dim
        local_ids, _names = self.robot.find_geoms(foot_geom_regex)
        if len(local_ids) == 0:
            raise ValueError(f"no geoms match {foot_geom_regex}")
        self.foot_geom_global = int(self.robot.indexing.geom_ids[local_ids[0]].item())
        self.com_body_global = int(self.robot.indexing.body_ids[self.robot.body_names.index(com_body_name)].item())
        self.com_nominal = torch.as_tensor(
            np.array(env.sim.mj_model.body_ipos[self.com_body_global]), dtype=torch.float32, device=env.device
        )
        if context_layout in ("khan_dr_v1", "khan_dr_v2", "khan_dr_v3"):
            self._init_khan()

    def _init_khan(self) -> None:
        # BuiltinPositionActuator gain randomization modifies the MuJoCo model,
        # not actuator.cfg.stiffness/damping. Read the effective gains in joint
        # order, including when mjlab fuses individual control groups.
        from mjlab.actuator import BuiltinPositionActuator

        targets, controls = [], []
        buffers = {}
        for actuator in self.robot.actuators:
            if not isinstance(actuator, BuiltinPositionActuator):
                raise TypeError(f"{self.layout.kind} requires joint BuiltinPositionActuators")
            targets.append(torch.as_tensor(actuator.target_ids, device=self.env.device).reshape(-1))
            controls.append(actuator.global_ctrl_ids.reshape(-1))
            buffer = actuator._delay_buffer
            if buffer is not None:
                buffers[id(buffer)] = buffer
        target_ids, ctrl_ids = torch.cat(targets), torch.cat(controls)
        ordered, order = target_ids.sort()
        if not torch.equal(ordered, torch.arange(self.num_joints, device=self.env.device)):
            raise ValueError("Khan context needs exactly one position actuator per joint")
        self.ctrl_ids = ctrl_ids[order]
        if len(buffers) > 1:
            raise ValueError("One delay scalar requires a single fused Khan delay buffer")
        self.delay_buffer = next(iter(buffers.values()), None)
        self.kp_nominal = self.env.sim.get_default_field("actuator_gainprm")[self.ctrl_ids, 0]
        self.kd_nominal = -self.env.sim.get_default_field("actuator_biasprm")[self.ctrl_ids, 2]
        joints = (KHAN_ANKLE_JOINTS if self.layout.kind == "khan_dr_v1"
                  else KHAN_LOWER_BODY_JOINTS)
        armature_ids = [self.robot.joint_names.index(name) for name in joints]
        self.armature_dof_ids = self.robot.indexing.joint_v_adr.reshape(-1)[armature_ids]
        if self.layout.kind == "khan_dr_v1":
            self.ankle_dof_ids = self.armature_dof_ids  # Historical diagnostics.
        self.armature_nominal = self.env.sim.get_default_field("dof_armature")[self.armature_dof_ids]
        self.mass_nominal = self.env.sim.get_default_field("body_mass")[self.com_body_global]
        if self.layout.kind == "khan_dr_v3":
            foot_ids = [self.robot.body_names.index(name) for name in KHAN_FOOT_BODIES]
            self.foot_body_global_ids = self.robot.indexing.body_ids.reshape(-1)[foot_ids]
            self.foot_mass_nominal = self.env.sim.get_default_field("body_mass")[self.foot_body_global_ids]
        for value in (self.kp_nominal, self.kd_nominal, self.armature_nominal):
            if not torch.isfinite(value).all() or (value <= 0).any():
                raise ValueError("Khan gain/armature normalization requires positive nominal parameters")

    def _field(self, name: str, ids, nominal_ndim: int) -> torch.Tensor:
        value = getattr(self.env.sim.model, name)
        if value.ndim == nominal_ndim + 1:
            return value[:, ids]
        return value[ids].unsqueeze(0).expand(self.env.num_envs, *value[ids].shape)

    def read(self) -> torch.Tensor:
        model = self.env.sim.model
        n = self.env.num_envs
        fr = model.geom_friction
        if fr.dim() == 3:  # expanded per world
            friction = fr[:, self.foot_geom_global, 0]
        else:
            friction = fr[self.foot_geom_global, 0].expand(n)
        ipos = model.body_ipos
        if ipos.dim() == 3:
            com = ipos[:, self.com_body_global] - self.com_nominal
        else:
            com = (ipos[self.com_body_global] - self.com_nominal).expand(n, 3)
        bias = self.robot.data.encoder_bias
        values = [friction.reshape(n, 1), bias, com]
        if self.layout.kind in ("khan_dr_v1", "khan_dr_v2", "khan_dr_v3"):
            kp = self._field("actuator_gainprm", self.ctrl_ids, 2)[..., 0] / self.kp_nominal
            kd = -self._field("actuator_biasprm", self.ctrl_ids, 2)[..., 2] / self.kd_nominal
            armature = self._field("dof_armature", self.armature_dof_ids, 1) / self.armature_nominal
            mass = self._field("body_mass", self.com_body_global, 1).reshape(n, 1) - self.mass_nominal
            # Official explicit/auto reset writes controls (and draws the lag)
            # before record_post_reset. Read it; never sample a second value.
            lag = (self.delay_buffer.current_lags.reshape(n, 1).float()
                   if self.delay_buffer is not None else torch.zeros(n, 1, device=self.env.device))
            values.extend((kp, kd, armature, mass, lag))
        if self.layout.kind == "khan_dr_v3":
            # Added point payload at the authored COM: read kg, not a mass factor.
            values.append(self._field("body_mass", self.foot_body_global_ids, 1) - self.foot_mass_nominal)
        xi = torch.cat(values, dim=-1)
        assert xi.shape == (n, self.xi_dim)
        return xi.clone()
