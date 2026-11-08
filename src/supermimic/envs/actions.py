"""Joint-position action with execution noise (plan D9).

The *clean* policy action stays in ``raw_action`` (so ``last_action`` observations and
the action-rate penalty see it); Gaussian noise ``eps ~ N(0, std^2)`` is added to the
normalized action only when computing the requested set-point. ``executed_raw`` exposes
``a + eps`` for replay. When actuator delay is enabled, this is the current command
before the physics-step target queue, not the historical target actually applied
on every substep. The world model intentionally fits that delay implicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.buffers import DelayBuffer


class NoisyJointPositionAction(JointPositionAction):
    cfg: NoisyJointPositionActionCfg

    def __init__(self, cfg: NoisyJointPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._executed_raw = torch.zeros_like(self._raw_actions)
        self.noise_std = float(cfg.noise_std)
        self._delay_reset_groups = (
            self._resolve_delay_reset_groups() if cfg.delay_reset_first_step_no_delay else []
        )
        self._delay_reset_env_ids: list[torch.Tensor] = []
        self._delay_reset_ready = False
        self._all_env_ids = torch.arange(self._raw_actions.shape[0], device=self._raw_actions.device)

    def _resolve_delay_reset_groups(self) -> list[tuple[DelayBuffer, torch.Tensor, tuple[str, ...]]]:
        """Resolve each physical queue once, in its fused actuator target order.

        mjlab aliases a fused buffer into all its actuators. Iterating actuator
        buffers directly would therefore backfill a shared queue with a single
        joint's command. Builtin position queues contain just position targets;
        explicit PD/custom queues contain position, velocity and effort targets.
        """
        groups = []
        seen = set()
        owned = set(self._target_ids.cpu().tolist())

        def add(buffer, target_ids, attrs):
            if buffer is None or id(buffer) in seen:
                return
            targets = set(target_ids.cpu().tolist())
            if not targets.intersection(owned):
                return
            if not targets.issubset(owned):
                raise ValueError(
                    "delay_reset_first_step_no_delay requires this action to own every joint "
                    "in its fused delay group; split-action queue priming is unsupported"
                )
            groups.append((buffer, target_ids, attrs))
            seen.add(id(buffer))

        entity = self._entity
        for group in entity._builtin_group._delayed_groups:
            if group.target_attr == "joint_pos_target":
                add(group.delay_buffer, group.target_ids, (group.target_attr,))
        joint_targets = ("joint_pos_target", "joint_vel_target", "joint_effort_target")
        for group in entity._fused_actuator_group._groups:
            if group.transmission_type == TransmissionType.JOINT:
                add(group.delay_buffer, group.target_ids, joint_targets)
        for actuator in entity._custom_actuators:
            if actuator.cfg.transmission_type == TransmissionType.JOINT:
                add(actuator._delay_buffer, actuator.target_ids, joint_targets)
        return groups

    @property
    def executed_raw(self) -> torch.Tensor:
        return self._executed_raw

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        if self.noise_std > 0:
            self._executed_raw = self._raw_actions + self.noise_std * torch.randn_like(self._raw_actions)
        else:
            self._executed_raw = self._raw_actions.clone()
        self._processed_actions = self._executed_raw * self._scale + self._offset
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        self._delay_reset_ready = bool(self._delay_reset_env_ids)

    def apply_actions(self) -> None:
        super().apply_actions()
        if not self._delay_reset_ready:
            return
        # The entity now holds the exact physical command, including encoder
        # bias, after action scaling, execution noise and optional clipping.
        # Backfill before the normal physics-cadence append. This starts with a0
        # immediately for the entire first control step; a1 is delayed normally.
        # No extra append/compute call: other worlds, lag draws and clocks retain
        # their normal cadence. Raw action history and first action-rate stay 0→a0.
        for buffer, target_ids, attrs in self._delay_reset_groups:
            if not buffer.is_initialized:
                # mjlab's first append initializes every history slot itself.
                continue
            fields = [getattr(self._entity.data, attr)[:, target_ids] for attr in attrs]
            target = fields[0] if len(fields) == 1 else torch.stack(fields, dim=-1)
            for env_ids in self._delay_reset_env_ids:
                buffer.backfill(target, env_ids)
        self._delay_reset_env_ids.clear()
        self._delay_reset_ready = False

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._executed_raw[env_ids] = 0.0
        if self._delay_reset_groups:
            # Cache IDs already produced by reset; do not nonzero a CUDA mask or
            # synchronize to CPU during ordinary action processing/substeps.
            if env_ids is None:
                self._delay_reset_env_ids = [self._all_env_ids]
            else:
                self._delay_reset_env_ids.append(self._all_env_ids[env_ids])
            # An apply without a new policy action must not prime stale targets.
            self._delay_reset_ready = False


@dataclass(kw_only=True)
class NoisyJointPositionActionCfg(JointPositionActionCfg):
    noise_std: float = 0.1
    delay_reset_first_step_no_delay: bool = False
    """Prime reset rows with the first physical policy target; subsequent steps use normal delay."""

    def build(self, env: ManagerBasedRlEnv) -> NoisyJointPositionAction:
        return NoisyJointPositionAction(self, env)
