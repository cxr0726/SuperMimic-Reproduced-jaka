"""Episode-held Khan ankle encoder offsets, shared by sensing and physical PD."""

import torch
from mjlab.envs.mdp import dr
from mjlab.managers.scene_entity_config import SceneEntityCfg


def khan_encoder_bias(env, env_ids, ankle_half_range: float, asset_cfg: SceneEntityCfg) -> None:
    """Keep the original full-joint draw, then rescale only the four ankle axes.

    ``asset_cfg`` resolves the ankle IDs in the entity's actual joint order.
    Preserving the draw also preserves other joints and the subsequent RNG stream.
    The entity bias is already used by both joint_pos_biased and the PD target.
    """
    dr.encoder_bias(env, env_ids, (-0.01, 0.01), SceneEntityCfg(asset_cfg.name))
    bias = env.scene[asset_cfg.name].data.encoder_bias
    factor = ankle_half_range / 0.01
    if env_ids is None:
        bias[:, asset_cfg.joint_ids] *= factor
    else:
        rows = env_ids.to(device=bias.device, dtype=torch.long)
        bias[rows[:, None], asset_cfg.joint_ids] *= factor
