"""Record deterministic policy rollouts (robot qpos + reference qpos per control step) for rendering.

  uv run python scripts/media/record_rollout.py --ckpt logs/e8/ckpt_0100000.pt --npz-dir data/npz/bones_seed \
      --clips cartwheelin_003__A486_M jump_ff_360_R_004__A407 --out media/rollouts.npz

Same protocol as the Appendix F benchmark (scripts/eval_benchmark.py): every clip starts at its first frame,
no observation/action noise, no reset noise, no domain randomization. Each clip is played once; recording
stops at the clip end or at the termination. Render the result with scripts/media/render_rollout.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import tyro

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.state_io import StateIO
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.robots.registry import get_robot_spec


@torch.no_grad()
def main(ckpt: Path, clips: list[str], npz_dir: Path | None = None, out: Path = Path("media/rollouts.npz"), device: str = "cuda:0") -> None:
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    robot_spec = get_robot_spec(cfg.get("robot", "g1"))
    npz_dir = npz_dir or robot_spec.default_npz_dir
    files = [str(f) for f in discover_npz(npz_dir, names=clips)]
    missing = set(clips) - {Path(f).stem for f in files}
    assert not missing, f"clips not found in {npz_dir}: {sorted(missing)}"
    n = len(files)
    args = SuperMimicEnvCfgArgs(
        npz_files=files, num_envs=n, future_frames=cfg["env"]["future_frames"], episode_length_s=1.0e5,
        robot=robot_spec.name, history_length=policy.cfg.history,
        action_noise_std=0.0, obs_noise=False, reset_noise=False, domain_randomization=False, sampling_mode="start",
        record_transitions=False, seed=0, anchor_obs=cfg["env"].get("anchor_obs", False),
    )
    env = make_env(args, device=device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        io = StateIO(env)
        env_clip = torch.arange(n, device=device)
        cmd.forced_frame[:] = lib.clip_start[env_clip]
        obs, _ = env.reset()
        cmd.forced_frame[:] = lib.clip_start[env_clip]  # keep the clip if an env is reset again
        max_r = env.step_dt * sum(max(0.0, term.weight) for term in env.cfg.rewards.values())
        rob, ref = [], []
        alive = torch.ones(n, dtype=torch.bool, device=device)
        rew_sum = torch.zeros(n, device=device)
        steps = torch.zeros(n, dtype=torch.long, device=device)

        def snapshot() -> None:
            rob.append(io.read()[:, :io.nq].cpu().numpy())
            ref.append(lib.ref_qpos_qvel(cmd.frame)[0].cpu().numpy())

        snapshot()
        while alive.any():
            act = policy(obs["policy"], obs["goal"])
            obs, rew, term, tout, _ = env.step(act)
            rew_sum += rew * alive
            steps += alive.long()
            done = (term | tout) & alive
            alive &= ~done
            if alive.any():
                snapshot()
            if steps.max() > 20000:
                break
        names = [lib.clips[i].name for i in range(n)]
        lengths = steps.cpu().numpy()  # recorded frames = steps (the terminal/reset state is not recorded)
        T = len(rob)
        robot_qpos, ref_qpos = np.stack(rob, 1), np.stack(ref, 1)
        for i in range(n):  # freeze frames after a clip's end so every row has the same length
            robot_qpos[i, lengths[i]:] = robot_qpos[i, lengths[i] - 1]
            ref_qpos[i, lengths[i]:] = ref_qpos[i, lengths[i] - 1]
        out.parent.mkdir(parents=True, exist_ok=True)
        norm_return = (rew_sum / (max_r * steps.clamp(min=1))).cpu().numpy()
        np.savez_compressed(out, names=np.array(names), robot_qpos=robot_qpos[:, :T], ref_qpos=ref_qpos[:, :T], lengths=lengths, norm_return=norm_return,
                            robot=np.array(robot_spec.name), fps=np.array(1.0 / env.step_dt))
        for i, nm in enumerate(names):
            print(f"{nm:48s} steps {int(lengths[i]):5d}/{lib.clips[i].length}  return {float(norm_return[i]):.3f}")
        print("saved", out)
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)
