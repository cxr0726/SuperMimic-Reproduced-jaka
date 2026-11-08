"""SuperMimic training loop (paper Algorithm A.1): collect -> fit world model -> improve policy."""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from supermimic.config import TrainConfig, resolve_training_observations
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.mdp.reset_sampling import reset_diagnostic_metrics
from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.observation_layout import goal_dimension
from supermimic.policy.nets import Critic, Policy
from supermimic.robots.contracts import robot_contract, validate_robot_contract
from supermimic.robots.registry import get_robot_spec
from supermimic.training.distributed import (
    allreduce_mean,
    broadcast_params,
    finalize,
    init_distributed,
)
from supermimic.training.schedule import cosine_lr
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.wm_trainer import WorldModelTrainer
from supermimic.world_model.world_model import WorldModel


def load_sampler_state(command, checkpoint: dict) -> bool:
    """Restore comparable curriculum statistics, or reset pre-v2 estimates.

    Legacy checkpoints estimated returns for every visited bin without normal
    clip-end continuation. Those estimates must not seed the new start-bin
    curriculum. Model/optimizer state and schedule restoration are independent.
    """
    if "sampler_contract" not in checkpoint:
        command.reset_sampler_statistics()
        warnings.warn(
            "Legacy checkpoint has no sampler_contract; reset sampler statistics "
            "are reinitialized for the new algorithm (bin returns zero, history "
            "uniform). Model/optimizer states and the training iteration are retained.",
            UserWarning,
            stacklevel=2,
        )
        return False

    saved_contract = checkpoint["sampler_contract"]
    if not isinstance(saved_contract, dict) or saved_contract != command.sampler_contract:
        raise ValueError(
            "Checkpoint sampler_contract differs from the current reset sampler "
            "algorithm, coefficients, or motion-bin mapping; use matching settings "
            "and motion data, or start a new run"
        )

    # Validate both before copying either, so a malformed checkpoint cannot leave
    # the command with a partially restored curriculum.
    statistics = {}
    for name in ("bin_return", "bin_memory"):
        value = checkpoint.get(name)
        target = getattr(command, name)
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise ValueError(f"Checkpoint sampler {name} must be a floating-point tensor")
        if value.shape != target.shape:
            raise ValueError(
                f"Checkpoint sampler {name} shape {tuple(value.shape)} differs from "
                f"the current motion bins {tuple(target.shape)}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"Checkpoint sampler {name} contains nonfinite values")
        converted = value.to(device=target.device, dtype=target.dtype)
        if not torch.isfinite(converted).all():
            raise ValueError(f"Checkpoint sampler {name} becomes nonfinite in the runtime dtype")
        statistics[name] = converted
    memory = statistics["bin_memory"]
    if (memory < 0).any() or not torch.isclose(
        memory.double().sum(), memory.new_tensor(1.0, dtype=torch.float64),
        rtol=1e-5, atol=1e-5,
    ):
        raise ValueError("Checkpoint sampler bin_memory must be a nonnegative probability distribution")
    sampleable = getattr(command, "sampleable_bins", None)
    if sampleable is not None and (memory[~sampleable.to(memory.device)] > 0).any():
        raise ValueError("Checkpoint sampler bin_memory assigns probability to bins with no valid reset frame")
    with torch.no_grad():
        command.bin_return.copy_(statistics["bin_return"])
        command.bin_memory.copy_(memory)
    return True


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.robot_spec = robot_spec = get_robot_spec(cfg.robot)
        if cfg.data.npz_dir is None:
            cfg.data.npz_dir = robot_spec.default_npz_dir
        resolve_training_observations(cfg)
        if cfg.data.library is not None:
            cfg.env.compact_library = True
        self.rank, local_rank, self.world = init_distributed()
        self.is_main = self.rank == 0
        if self.world > 1:
            cfg.device = f"cuda:{local_rank}"
            # Paper: lr x sqrt(2) for 2 GPUs (each GPU keeps its own 4096 envs and buffer).
            cfg.wm_trainer.lr *= self.world**0.5
            cfg.policy_trainer.lr *= self.world**0.5
            cfg.policy_trainer.critic_lr *= self.world**0.5
        torch.manual_seed(cfg.seed + 1000 * self.rank)
        self.device = cfg.device
        self.log_dir = Path(cfg.log_root) / cfg.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.is_main:
            (self.log_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
        self.writer = SummaryWriter(str(self.log_dir / "tb")) if self.is_main else None

        if cfg.data.library is not None:
            files = [cfg.data.library]
        else:
            files = discover_npz(cfg.data.npz_dir, limit=cfg.data.limit, names=cfg.data.clips, manifest=cfg.data.manifest)
        if not files:
            raise FileNotFoundError(f"no clips under {cfg.data.npz_dir}")
        self.env = make_env(
            SuperMimicEnvCfgArgs(
                npz_files=[str(f) for f in files],
                robot=cfg.robot,
                num_envs=cfg.env.num_envs,
                future_frames=cfg.env.future_frames,
                history_length=cfg.policy.history,
                observation_layout=cfg.env.observation_layout,
                episode_length_s=cfg.env.episode_length_s,
                action_noise_std=cfg.env.action_noise_std,
                obs_noise=cfg.env.obs_noise,
                reset_noise=cfg.env.reset_noise,
                domain_randomization=cfg.env.domain_randomization,
                actuator_delay=cfg.env.actuator_delay,
                actuator_delay_max_ms=cfg.env.actuator_delay_max_ms,
                reset_root_z_offset=cfg.env.reset_root_z_offset,
                sampling_mode=cfg.env.sampling_mode,
                replay_capacity=cfg.env.replay_capacity,
                compact_library=cfg.env.compact_library,
                reference_fk_compile=cfg.env.reference_fk_compile,
                merge_short_tail_bins=cfg.env.merge_short_tail_bins,
                anchor_obs=cfg.env.anchor_obs,
                seed=cfg.seed + 1000 * self.rank,
            ),
            device=self.device,
        )
        env = self.env
        self.command = env.command_manager.get_term("motion")
        self.lib = self.command.library
        self.recorder = env.recorder_manager.get_term("replay")
        self.buffer = self.recorder.buffer
        robot = env.scene["robot"]

        fk = G1Kinematics(robot_spec.load_model(), device=self.device)
        heads = AnalyticHeads(fk, robot_spec.tracked_bodies, robot_spec.anchor_body, robot_spec.root_body, robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone())
        self.wm = WorldModel(cfg.world_model, heads, self.device)
        self.policy = Policy(cfg.policy).to(self.device)
        if self.buffer.xi.shape[-1] != self.wm.spec.xi_dim:
            raise ValueError("Simulator replay and world model context layouts differ")
        self.critic = Critic(cfg.critic, pcfg=cfg.policy, xi_dim=self.wm.spec.xi_dim).to(self.device)
        for m in (self.wm, self.policy, self.critic):
            broadcast_params(m)
        self.wm_trainer = WorldModelTrainer(self.wm, self.lib, cfg.wm_trainer)
        self.policy_trainer = PolicyTrainer(self.wm, self.policy, self.critic, self.lib, cfg.policy_trainer)
        n_params = lambda m: sum(p.numel() for p in m.parameters())
        if self.is_main:
            print(f"[INFO] params: world model {n_params(self.wm)/1e6:.2f}M, policy {n_params(self.policy)/1e6:.2f}M, critic {n_params(self.critic)/1e6:.3f}M")
            print(f"[INFO] sim context: {self.wm.spec.context.kind}, xi={self.wm.spec.xi_dim}, "
                  f"fields={self.wm.spec.context.sizes}; delay queue remains implicit")

        # Episode statistics of the training env (paper's "training return").
        self.ep_reward_sum = torch.zeros(env.num_envs, device=self.device)
        self.ep_len = torch.zeros(env.num_envs, device=self.device)
        self.max_step_reward = heads.reward_cfg.max_step_reward
        self.iteration = 0
        self.start_iteration = 0  # warm-up (buffer refill) counts from here, also after a resume
        self.num_updates = 0
        self._utd_acc = 0.0
        if cfg.resume is not None:
            self.load(cfg.resume)

    # ------------------------------------------------------------------ checkpoints

    def save(self, name: str = "ckpt_latest.pt") -> None:
        if not self.is_main:
            return
        torch.save(
            {
                "iteration": self.iteration,
                "num_updates": self.num_updates,
                "wm": self.wm.state_dict(),
                "policy": self.policy.state_dict(),
                "critic": self.critic.state_dict(),
                "wm_opt": self.wm_trainer.opt.state_dict(),
                "policy_opt": self.policy_trainer.opt.state_dict(),
                "critic_opt": self.policy_trainer.critic_opt.state_dict(),
                "bin_return": self.command.bin_return,
                "bin_memory": self.command.bin_memory,
                "sampler_contract": self.command.sampler_contract,
                "config": asdict(self.cfg),
                "robot_contract": robot_contract(self.cfg.robot, self.cfg.env.observation_layout,
                                                 env_config=asdict(self.cfg.env)),
            },
            self.log_dir / name,
        )

    def load(self, path: Path) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        saved_robot = ck["config"].get("robot", "g1")
        if saved_robot != self.cfg.robot:
            raise ValueError(f"Checkpoint robot {saved_robot!r} differs from {self.cfg.robot!r}")
        saved_anchor = ck["config"].get("env", {}).get("anchor_obs", False)
        if saved_anchor != self.cfg.env.anchor_obs:
            raise ValueError(
                f"Checkpoint anchor_obs={saved_anchor} differs from requested "
                f"anchor_obs={self.cfg.env.anchor_obs}; use matching settings or start a new run"
            )
        saved_env = ck["config"].get("env", {})
        saved_policy = ck["config"].get("policy", {})
        saved_profile = saved_env.get("observation_layout", "supermimic")
        saved_future = saved_env.get("future_frames", 10)
        saved_layout = (saved_profile, saved_future, saved_policy.get("history", 10),
                        saved_policy.get("goal_dim", goal_dimension(
                            saved_profile, self.robot_spec.num_joints, saved_future, saved_anchor)))
        current_layout = (self.cfg.env.observation_layout, self.cfg.env.future_frames,
                          self.cfg.policy.history, self.cfg.policy.goal_dim)
        if saved_layout != current_layout:
            raise ValueError(
                f"Checkpoint goal/observation layout differs: saved {saved_layout}, requested {current_layout}; "
                "use matching settings or start a new run"
            )
        validate_robot_contract(ck, self.cfg.robot, self.cfg.env.observation_layout)
        if self.cfg.robot == "khan_mini" and ck["robot_contract"] != robot_contract(
            self.cfg.robot, self.cfg.env.observation_layout, env_config=asdict(self.cfg.env),
        ):
            raise ValueError("Checkpoint environment semantics differ (delay/reset/DR); start a new run or use matching settings")
        self.wm.load_state_dict(ck["wm"])
        self.policy.load_state_dict(ck["policy"])
        self.critic.load_state_dict(ck["critic"])
        self.wm_trainer.opt.load_state_dict(ck["wm_opt"])
        self.policy_trainer.opt.load_state_dict(ck["policy_opt"])
        self.policy_trainer.critic_opt.load_state_dict(ck["critic_opt"])
        load_sampler_state(self.command, ck)
        self.iteration = ck["iteration"]
        self.start_iteration = self.iteration
        self.num_updates = ck["num_updates"]
        print(f"[INFO] resumed from {path} at iteration {self.iteration}")

    # ------------------------------------------------------------------ main loop

    @torch.no_grad()
    def act(self, obs) -> torch.Tensor:
        return self.policy(obs["policy"], obs["goal"])

    def run(self) -> None:
        cfg = self.cfg
        sch = cfg.schedule
        env = self.env
        obs, _ = env.reset()
        t_start = time.time()
        timers = {"sim": 0.0, "wm": 0.0, "policy": 0.0}
        agg: dict[str, float] = {}
        agg_n = 0
        finished_returns: list[float] = []
        finished_lens: list[float] = []
        last_log = time.time()
        stop_iteration = sch.total_iterations
        if sch.stop_after_iterations is not None:
            if sch.stop_after_iterations < 1:
                raise ValueError("stop_after_iterations must be positive")
            stop_iteration = min(stop_iteration, self.iteration + sch.stop_after_iterations)

        while self.iteration < stop_iteration:
            # ---------------- (1) collect one vectorised env step
            t0 = time.perf_counter()
            action = self.act(obs)
            obs, rew, term, tout, _extras = env.step(action)
            self.ep_reward_sum += rew
            self.ep_len += 1
            done = term | tout
            if done.any():
                ids = done.nonzero().squeeze(-1)
                lens = self.ep_len[ids]
                norm_ret = self.ep_reward_sum[ids] / (self.max_step_reward * lens.clamp_min(1))
                finished_returns.extend(norm_ret.tolist())
                finished_lens.extend(lens.tolist())
                self.ep_reward_sum[ids] = 0.0
                self.ep_len[ids] = 0.0
            torch.cuda.synchronize()
            timers["sim"] += time.perf_counter() - t0
            self.iteration += 1

            # ---------------- (2)+(3) updates at the configured update-to-data ratio
            if self.iteration - self.start_iteration >= sch.warmup_iterations:
                self._utd_acc += sch.updates_per_env_step
                while self._utd_acc >= 1.0:
                    self._utd_acc -= 1.0
                    lr = cosine_lr(self.iteration, sch.total_iterations, cfg.wm_trainer.lr, sch.lr_final_ratio, sch.lr_decay_fraction)
                    self.wm_trainer.set_lr(lr)
                    self.policy_trainer.set_lr(lr * cfg.policy_trainer.lr / cfg.wm_trainer.lr)
                    t0 = time.perf_counter()
                    m_wm = self.wm_trainer.update(self.buffer)
                    torch.cuda.synchronize()
                    timers["wm"] += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    for _ in range(sch.policy_updates_per_wm_update):
                        m_pi = self.policy_trainer.update(self.buffer)
                    torch.cuda.synchronize()
                    timers["policy"] += time.perf_counter() - t0
                    self.num_updates += 1
                    for k, v in m_wm.items():
                        agg[f"wm/{k}"] = agg.get(f"wm/{k}", 0.0) + v
                    for k, v in m_pi.items():
                        agg[f"policy/{k}"] = agg.get(f"policy/{k}", 0.0) + v
                    agg["lr"] = lr
                    agg_n += 1

            # ---------------- logging / checkpoints
            if self.iteration % sch.log_interval == 0:
                it = self.iteration
                elapsed = time.time() - t_start
                diagnostic_totals = self.command.reset_diagnostics.consume()
                if self.world > 1:
                    # Every rank contributes sums/counts, even with no completed
                    # episodes. Only the log interval converts these to CPU.
                    torch.distributed.all_reduce(diagnostic_totals)
                if self.is_main:
                    for key, value in reset_diagnostic_metrics(diagnostic_totals, self.max_step_reward).items():
                        self.writer.add_scalar(key, value, it)
                if self.world > 1:
                    # Every rank must issue the same collectives: reduce (sum, count) unconditionally.
                    tot = allreduce_mean(float(sum(finished_returns))) * self.world
                    cnt = allreduce_mean(float(len(finished_returns))) * self.world
                    finished_returns = [tot / cnt] if cnt > 0 else []
                    n_finished = round(cnt)
                else:
                    n_finished = len(finished_returns)
                if not self.is_main:
                    agg, agg_n = {}, 0
                    finished_returns, finished_lens = [], []
                    timers = {k: 0.0 for k in timers}
                elif agg_n > 0:
                    for k, v in agg.items():
                        self.writer.add_scalar(k, v / (agg_n if k != "lr" else 1), it)
                if not self.is_main:
                    tr = float("nan")
                elif finished_returns:
                    tr = sum(finished_returns) / len(finished_returns)
                    self.writer.add_scalar("train/episode_norm_return", tr, it)
                    self.writer.add_scalar("train/episode_len", sum(finished_lens) / max(len(finished_lens), 1), it)
                else:
                    tr = float("nan")
                with torch.no_grad():
                    if self.is_main and self.buffer.size > 2048:
                        err = self.wm_trainer.online_error(self.buffer, n=1024, k=8)
                        self.writer.add_scalar("wm/online_nse_1step", float(err[0]), it)
                        self.writer.add_scalar("wm/online_nse_8step", float(err[-1]), it)
                    else:
                        err = None
                if self.is_main:
                    for k, v in timers.items():
                        self.writer.add_scalar(f"time/{k}_s_per_iter", v / sch.log_interval, it)
                    self.writer.add_scalar("time/hours", elapsed / 3600, it)
                    self.writer.add_scalar("time/updates", self.num_updates, it)
                    self.writer.add_scalar("sampler/entropy", float(self.command.metrics["sampling_entropy"][0]), it)
                    self.writer.add_scalar("sampler/bin_return_mean", float(self.command.bin_return.mean()), it)
                    dt_log = time.time() - last_log
                    last_log = time.time()
                    msg = (
                        f"it {it} | {elapsed/3600:.2f} h | {sch.log_interval / dt_log:.1f} it/s | updates {self.num_updates} | "
                        f"train_ret {tr:.3f} (n={n_finished}) | "
                        f"wm nse1 {agg.get('wm/nse_1step', float('nan'))/max(agg_n,1):.4f} nse8 {(float(err[-1]) if err is not None else float('nan')):.4f} | "
                        f"imag_ret {agg.get('policy/imag_return_undisc', float('nan'))/max(agg_n,1):.3f} surv {agg.get('policy/imag_survival_H', float('nan'))/max(agg_n,1):.3f} | "
                        f"t sim/wm/pi {timers['sim']:.1f}/{timers['wm']:.1f}/{timers['policy']:.1f}s"
                    )
                    print(msg, flush=True)
                agg, agg_n = {}, 0
                finished_returns, finished_lens = [], []
                timers = {k: 0.0 for k in timers}
            if self.iteration % sch.checkpoint_interval == 0:
                self.save("ckpt_latest.pt")
                self.save(f"ckpt_{self.iteration:07d}.pt")
            if sch.max_hours is not None and self.iteration % sch.log_interval == 0:
                stop = float((time.time() - t_start) / 3600 > sch.max_hours)
                if self.world > 1:
                    stop = allreduce_mean(stop)  # rank 0's clock decides (all ranks agree on the mean > 0)
                if stop > 0:
                    if self.is_main:
                        print("[INFO] max_hours reached, stopping")
                    break
        self.save("ckpt_latest.pt")
        if self.writer is not None:
            self.writer.flush()
        finalize()
