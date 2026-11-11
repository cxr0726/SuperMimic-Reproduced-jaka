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


def validate_policy_training_semantics(saved_policy: dict, requested) -> None:
    """Do not reinterpret a trained decoder or silently change goal corruption.

    Missing fields in historical checkpoints mean default-pose actions and a
    clean goal. Inactive residual fields do not affect legacy policy behavior.
    """
    defaults = {"action_mode": "default", "reference_residual_beta": 1.0,
                "reference_frame_offset": 1, "goal_noise_z": 0.0,
                "goal_noise_joint_pos": 0.0, "running_normalization": False}
    keys = ["action_mode", "goal_noise_z", "goal_noise_joint_pos", "running_normalization"]
    if requested.action_mode == "reference_residual":
        keys += ["reference_residual_beta", "reference_frame_offset"]
    mismatches = {key: (saved_policy.get(key, defaults[key]), getattr(requested, key))
                  for key in keys if saved_policy.get(key, defaults[key]) != getattr(requested, key)}
    if mismatches:
        raise ValueError(
            f"Checkpoint action/goal-noise semantics differ: {mismatches}. Start a new run "
            "for a different action, goal-noise or normalization configuration. "
            "A legacy run requires its original configuration and branch."
        )


def validate_policy_architecture(saved_policy: dict, requested) -> None:
    """Reject incompatible resume shapes before modifying model/optimizer state."""
    legacy_hidden = (2048, 1024, 512, 256)
    mismatches = {}
    for key in ("encoder_hidden", "decoder_hidden"):
        saved = tuple(saved_policy.get(key, legacy_hidden))
        current = tuple(getattr(requested, key))
        if saved != current:
            mismatches[key] = (saved, current)
    if saved_policy.get("latent_dim", 32) != requested.latent_dim:
        mismatches["latent_dim"] = (saved_policy.get("latent_dim", 32), requested.latent_dim)
    for key, legacy in (("hidden_layer_norm", False), ("hidden_activation", "silu")):
        saved, current = saved_policy.get(key, legacy), getattr(requested, key)
        if saved != current:
            mismatches[key] = (saved, current)
    if mismatches:
        raise ValueError(
            f"Checkpoint policy architecture differs: {mismatches}; start a new run "
            "for the new hidden layers, or explicitly request the saved architecture to resume"
        )


def validate_critic_architecture(saved_critic: dict, requested) -> None:
    """Activation changes are semantic changes even when tensor shapes match."""
    defaults = {"hidden": (256, 128), "hidden_layer_norm": False, "hidden_activation": "silu"}
    mismatches = {}
    for key, legacy in defaults.items():
        saved, current = saved_critic.get(key, legacy), getattr(requested, key)
        if key == "hidden":
            saved, current = tuple(saved), tuple(current)
        if saved != current:
            mismatches[key] = (saved, current)
    if mismatches:
        raise ValueError(f"Checkpoint critic architecture differs: {mismatches}; start a new run "
                         "or explicitly request the saved architecture to resume")


def load_sampler_state(command, checkpoint: dict) -> bool:
    """Restore comparable curriculum statistics, or reset pre-v2 estimates.

    Legacy checkpoints estimated returns for every visited bin without normal
    clip-end continuation. Those estimates must not seed the new start-bin
    curriculum. Model/optimizer state and schedule restoration are independent.
    """
    window = getattr(command, "window_sampler", None)
    if window is not None:
        if checkpoint.get("sampler_contract") != command.sampler_contract:
            raise ValueError("Checkpoint window-value sampler contract differs; start a new run")
        state = checkpoint.get("window_sampler")
        if state is None:
            raise ValueError("Checkpoint is missing its window-value sampler state")
        # Physical worlds/replay are freshly initialized on resume. A partial
        # window cannot be joined to the new trajectories as one episode.
        command.load_window_sampler_state_dict(state, discard_pending=True)
        print("[INFO] restored window sampler value/count/effort; discarded pending "
              "physical window because simulator/replay restart on resume")
        return True
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
                delay_reset_first_step_no_delay=cfg.env.delay_reset_first_step_no_delay,
                reset_root_z_offset=cfg.env.reset_root_z_offset,
                sampling_mode=cfg.env.sampling_mode,
                reset_sampler=cfg.env.reset_sampler,
                return_horizon=cfg.env.return_horizon,
                return_memory=cfg.env.return_memory,
                replay_capacity=cfg.env.replay_capacity,
                compact_library=cfg.env.compact_library,
                reference_fk_compile=cfg.env.reference_fk_compile,
                merge_short_tail_bins=cfg.env.merge_short_tail_bins,
                tail_bin_merge_fraction=cfg.env.tail_bin_merge_fraction,
                reset_min_remaining_steps=cfg.env.reset_min_remaining_steps,
                anchor_obs=cfg.env.anchor_obs,
                root_height_threshold=cfg.world_model.height_threshold,
                root_tilt_threshold=cfg.world_model.tilt_threshold,
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
        action_term = env.action_manager.get_term("joint_pos")
        q_default = robot.data.default_joint_pos[0].clone()
        scale = torch.as_tensor(action_term._scale, device=self.device).expand_as(
            robot.data.default_joint_pos)[0].clone()
        offset = torch.as_tensor(action_term._offset, device=self.device).expand_as(
            robot.data.default_joint_pos)[0].clone()
        self.wm = WorldModel(cfg.world_model, heads, self.device,
                             action_scale=scale, action_offset=offset)
        policy_robot_args = {}
        if cfg.policy.action_mode == "reference_residual":
            # Use exactly the resolved action term's joint order and physical
            # affine mapping, not a second regex expansion of robot defaults.
            action_term = env.action_manager.get_term("joint_pos")
            q_default = robot.data.default_joint_pos[0].clone()
            scale = torch.as_tensor(action_term._scale, device=self.device).expand_as(
                robot.data.default_joint_pos)[0].clone()
            offset = torch.as_tensor(action_term._offset, device=self.device).expand_as(
                robot.data.default_joint_pos)[0]
            if not torch.equal(offset, q_default):
                raise ValueError("Reference residual requires default-pose action offsets")
            policy_robot_args = {"default_joint_pos": q_default, "action_scale": scale}
        self.policy = Policy(cfg.policy, **policy_robot_args).to(self.device)
        if self.buffer.xi.shape[-1] != self.wm.spec.xi_dim:
            raise ValueError("Simulator replay and world model context layouts differ")
        self.critic = Critic(cfg.critic, pcfg=cfg.policy, xi_dim=self.wm.spec.xi_dim).to(self.device)
        for m in (self.wm, self.policy, self.critic):
            broadcast_params(m)
        self.wm_trainer = WorldModelTrainer(self.wm, self.lib, cfg.wm_trainer)
        self.policy_trainer = PolicyTrainer(self.wm, self.policy, self.critic, self.lib, cfg.policy_trainer)
        n_params = lambda m: sum(p.numel() for p in m.parameters())
        if self.is_main:
            if cfg.policy.action_mode == "reference_residual":
                print(f"[INFO] action mode: reference_residual; beta={cfg.policy.reference_residual_beta:g}, "
                      f"frame offset={cfg.policy.reference_frame_offset}")
            else:
                print("[INFO] action mode: default; q_target=q_default+action_scale*actor_output")
            print(f"[INFO] encoder-only goal noise: z +/-{cfg.policy.goal_noise_z:g} m, "
                  f"joints +/-{cfg.policy.goal_noise_joint_pos:g} rad (uniform); "
                  f"collection enabled={cfg.env.obs_noise}, imagination enabled={cfg.policy_trainer.obs_noise}; "
                  "critic and reward references stay clean")
            print(f"[INFO] params: world model {n_params(self.wm)/1e6:.2f}M, policy {n_params(self.policy)/1e6:.2f}M, critic {n_params(self.critic)/1e6:.3f}M")
            print(f"[INFO] policy hidden layers: encoder={cfg.policy.encoder_hidden}, "
                  f"decoder={cfg.policy.decoder_hidden}; "
                  f"actor EMA decay={cfg.policy_trainer.ema_decay:g} "
                  "(0=disabled); collection/optimization use live actor")
            print(f"[INFO] actor blocks: LayerNorm={cfg.policy.hidden_layer_norm}, "
                  f"activation={cfg.policy.hidden_activation}; critic blocks: "
                  f"LayerNorm={cfg.critic.hidden_layer_norm}, activation={cfg.critic.hidden_activation}")
            print(f"[INFO] sim context: {self.wm.spec.context.kind}, xi={self.wm.spec.xi_dim}, "
                  f"fields={self.wm.spec.context.sizes}; delay queue remains implicit")
            delay_active = cfg.env.actuator_delay and cfg.env.domain_randomization
            print(f"[INFO] actuator delay: {'ON' if delay_active else 'OFF'}, "
                  f"range=0..{cfg.env.actuator_delay_max_ms if delay_active else 0:g} ms; "
                  f"first control step immediate={cfg.env.delay_reset_first_step_no_delay}")
            print(f"[INFO] reset sampler: {cfg.env.reset_sampler}; "
                  f"window={cfg.env.return_horizon}, memory={cfg.env.return_memory}; "
                  f"min remaining={cfg.env.reset_min_remaining_steps} steps; "
                  f"merge tails={cfg.env.merge_short_tail_bins}, fraction={cfg.env.tail_bin_merge_fraction:g}; "
                  f"eligible bins={self.command.num_sampleable_bins}/{self.lib.num_bins}; "
                  "first-step action-rate unchanged")
            print(f"[INFO] official components: WM input running norm={cfg.world_model.input_normalization}, "
                  f"WM LayerNorm={cfg.world_model.hidden_layer_norm}/{cfg.world_model.hidden_activation}; "
                  f"termination={cfg.world_model.termination_mode}; actor running norm="
                  f"{cfg.policy.running_normalization}; critic upper bound={cfg.policy_trainer.critic_value_bound}")

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
        checkpoint = {
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
        }
        ema = getattr(self.policy_trainer, "ema", None)
        window = getattr(self.command, "window_sampler", None)
        if window is not None:
            checkpoint["window_sampler"] = self.command.window_sampler_state_dict()
        if ema is not None:
            checkpoint.update(policy_ema=ema.state_dict(), policy_ema_updates=ema.num_updates,
                              policy_ema_decay=ema.decay)
        torch.save(checkpoint, self.log_dir / name)

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
        validate_policy_training_semantics(saved_policy, self.cfg.policy)
        saved_wm = ck["config"].get("world_model", {})
        legacy_wm = {"input_normalization": False, "hidden_layer_norm": False,
                     "hidden_activation": "silu", "termination_mode": "learned",
                     "termination_softness": .05, "height_threshold": .25, "tilt_threshold": .8}
        differences = {name: (saved_wm.get(name, default), getattr(self.cfg.world_model, name))
                       for name, default in legacy_wm.items()
                       if saved_wm.get(name, default) != getattr(self.cfg.world_model, name)}
        saved_bound = ck["config"].get("policy_trainer", {}).get("critic_value_bound", False)
        if saved_bound != self.cfg.policy_trainer.critic_value_bound:
            differences["critic_value_bound"] = (saved_bound, self.cfg.policy_trainer.critic_value_bound)
        if differences:
            raise ValueError(f"Checkpoint training algorithm differs: {differences}; start a new run")
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
        if self.cfg.policy.action_mode == "reference_residual":
            for key in ("reference_q_default", "reference_action_scale"):
                actual = ck["policy"].get(key)
                expected = getattr(self.policy, key)
                if (not isinstance(actual, torch.Tensor) or actual.dtype != expected.dtype
                        or actual.shape != expected.shape
                        or not torch.equal(actual.to(expected), expected)):
                    raise ValueError(f"Checkpoint {key} differs from the current physical action mapping")
        validate_policy_architecture(saved_policy, self.cfg.policy)
        validate_critic_architecture(ck["config"].get("critic", {}), self.cfg.critic)
        ema = getattr(self.policy_trainer, "ema", None)
        if ema is not None:
            ema.validate_checkpoint(ck)
        if self.cfg.world_model.input_normalization:
            for name in ("action_scale", "action_offset"):
                actual, expected = ck["wm"].get(name), getattr(self.wm, name)
                if not isinstance(actual, torch.Tensor) or not torch.equal(actual.to(expected), expected):
                    raise ValueError(f"Checkpoint WM {name} differs from the simulator action mapping")
        if getattr(self.command, "window_sampler", None) is not None:
            if ck.get("sampler_contract") != self.command.sampler_contract:
                raise ValueError("Checkpoint window-value sampler contract differs; start a new run")
            self.command.window_sampler.validate_state_dict(ck.get("window_sampler"))
        self.wm.load_state_dict(ck["wm"])
        self.policy.load_state_dict(ck["policy"])
        self.critic.load_state_dict(ck["critic"])
        self.wm_trainer.opt.load_state_dict(ck["wm_opt"])
        self.policy_trainer.opt.load_state_dict(ck["policy_opt"])
        self.policy_trainer.critic_opt.load_state_dict(ck["critic_opt"])
        if ema is not None:
            ema.restore(self.policy, ck)
            print(f"[INFO] actor EMA: restored={ck.get('policy_ema') is not None}, "
                  f"updates={ema.num_updates}, decay={ema.decay:g}")
        load_sampler_state(self.command, ck)
        self.iteration = ck["iteration"]
        self.start_iteration = self.iteration
        self.num_updates = ck["num_updates"]
        print(f"[INFO] resumed from {path} at iteration {self.iteration}")

    # ------------------------------------------------------------------ main loop

    @torch.no_grad()
    def act(self, obs) -> torch.Tensor:
        # The environment supplies one clean reference window. Only the motion
        # encoder sees corruption; never overwrite the reference/PD skip.
        encoder_goal = self.policy.make_encoder_goal(obs["goal"], enabled=self.cfg.env.obs_noise)
        if self.policy.cfg.running_normalization:
            # Reuse exactly the corruption used by the actual physical action.
            self._actor_norm_inputs = (obs["policy"].detach().clone(), encoder_goal.detach().clone())
        return self.policy(obs["policy"], obs["goal"], encoder_goal=encoder_goal)

    @torch.no_grad()
    def update_collection_normalization(self) -> None:
        """Update once per physical step, after terminal transitions were captured."""
        self.wm.update_normalization(self.recorder.last_step_states,
                                     self.recorder.last_step_actions)
        if self.cfg.policy.running_normalization:
            sensor, encoder_goal = self._actor_norm_inputs
            self.policy.update_normalization(sensor, encoder_goal, encoder_goal=encoder_goal)
            if self.policy_trainer.ema is not None:
                self.policy_trainer.ema.sync_buffers(self.policy)

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
            pre_frames = self.command.frame.clone()
            action = self.act(obs)
            obs, rew, term, tout, _extras = env.step(action)
            self.command.record_window_transition(
                pre_frames, rew, term, tout, env.termination_manager.get_term("clip_end"))
            self.update_collection_normalization()
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
                    for key, value in reset_diagnostic_metrics(
                        diagnostic_totals, self.max_step_reward, sampler_kind=cfg.env.reset_sampler,
                    ).items():
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
                    if self.policy_trainer.ema is not None:
                        self.writer.add_scalar("policy/ema_updates", self.policy_trainer.ema.num_updates, it)
                    self.writer.add_scalar("sampler/entropy", float(self.command.metrics["sampling_entropy"][0]), it)
                    self.writer.add_scalar("sampler/bin_return_mean", float(self.command.bin_return.mean()), it)
                    window = getattr(self.command, "window_sampler", None)
                    if window is not None:
                        self.writer.add_scalar("sampler/window_updates", window.num_updates, it)
                        self.writer.add_scalar("sampler/visited_bin_fraction", float((window.count > 0).float().mean()), it)
                        self.writer.add_scalar("sampler/max_probability", float(window.probabilities.max()), it)
                    if cfg.world_model.input_normalization:
                        self.writer.add_scalar("normalization/wm_count", int(self.wm.state_normalizer.count), it)
                    if cfg.policy.running_normalization:
                        self.writer.add_scalar("normalization/actor_sensor_count",
                                               int(self.policy.decoder.observation_normalizer.sensor.count), it)
                        self.writer.add_scalar("normalization/actor_goal_count",
                                               int(self.policy.encoders["motion"].normalizer.count), it)
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
