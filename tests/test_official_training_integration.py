"""Fresh-physics collection and checkpoint semantic boundaries of the official bundle."""

import copy
from dataclasses import asdict
from types import MethodType, SimpleNamespace

import pytest
import torch

from supermimic.config import TrainConfig, resolve_training_observations
from supermimic.data.recorder import ReplayRecorder
from supermimic.envs.mdp.commands import MultiClipMotionCommand
from supermimic.envs.mdp.reset_sampling import WindowValueSampler, WindowValueSamplerCfg
from supermimic.policy.ema import ActorEMA
from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg
from supermimic.robots.registry import get_robot_spec
from supermimic.training import loop as loop_module
from supermimic.training.loop import Trainer
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.world_model import WorldModel


def _trainer(tmp_path, *, actor_noise=True):
    trainer = Trainer.__new__(Trainer)
    policy_cfg = PolicyCfg(
        encoder_hidden=(8,), decoder_hidden=(8,), latent_dim=4, history=5,
        future_frames=5, num_joints=27, goal_dim=155, observation_layout="isaac_khan",
        action_mode="default", goal_noise_z=.01, goal_noise_joint_pos=.05,
        running_normalization=True, hidden_layer_norm=True, hidden_activation="gelu",
    )
    critic_cfg = CriticCfg(hidden=(8,), hidden_layer_norm=True, hidden_activation="gelu")
    trainer.cfg = TrainConfig(robot="khan_mini", device="cpu", policy=policy_cfg, critic=critic_cfg)
    trainer.cfg.env.obs_noise = actor_noise
    trainer.cfg.env.num_envs = 3
    trainer.cfg.world_model.hidden = (8,)
    trainer.cfg.schedule.total_iterations = 3
    trainer.cfg.schedule.warmup_iterations = 40
    trainer.cfg.schedule.log_interval = 50
    trainer.cfg.schedule.checkpoint_interval = 50
    resolve_training_observations(trainer.cfg)
    heads = AnalyticHeads.__new__(AnalyticHeads)
    heads.num_joints, heads.default_joint_pos = 27, torch.zeros(27)
    trainer.wm = WorldModel(
        trainer.cfg.world_model, heads, "cpu", action_scale=torch.full((27,), .5),
        action_offset=torch.linspace(-.2, .2, 27),
    )
    trainer.policy = Policy(policy_cfg)
    trainer.critic = Critic(critic_cfg, pcfg=policy_cfg, xi_dim=trainer.wm.spec.xi_dim)
    trainer.wm_trainer = SimpleNamespace(opt=torch.optim.Adam(trainer.wm.parameters()))
    trainer.policy_trainer = SimpleNamespace(
        opt=torch.optim.Adam(trainer.policy.parameters()),
        critic_opt=torch.optim.Adam(trainer.critic.parameters()),
        ema=ActorEMA(trainer.policy),
    )
    last_frame_of = lambda frames: torch.full_like(frames, 99)
    window = WindowValueSampler(WindowValueSamplerCfg(horizon=3), 4, 100, 3, .112,
                                last_frame_of=last_frame_of)
    trainer.command = SimpleNamespace(
        frame=torch.tensor([1, 20, 40]), window_sampler=window,
        bin_return=window.value, bin_memory=torch.full((4,), .25),
        sampler_contract={"version": 6, "mapping": "4bins100frames",
                          "bootstrap_next_frame": "clip_clamped"},
        metrics={"sampling_entropy": torch.ones(3)}, num_sampleable_bins=4,
        sampleable_bins=torch.ones(4, dtype=torch.bool),
        cfg=SimpleNamespace(sampling_mode="return"),
        library=SimpleNamespace(frame_clip=torch.zeros(100, dtype=torch.long),
                                last_frame_of=last_frame_of),
    )
    trainer.command.window_sampler_state_dict = MethodType(
        MultiClipMotionCommand.window_sampler_state_dict, trainer.command)
    trainer.command.load_window_sampler_state_dict = MethodType(
        MultiClipMotionCommand.load_window_sampler_state_dict, trainer.command)
    trainer.command.record_window_transition = MethodType(
        MultiClipMotionCommand.record_window_transition, trainer.command)
    trainer.is_main, trainer.world, trainer.rank = True, 1, 0
    trainer.device, trainer.log_dir, trainer.writer = "cpu", tmp_path, None
    trainer.robot_spec = get_robot_spec("khan_mini")
    trainer.iteration, trainer.start_iteration, trainer.num_updates, trainer._utd_acc = 0, 0, 0, 0.
    trainer.ep_reward_sum, trainer.ep_len = torch.zeros(3), torch.zeros(3)
    trainer.max_step_reward = .112
    return trainer


class _RecordedBuffer:
    def __init__(self):
        self.rows = []
        self.end_calls = []

    def add(self, ids, **kwargs):
        self.rows.append((ids.clone(), {key: value.clone() for key, value in kwargs.items()}))

    def end_episode(self, ids):
        self.end_calls.append(ids.clone())


class _Physics:
    """Mimic mjlab's pre_reset -> auto reset -> post_reset -> post_step hook order."""

    def __init__(self, trainer):
        self.num_envs, self.device, self.count = 3, "cpu", 0
        self.state = torch.zeros(3, 67)
        self.state[:, 3] = 1.
        self.state[:, 2] = torch.tensor([.6, .7, .8])
        self.xi = trainer.wm.spec.xi_offset.repeat(3, 1)
        self.cmd = trainer.command
        self.command_manager = SimpleNamespace(get_term=lambda name: self.cmd)
        self.actuator = SimpleNamespace(executed_raw=torch.zeros(3, 27))
        self.action_manager = SimpleNamespace(
            action=torch.zeros(3, 27), get_term=lambda name: self.actuator,
        )
        self.reset_buf = torch.zeros(3, dtype=torch.bool)
        self.reset_terminated = self.reset_buf.clone()
        self.clip_ended = self.reset_buf.clone()
        self.termination_manager = SimpleNamespace(get_term=lambda name: self.clip_ended)
        self.reward_buf = torch.zeros(3)
        self.recorder = ReplayRecorder.__new__(ReplayRecorder)
        self.recorder._env = self
        self.recorder.enabled, self.recorder.command_name, self.recorder.action_name = True, "motion", "joint_pos"
        self.recorder.state_io = SimpleNamespace(read=lambda: self.state)
        self.recorder.ctx = SimpleNamespace(read=lambda: self.xi)
        self.recorder.prev_s, self.recorder.prev_xi = self.state.clone(), self.xi.clone()
        self.recorder.prev_frame, self.recorder.prev_clip = self.cmd.frame.clone(), torch.zeros(3, dtype=torch.long)
        self.recorder.last_step_states, self.recorder.last_step_actions = self.state.clone(), torch.zeros(3, 27)
        self.recorder.ep_start = torch.ones(3, dtype=torch.bool)
        self.recorder.action_dim = 27
        self.recorder.buffer = _RecordedBuffer()
        self.pre_frames, self.pre_states, self.executed = [], [], []

    def observation(self):
        # Distinct physical time values reveal accidental use of post-reset obs.
        return {
            "policy": torch.full((3, 465), .1 * self.count),
            "goal": torch.full((3, 155), .2 + .01 * self.count),
        }

    def reset(self):
        self.recorder.record_post_reset(torch.arange(3))
        return self.observation(), {}

    def step(self, action):
        self.pre_frames.append(self.cmd.frame.clone())
        self.pre_states.append(self.state.clone())
        self.action_manager.action.copy_(action)
        self.actuator.executed_raw.copy_(action + .1 * (self.count + 1))
        self.executed.append(self.actuator.executed_raw.clone())
        self.state[:, :3] += .01
        self.cmd.frame += 1
        if self.count == 0:
            terminated = torch.tensor([True, False, True])
            timeouts = torch.tensor([False, True, True])
            self.clip_ended.copy_(torch.tensor([False, True, False]))
        else:
            terminated, timeouts = torch.zeros(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool)
            self.clip_ended.zero_()
        self.reset_terminated, self.reset_buf = terminated, terminated | timeouts
        self.reward_buf.fill_(.1)
        ids = self.reset_buf.nonzero().flatten()
        if ids.numel():
            self.recorder.record_pre_reset(ids)
            self.state[ids, :3] = -10.  # Distinguish reset state from previous physical state.
            self.cmd.frame[ids] = torch.tensor([55, 65, 75])[ids]
            self.action_manager.action[ids] = 0.
            self.actuator.executed_raw[ids] = 0.
            self.recorder.record_post_reset(ids)
        self.recorder.record_post_step()
        self.count += 1
        return self.observation(), self.reward_buf, terminated, timeouts, {}


@pytest.mark.parametrize("actor_noise", [False, True])
def test_real_run_records_terminal_actions_preframes_and_fresh_statistics_once(tmp_path, monkeypatch, actor_noise):
    trainer = _trainer(tmp_path, actor_noise=actor_noise)
    env = _Physics(trainer)
    trainer.env, trainer.recorder = env, env.recorder
    recorded_goals, updated_goals, recorded_frames = [], [], []
    noise_draws = []
    make_goal, actor_update = trainer.policy.make_encoder_goal, trainer.policy.update_normalization
    record_window = trainer.command.record_window_transition

    def draw_goal(goal, enabled=True):
        result = make_goal(goal, enabled)
        noise_draws.append(result.clone())
        return result

    def update_actor(obs, goal, **kwargs):
        updated_goals.append(kwargs["encoder_goal"].clone())
        return actor_update(obs, goal, **kwargs)

    def record_frame(*args):
        recorded_frames.append(tuple(value.clone() for value in args))
        return record_window(*args)

    hook = trainer.policy.encoders["motion"].normalizer.register_forward_pre_hook(
        lambda module, args: recorded_goals.append(args[0].clone()))
    monkeypatch.setattr(trainer.policy, "make_encoder_goal", draw_goal)
    monkeypatch.setattr(trainer.policy, "update_normalization", update_actor)
    monkeypatch.setattr(trainer.command, "record_window_transition", record_frame)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(loop_module, "finalize", lambda: None)
    saved = []
    monkeypatch.setattr(trainer, "save", lambda *args: saved.append(trainer.iteration))
    trainer.run()
    hook.remove()

    assert saved == [3] and env.count == 3
    assert len(recorded_frames) == len(recorded_goals) == len(updated_goals) == len(noise_draws) == 3
    for frames, expected in zip(recorded_frames, env.pre_frames, strict=True):
        torch.testing.assert_close(frames[0], expected, rtol=0, atol=0)
    assert not torch.equal(recorded_frames[0][0], env.cmd.frame)
    for used, updated, drawn in zip(recorded_goals, updated_goals, noise_draws, strict=True):
        torch.testing.assert_close(used, updated, rtol=0, atol=0)
        torch.testing.assert_close(used, drawn, rtol=0, atol=0)
    assert trainer.wm.state_normalizer.count.item() == 9
    assert trainer.wm.action_normalizer.count.item() == 9
    assert trainer.policy.decoder.observation_normalizer.sensor.count.item() == 9
    assert trainer.policy.decoder.observation_normalizer.orientation.count.item() == 9
    assert trainer.policy.encoders["motion"].normalizer.count.item() == 9
    all_states, all_actions = torch.cat(env.pre_states), torch.cat(env.executed)
    torch.testing.assert_close(trainer.wm.state_normalizer.mean,
                               (trainer.wm.spec.canonical(all_states) * trainer.wm.spec.in_scale).mean(0))
    torch.testing.assert_close(trainer.wm.action_normalizer.mean,
                               trainer.wm.proc_action(all_actions).mean(0))
    torch.testing.assert_close(trainer.policy.encoders["motion"].normalizer.mean,
                               torch.cat(recorded_goals).mean(0))
    assert trainer.command.window_sampler.num_updates == 1
    assert trainer.command.window_sampler._step == 0
    # Both timeout+failure and clip_end are truncations in the official sampler.
    assert trainer.command.window_sampler._terminated[:, 0].tolist() == [True, False, False]
    assert trainer.command.window_sampler._clip_end[:, 0].tolist() == [False, True, False]
    assert trainer.command.window_sampler._timed_out[:, 0].tolist() == [False, False, True]
    # Fresh stats are synchronized to the deployment actor without EMA weight updates.
    assert trainer.policy_trainer.ema.num_updates == 0
    live_buffers = dict(trainer.policy.named_buffers())
    for name, shadow in trainer.policy_trainer.ema.module.named_buffers():
        torch.testing.assert_close(shadow, live_buffers[name], rtol=0, atol=0)
    assert len(env.recorder.buffer.end_calls) == 1


class _TailPhysics(_Physics):
    """Use the real command advance after termination, including partial resets."""

    def __init__(self, trainer):
        super().__init__(trainer)
        self.cmd.frame = torch.tensor([99, 98, 99])
        self.cmd.clip_end = torch.zeros(3, dtype=torch.bool)
        self.cmd.library = SimpleNamespace(
            frame_clip=torch.arange(200) // 100,
            last_frame_of=lambda frames: torch.where(frames < 100, 99, 199),
        )
        self.cmd._log_step_rewards = lambda advance: None
        self.cmd._refresh_ref = lambda: None
        self.cmd.update_relative_body_poses = lambda: None
        self.post_frames, self.manager_end, self.command_end = [], [], []

    def step(self, action):
        self.pre_frames.append(self.cmd.frame.clone())
        self.pre_states.append(self.state.clone())
        self.action_manager.action.copy_(action)
        self.actuator.executed_raw.copy_(action + .1 * (self.count + 1))
        self.executed.append(self.actuator.executed_raw.clone())
        self.state[:, :3] += .01
        # The manager sees the previous command flag, before the command advances.
        self.clip_ended.copy_(self.cmd.clip_end)
        self.manager_end.append(self.clip_ended.clone())
        terminated = torch.tensor([False, False, self.count == 0])
        timeouts = self.clip_ended.clone()
        if self.count == 0:
            timeouts[2] = True  # Simultaneous raw failure+timeout retains timeout priority.
        self.reset_terminated, self.reset_buf = terminated, terminated | timeouts
        self.reward_buf.fill_(.1)
        ids = self.reset_buf.nonzero().flatten()
        if ids.numel():
            self.recorder.record_pre_reset(ids)
            self.state[ids, :3] = -10.
            self.cmd.frame[ids] = 150  # A different clip, never the old bootstrap successor.
            self.cmd.clip_end[ids] = False
            self.action_manager.action[ids] = 0.
            self.actuator.executed_raw[ids] = 0.
        self.cmd._compute_dt = torch.full((3,), .02)
        self.cmd._compute_dt[ids] = 0.
        MultiClipMotionCommand._update_command(self.cmd)
        self.post_frames.append(self.cmd.frame.clone())
        self.command_end.append(self.cmd.clip_end.clone())
        if ids.numel():
            self.recorder.record_post_reset(ids)
        self.recorder.record_post_step()
        self.count += 1
        return self.observation(), self.reward_buf, terminated, timeouts, {}


def test_run_bootstrap_uses_source_clip_with_real_command_order_and_partial_reset(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path, actor_noise=False)
    env = _TailPhysics(trainer)
    trainer.env, trainer.recorder = env, env.recorder
    # Force every collection column to need a bootstrap; the sampler boundary
    # suite separately covers the actual H20 end-column alignment.
    window = WindowValueSampler(
        WindowValueSamplerCfg(horizon=1), 4, 200, 3, .112,
        bin_of_frame=lambda frames: frames // 50,
        last_frame_of=env.cmd.library.last_frame_of,
    )
    window.value.copy_(torch.tensor([2., 4., 8., 9.]))
    trainer.command.window_sampler = window
    trainer.command.bin_return = window.value
    returns = []
    original_update = window._update

    def capture_update():
        returns.append(window._returns().clone())
        original_update()

    monkeypatch.setattr(window, "_update", capture_update)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(loop_module, "finalize", lambda: None)
    monkeypatch.setattr(trainer, "save", lambda *args: None)
    trainer.run()

    # First-last-frame does not become an earlier physical clip termination.
    assert [row.tolist() for row in env.pre_frames] == [
        [99, 98, 99], [99, 99, 150], [150, 99, 151],
    ]
    assert [row.tolist() for row in env.post_frames] == [
        [99, 99, 150], [150, 99, 151], [151, 150, 152],
    ]
    assert [row.tolist() for row in env.manager_end] == [
        [False, False, False], [True, False, False], [False, True, False],
    ]
    assert env.command_end[0].tolist() == [True, False, False]
    # Env0 hits the terminal frame without a done yet; env2 timed out and reset
    # to B's frame150. Both must bootstrap A's frame99 (V4), not B's V8/V9.
    torch.testing.assert_close(returns[0][:, 0], torch.full((3,), .1 + .99 * 4),
                               rtol=1e-6, atol=1e-6)
    assert returns[0][2, 0].item() != pytest.approx(.1 + .99 * 9)
    assert window.num_updates == 3
    assert trainer.wm.state_normalizer.count.item() == 9
    assert trainer.policy.encoders["motion"].normalizer.count.item() == 9
    assert trainer.policy.decoder.observation_normalizer.sensor.count.item() == 9
    assert trainer.policy_trainer.ema.num_updates == 0


def _state(trainer):
    return {
        "models": [copy.deepcopy(model.state_dict()) for model in
                   (trainer.wm, trainer.policy, trainer.critic, trainer.policy_trainer.ema.module)],
        "optimizers": [copy.deepcopy(opt.state_dict()) for opt in
                       (trainer.wm_trainer.opt, trainer.policy_trainer.opt,
                        trainer.policy_trainer.critic_opt)],
        "sampler": copy.deepcopy(trainer.command.window_sampler.state_dict()),
        "iteration": (trainer.iteration, trainer.num_updates, trainer.policy_trainer.ema.num_updates),
    }


def _assert_tree_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            _assert_tree_equal(actual[key], value)
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_tree_equal(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize("change, error", [
    ("actor_norm", "semantics differ"), ("action_mode", "semantics differ"),
    ("wm_norm", "training algorithm differs"), ("wm_ln", "training algorithm differs"),
    ("wm_term", "training algorithm differs"), ("critic_bound", "training algorithm differs"),
    ("action_scale", "WM action_scale differs"), ("sampler_contract", "sampler contract differs"),
    ("sampler_state", "Window sampler"),
    ("actor_ln", "policy architecture differs"),
    ("actor_activation", "policy architecture differs"),
    ("critic_ln", "critic architecture differs"),
    ("critic_activation", "critic architecture differs"),
    ("sampler_successor", "bootstrap_next_frame differs"),
])
def test_resume_semantic_rejections_precede_all_model_optimizer_and_sampler_mutations(tmp_path, change, error):
    original = _trainer(tmp_path)
    original.save()
    path = tmp_path / "ckpt_latest.pt"
    saved = torch.load(path, weights_only=False)
    if change == "actor_norm":
        saved["config"]["policy"].pop("running_normalization")
    elif change == "action_mode":
        saved["config"]["policy"]["action_mode"] = "reference_residual"
    elif change == "wm_norm":
        saved["config"]["world_model"].pop("input_normalization")
    elif change == "wm_ln":
        saved["config"]["world_model"]["hidden_layer_norm"] = False
    elif change == "wm_term":
        saved["config"]["world_model"]["termination_mode"] = "learned"
    elif change == "critic_bound":
        saved["config"]["policy_trainer"].pop("critic_value_bound")
    elif change == "action_scale":
        saved["wm"]["action_scale"].fill_(.25)
    elif change == "sampler_contract":
        saved["sampler_contract"] = {"version": "different"}
    elif change == "sampler_state":
        saved["window_sampler"] = None
    elif change == "actor_ln":
        saved["config"]["policy"].pop("hidden_layer_norm")
    elif change == "actor_activation":
        saved["config"]["policy"].pop("hidden_activation")
    elif change == "critic_ln":
        saved["config"]["critic"].pop("hidden_layer_norm")
    elif change == "critic_activation":
        saved["config"]["critic"].pop("hidden_activation")
    elif change == "sampler_successor":
        saved["window_sampler"]["bootstrap_next_frame"] = "global_clamped"
    # If load starts copying weights too early, these incompatible values expose it.
    parameter_name = next(iter(dict(original.wm.named_parameters())))
    saved["wm"][parameter_name].fill_(7.)
    torch.save(saved, path)
    restored = _trainer(tmp_path)
    before = _state(restored)
    with pytest.raises(ValueError, match=error):
        restored.load(path)
    _assert_tree_equal(_state(restored), before)


def test_resume_restores_window_moments_but_drops_pending_physical_trajectory(tmp_path):
    original = _trainer(tmp_path)
    state = torch.zeros(3, 67)
    state[:, 3] = 1.
    state[:, 2] = torch.tensor([.5, .6, .7])
    original.wm.update_normalization(state, torch.randn(3, 27))
    original.policy.update_normalization(torch.randn(3, 465), torch.randn(3, 155))
    original.policy_trainer.ema.sync_buffers(original.policy)
    window = original.command.window_sampler
    window.value.copy_(torch.tensor([1., 2., 3., 4.]))
    window.count.copy_(torch.tensor([3., 4., 5., 6.]))
    window.record(torch.tensor([1, 20, 40]), torch.full((3,), .1),
                  torch.zeros(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool),
                  torch.zeros(3, dtype=torch.bool))
    assert window._step == 1
    original.save()
    restored = _trainer(tmp_path)
    restored.load(tmp_path / "ckpt_latest.pt")
    for name in ("models", "optimizers", "iteration"):
        _assert_tree_equal(_state(restored)[name], _state(original)[name])
    assert restored.wm.state_normalizer.count.item() == 3
    assert restored.policy.encoders["motion"].normalizer.count.item() == 3
    assert restored.command.window_sampler._step == 0
    for name in ("value", "count", "effort", "probabilities"):
        torch.testing.assert_close(getattr(restored.command.window_sampler, name),
                                   getattr(window, name), rtol=0, atol=0)
    for name in ("_frames", "_rewards", "_terminated", "_clip_end", "_timed_out"):
        assert getattr(restored.command.window_sampler, name).count_nonzero() == 0
    assert asdict(restored.cfg)["policy"]["action_mode"] == "default"
