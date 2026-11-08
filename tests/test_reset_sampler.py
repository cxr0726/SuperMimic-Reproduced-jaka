"""Reset curriculum invariants, exercised without a simulator or private checkpoint."""

from types import SimpleNamespace

import pytest
import torch

from supermimic.envs.mdp.commands import MultiClipMotionCommand, MultiClipMotionCommandCfg
from supermimic.envs.mdp.reset_sampling import ResetEpisodeDiagnostics, reset_diagnostic_metrics


def _command(n=4):
    """Real command methods on a small CPU library with four one-second bins."""
    cmd = MultiClipMotionCommand.__new__(MultiClipMotionCommand)
    cmd.cfg = MultiClipMotionCommandCfg(
        npz_files=[], anchor_body_name="pelvis", body_names=("pelvis",),
    )
    cmd._env = SimpleNamespace(
        num_envs=n, device="cpu", common_step_counter=0,
        reward_buf=torch.full((n,), 0.08), reset_buf=torch.zeros(n, dtype=torch.bool),
        termination_manager=SimpleNamespace(terminated=torch.zeros(n, dtype=torch.bool)),
        scene=SimpleNamespace(env_origins=torch.zeros(n, 3)),
    )
    lib = SimpleNamespace(
        num_bins=4, num_clips=1, num_frames=200,
        clip_start=torch.tensor([0]), clip_last=torch.tensor([199]),
        frame_clip=torch.zeros(200, dtype=torch.long),
        bin_start=torch.tensor([0, 50, 100, 150]),
        bin_end=torch.tensor([50, 100, 150, 200]),
        bin_clip=torch.zeros(4, dtype=torch.long),
    )
    lib.last_frame_of = lambda frame: torch.full_like(frame, 199)
    lib.bin_of_frame = lambda frame: frame // 50

    def reference(frame):
        count = frame.numel()
        quat = torch.zeros(count, 4)
        quat[:, 0] = 1
        return SimpleNamespace(
            root_pos_w=torch.zeros(count, 3), root_quat_w=quat,
            root_lin_vel_w=torch.zeros(count, 3), root_ang_vel_w=torch.zeros(count, 3),
            joint_pos=torch.zeros(count, 2), joint_vel=torch.zeros(count, 2),
        )

    lib.ref = reference
    cmd.library = lib
    cmd.frame = torch.arange(n, dtype=torch.long) * 50
    cmd.clip_end = torch.zeros(n, dtype=torch.bool)
    cmd.forced_frame = torch.full((n,), -1, dtype=torch.long)
    cmd.bin_return = torch.zeros(4)
    cmd.bin_memory = torch.full((4,), 0.25)
    cmd.sampleable_bins = torch.ones(4, dtype=torch.bool)
    cmd.num_sampleable_bins = 4
    cmd.sample_bin_end = torch.tensor([50, 100, 150, 199])
    cmd.ep_len = torch.zeros(n, dtype=torch.long)
    cmd.ep_reward_sum = torch.zeros(n, dtype=torch.float64)
    cmd.ep_discounted_return = torch.zeros(n, dtype=torch.float64)
    cmd.episode_start_bin = torch.zeros(n, dtype=torch.long)
    cmd.last_logged_step = torch.zeros(n, dtype=torch.long)
    cmd.reset_diagnostics = ResetEpisodeDiagnostics(n, "cpu", cmd.cfg.gamma)
    cmd.metrics = {
        "sampling_entropy": torch.zeros(n), "episode_return_to_go": torch.zeros(n),
    }
    cmd.time_left = torch.full((n,), 1e9)
    cmd.command_counter = torch.zeros(n, dtype=torch.long)
    cmd._update_metrics = lambda: None
    cmd._refresh_ref = lambda: None
    cmd.update_relative_body_poses = lambda: None
    cmd._write_reference_state_to_sim = lambda *args, **kwargs: None
    return cmd


def test_production_sampler_defaults():
    cmd = _command()
    assert cmd.cfg.gamma == 0.99
    assert cmd.cfg.ema_alpha == 0.2
    assert cmd.cfg.memory_alpha == 0.001
    assert (cmd.cfg.uniform_ratio, cmd.cfg.hard_ratio, cmd.cfg.memory_ratio) == (0.5, 0.4, 0.1)


@pytest.mark.parametrize("steps", [1, 3, 10, 50, 500, 1500])
def test_equal_quality_normal_cutoffs_have_equal_score(steps):
    from supermimic.envs.mdp.reset_sampling import mean_reward_continuation

    gamma, reward = 0.99, 0.08
    discount = torch.tensor(gamma, dtype=torch.float64) ** torch.arange(steps)
    score = mean_reward_continuation(
        (discount.sum() * reward).reshape(1), torch.tensor([steps * reward], dtype=torch.float64),
        torch.tensor([steps]), torch.tensor([False]), gamma,
    )
    torch.testing.assert_close(score, torch.tensor([8.0], dtype=score.dtype), atol=1e-6, rtol=1e-6)


def test_true_failure_gets_no_continuation():
    from supermimic.envs.mdp.reset_sampling import mean_reward_continuation

    raw = torch.tensor([0.237608, 0.237608], dtype=torch.float64)
    score = mean_reward_continuation(
        raw, torch.tensor([0.24, 0.24], dtype=torch.float64),
        torch.tensor([3, 3]), torch.tensor([False, True]), 0.99,
    )
    assert score[0] == pytest.approx(8.0)
    assert score[1] == raw[1]


def test_duplicate_start_bins_are_averaged_before_one_ema():
    from supermimic.envs.mdp.reset_sampling import update_bin_returns

    bins, targets = torch.tensor([1, 1, 3]), torch.tensor([3.0, 7.0, 2.0])
    first, reverse = torch.ones(4), torch.ones(4)
    update_bin_returns(first, bins, targets, 0.2)
    update_bin_returns(reverse, bins.flip(0), targets.flip(0), 0.2)
    torch.testing.assert_close(first, torch.tensor([1.0, 1.8, 1.0, 1.2]))
    torch.testing.assert_close(reverse, first)


def test_only_first_real_reward_sets_start_bin_and_only_that_bin_updates():
    cmd = _command(1)
    cmd.frame[:] = 48
    for step in range(1, 6):
        cmd._env.common_step_counter = step
        cmd._log_step_rewards()
        cmd.frame += 1
    assert cmd.episode_start_bin.item() == 0
    assert cmd.frame.item() == 53  # The episode also visited bin 1.
    cmd._finalize_episodes(torch.tensor([0]))
    torch.testing.assert_close(cmd.bin_return, torch.tensor([1.6, 0.0, 0.0, 0.0]))
    assert cmd.metrics["episode_return_to_go"].item() == pytest.approx(8.0)


def test_initial_reset_and_same_step_calls_do_not_add_phantom_rewards():
    cmd = _command(2)
    cmd._env.reward_buf[:] = 9  # Stale buffers must not be interpreted as a transition.
    cmd._log_step_rewards()
    assert cmd.ep_len.tolist() == [0, 0]
    cmd._env.common_step_counter = 1
    cmd._log_step_rewards()
    cmd._log_step_rewards(torch.tensor([True, False]))
    assert cmd.ep_len.tolist() == [1, 1]
    torch.testing.assert_close(cmd.ep_reward_sum, torch.tensor([9.0, 9.0], dtype=torch.float64))


def test_initial_resample_does_not_update_zero_initial_scores():
    cmd = _command(2)
    cmd._env.reward_buf[:] = 9
    cmd._resample_command(torch.tensor([0, 1]))
    torch.testing.assert_close(cmd.bin_return, torch.zeros(4))
    assert cmd.ep_len.tolist() == [0, 0]
    assert cmd.metrics["episode_return_to_go"].tolist() == [0, 0]


def test_physical_failure_still_gets_no_tail_when_clip_end_also_true():
    cmd = _command(1)
    for step in range(1, 4):
        cmd._env.common_step_counter = step
        cmd._log_step_rewards()
    cmd.clip_end[:] = True
    cmd._env.termination_manager.terminated[:] = True
    cmd._finalize_episodes(torch.tensor([0]))
    raw = 0.08 * (1 + 0.99 + 0.99**2)
    assert cmd.metrics["episode_return_to_go"].item() == pytest.approx(raw)
    assert cmd.bin_return[0] == pytest.approx(0.2 * raw)
    # A second finalize has no data and cannot update the bin again.
    before = cmd.bin_return.clone()
    cmd._finalize_episodes(torch.tensor([0]))
    torch.testing.assert_close(cmd.bin_return, before)


def test_manual_reset_does_not_duplicate_the_last_already_logged_transition():
    cmd = _command(1)
    for step, reward in enumerate([0.08, 0.16], 1):
        cmd._env.common_step_counter = step
        cmd._env.reward_buf[:] = reward
        cmd._log_step_rewards()
    cmd._resample_command(torch.tensor([0]))
    expected = 0.08 + 0.99 * 0.16 + 0.99**2 * 0.12 / (1 - 0.99)
    assert cmd.metrics["episode_return_to_go"].item() == pytest.approx(expected)


def test_terminal_reward_is_recorded_once_and_new_episode_remains_empty():
    cmd = _command(2)
    cmd._env.common_step_counter = 1
    cmd.compute(0.02)
    cmd._env.common_step_counter = 2
    cmd._env.reset_buf[0] = True
    cmd._resample_command(torch.tensor([0]))
    assert cmd.ep_len.tolist() == [0, 1]
    assert cmd.metrics["episode_return_to_go"][0] == pytest.approx(8.0)
    reset_frame = cmd.frame[0].clone()
    cmd.compute(torch.tensor([0.0, 0.02]))
    assert cmd.frame[0] == reset_frame
    assert cmd.ep_len.tolist() == [0, 2]
    assert cmd.bin_return[0] == pytest.approx(1.6)


def test_zero_dt_partial_reset_does_not_advance_any_reference_or_log_reward():
    cmd = _command(3)
    cmd._env.common_step_counter = 7
    before = cmd.frame.clone()
    cmd.compute(0.0, env_ids=torch.tensor([1]))
    torch.testing.assert_close(cmd.frame, before)
    assert cmd.ep_len.tolist() == [0, 0, 0]
    cmd.compute(0.02, env_ids=torch.tensor([1]))
    torch.testing.assert_close(cmd.frame, before + torch.tensor([0, 1, 0]))
    assert cmd.ep_len.tolist() == [0, 1, 0]


def test_long_episode_accumulates_without_a_fixed_1024_step_log_cap():
    cmd = _command(1)
    cmd.frame[:] = 10
    for step in range(1, 1501):
        cmd._env.common_step_counter = step
        cmd._log_step_rewards()
    assert cmd.ep_len.item() == 1500
    assert cmd.ep_reward_sum.item() == pytest.approx(120.0, abs=1e-5)
    cmd._finalize_episodes(torch.tensor([0]))
    assert cmd.metrics["episode_return_to_go"].item() == pytest.approx(8.0, abs=1e-5)


def test_probability_mixture_and_history_update_once_per_sampling_call(monkeypatch):
    cmd = _command()
    cmd.bin_return[:] = torch.tensor([0.0, 2.0, 4.0, 6.0])
    old_memory = cmd.bin_memory.clone()
    hard = torch.softmax(-25 * cmd.bin_return / 10, dim=0)
    expected_memory = 0.999 * old_memory + 0.001 * hard
    expected = 0.5 / 4 + 0.4 * hard + 0.1 * expected_memory
    seen = []

    def multinomial(probabilities, n, replacement):
        seen.append(probabilities.clone())
        assert replacement
        return torch.zeros(n, dtype=torch.long)

    monkeypatch.setattr(torch, "multinomial", multinomial)
    frames = cmd._sample_frames(3)
    assert ((frames >= 0) & (frames < 50)).all()
    torch.testing.assert_close(seen[0], expected)
    torch.testing.assert_close(cmd.bin_memory, expected_memory)


def test_zero_initialization_and_invalid_tail_bin_are_excluded_from_all_mixture_terms(monkeypatch):
    cmd = _command()
    # A 151-frame clip has three valid bins and a singleton bin at its last frame.
    cmd.library.num_frames = 151
    cmd.library.clip_last[:] = 150
    cmd.library.bin_end[:] = torch.tensor([50, 100, 150, 151])
    cmd.library.last_frame_of = lambda frame: torch.full_like(frame, 150)
    cmd.sample_bin_end = torch.tensor([50, 100, 150, 150])
    cmd.sampleable_bins = torch.tensor([True, True, True, False])
    cmd.num_sampleable_bins = 3
    cmd.reset_sampler_statistics()
    torch.testing.assert_close(cmd.bin_return, torch.zeros(4))
    torch.testing.assert_close(cmd.bin_memory, torch.tensor([1 / 3, 1 / 3, 1 / 3, 0.0]))
    original = torch.multinomial
    seen = []

    def multinomial(probabilities, n, replacement):
        seen.append(probabilities.clone())
        return original(probabilities, n, replacement)

    monkeypatch.setattr(torch, "multinomial", multinomial)
    frames = cmd._sample_frames(10000)
    assert (frames < 150).all()
    assert seen[0][3] == 0
    torch.testing.assert_close(seen[0][:3], torch.full((3,), 1 / 3))


def test_first_step_diagnostics_use_actual_weighted_rewards_and_preserve_sampler_target():
    cmd = _command(2)
    rates = torch.zeros(2)
    cmd._env.reward_manager = SimpleNamespace(
        active_terms=["action_rate_l2"], get_term=lambda name: rates,
    )
    # Env0 has a one-step real failure; env1 has a three-step normal cutoff.
    for step, (rewards, action_rates) in enumerate([
        ([-0.02, -0.04], [-0.12, -0.14]),
        ([8.0, 0.08], [-9.0, -0.02]),
        ([8.0, 0.10], [-9.0, 0.0]),
    ], 1):
        cmd._env.common_step_counter = step
        cmd._env.reward_buf[:] = torch.tensor(rewards)
        rates[:] = torch.tensor(action_rates)
        mask = torch.tensor([step == 1, True])
        cmd._log_step_rewards(mask)
        # Repeated calls must not count a transition twice.
        cmd._log_step_rewards(mask)
    cmd._env.termination_manager.terminated[0] = True
    cmd._finalize_episodes(torch.tensor([0, 1]))
    totals = cmd.reset_diagnostics.consume()
    stats = reset_diagnostic_metrics(totals, max_step_reward=0.112)
    assert stats["reset/completed_episodes"] == 2
    assert stats["tracking/post_first_step_episodes"] == 1
    assert stats["reset/first_step_reward"] == pytest.approx(-0.03)
    assert stats["reset/first_step_action_rate_reward"] == pytest.approx(-0.13)
    assert stats["tracking/post_first_step_reward"] == pytest.approx(0.09)
    assert stats["tracking/post_first_step_action_rate_reward"] == pytest.approx(-0.01)
    assert stats["tracking/post_first_step_norm_return"] == pytest.approx(0.09 / 0.112)
    factor = 1 + 0.99 ** 3 / (3 * (1 - 0.99))
    assert stats["reset/first_step_action_rate_target_contribution"] == pytest.approx(
        (-0.12 - 0.14 * factor) / 2,
    )
    assert stats["reset/first_step_target_contribution"] + stats[
        "tracking/post_first_step_target_contribution"
    ] == pytest.approx(stats["sampler/completed_episode_target"])
    # Diagnostics remain observational: production G still uses all rewards.
    target1 = -0.04 + 0.99 * 0.08 + 0.99**2 * 0.10 + 0.99**3 * (0.14 / 3) / 0.01
    torch.testing.assert_close(cmd.bin_return[:2], torch.tensor([-0.004, 0.2 * target1]))
    assert cmd.reset_diagnostics.consume().count_nonzero() == 0
    cmd._finalize_episodes(torch.tensor([0, 1]))
    assert cmd.reset_diagnostics.consume().count_nonzero() == 0


def test_single_step_episodes_do_not_fabricate_post_first_means():
    cmd = _command(1)
    cmd._env.common_step_counter = 1
    cmd._log_step_rewards()
    cmd._finalize_episodes(torch.tensor([0]))
    metrics = reset_diagnostic_metrics(cmd.reset_diagnostics.consume(), 0.112)
    assert metrics["reset/completed_episodes"] == 1
    assert metrics["tracking/post_first_step_episodes"] == 0
    assert "tracking/post_first_step_reward" not in metrics
    assert "tracking/post_first_step_norm_return" not in metrics


def test_partial_reset_preserves_other_episode_diagnostics_and_clear_does_not_clear_live_state():
    cmd = _command(2)
    cmd._env.common_step_counter = 1
    cmd._env.reward_buf[:] = torch.tensor([0.02, 0.04])
    cmd._log_step_rewards()
    cmd._resample_command(torch.tensor([0]))
    stats = reset_diagnostic_metrics(cmd.reset_diagnostics.consume(), 0.112)
    assert stats["reset/first_step_reward"] == pytest.approx(0.02)
    assert cmd.reset_diagnostics.first_reward[1] == pytest.approx(0.04)
    cmd._env.common_step_counter = 2
    cmd._env.reward_buf[:] = torch.tensor([0.06, 0.08])
    cmd._log_step_rewards()
    cmd._resample_command(torch.tensor([1]))
    stats = reset_diagnostic_metrics(cmd.reset_diagnostics.consume(), 0.112)
    assert stats["reset/first_step_reward"] == pytest.approx(0.04)
    assert stats["tracking/post_first_step_reward"] == pytest.approx(0.08)


def test_rank_sums_keep_episode_and_step_weighted_diagnostic_denominators():
    rank0, rank1 = ResetEpisodeDiagnostics(1, "cpu", 0.99), ResetEpisodeDiagnostics(1, "cpu", 0.99)
    ids = torch.tensor([0])
    for diag, steps, first, total in [(rank0, 2, 0.01, 0.05), (rank1, 4, 0.02, 0.20)]:
        diag.record(ids, torch.tensor([first]), torch.tensor([-0.03]), torch.tensor([0]))
        diag.finalize(ids, torch.tensor([total]), torch.tensor([steps]), torch.tensor([False]), torch.tensor([1.0]))
    metrics = reset_diagnostic_metrics(rank0.consume() + rank1.consume(), 0.1)
    assert metrics["reset/completed_episodes"] == 2
    assert metrics["tracking/post_first_step_reward"] == pytest.approx((0.04 + 0.18 / 3) / 2)
    assert metrics["tracking/post_first_step_norm_return_step_weighted"] == pytest.approx(0.22 / (4 * 0.1))


def _horizon_command(horizon=32):
    cmd = _command(1)
    # Include short clips before and between eligible clips, and bins that
    # contain frames but no admissible H32 reset state.
    lengths = torch.tensor([10, 33, 32, 101, 125])
    starts = lengths.cumsum(0) - lengths
    bins_per_clip = (lengths + 49) // 50
    bin_clip = torch.repeat_interleave(torch.arange(len(lengths)), bins_per_clip)
    offsets = bins_per_clip.cumsum(0) - bins_per_clip
    bin_start = starts[bin_clip] + (torch.arange(bin_clip.numel()) - offsets[bin_clip]) * 50
    lib = cmd.library
    lib.num_clips, lib.num_frames, lib.num_bins = len(lengths), int(lengths.sum()), bin_clip.numel()
    lib.clip_start, lib.clip_last = starts, starts + lengths - 1
    lib.frame_clip = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    lib.bin_clip, lib.bin_start = bin_clip, bin_start
    lib.bin_end = torch.minimum(bin_start + 50, (starts + lengths)[bin_clip])
    lib.last_frame_of = lambda frames: lib.clip_last[lib.frame_clip[frames]]
    lib.clips = [SimpleNamespace(name=f"clip{i}", length=int(length)) for i, length in enumerate(lengths)]
    cmd.cfg.reset_min_remaining_steps = horizon
    cmd._configure_reset_sampling()
    cmd.bin_return = torch.zeros(lib.num_bins)
    cmd.bin_memory = torch.zeros(lib.num_bins)
    cmd.reset_sampler_statistics()
    return cmd


@pytest.mark.parametrize("mode", ["start", "uniform", "return"])
def test_all_physical_sampling_modes_require_full_horizon(mode):
    torch.manual_seed(54)
    cmd = _horizon_command()
    cmd.cfg.sampling_mode = mode
    frames = cmd._sample_frames(10000)
    clips = cmd.library.frame_clip[frames]
    assert set(clips.tolist()) == {1, 3, 4}
    assert (frames + 32 <= cmd.library.last_frame_of(frames)).all()
    assert cmd.num_sampleable_frames == (33 - 32) + (101 - 32) + (125 - 32)
    assert (cmd.bin_memory[~cmd.sampleable_bins] == 0).all()
    if mode == "start":
        torch.testing.assert_close(frames, cmd.library.clip_start[clips])


def test_uniform_horizon_sampler_reaches_every_admissible_frame(monkeypatch):
    cmd = _horizon_command()
    cmd.cfg.sampling_mode = "uniform"
    monkeypatch.setattr(torch, "randint", lambda low, high, size, device: torch.arange(high))
    frames = cmd._sample_frames(cmd.num_sampleable_frames)
    all_frames = torch.arange(cmd.library.num_frames)
    expected = all_frames[all_frames + 32 <= cmd.library.last_frame_of(all_frames)]
    torch.testing.assert_close(frames, expected)


def test_horizon_reset_allows_exact_boundary_and_rejects_overrides_past_it():
    cmd = _horizon_command()
    valid = cmd.library.clip_last[3] - 32
    cmd.forced_frame[0] = valid
    cmd._resample_command(torch.tensor([0]))
    assert cmd.frame[0] == valid
    cmd.forced_frame[0] = valid + 1
    with pytest.raises(RuntimeError, match="32 subsequent"):
        cmd._resample_command(torch.tensor([0]))
    for bad in [valid + 1, torch.tensor(-1), torch.tensor(cmd.library.num_frames)]:
        before = cmd.frame.clone()
        with pytest.raises(ValueError, match="32 subsequent"):
            cmd.reset_to_frame(torch.tensor([0]), bad.reshape(1))
        torch.testing.assert_close(cmd.frame, before)
    cmd.reset_to_frame(torch.tensor([0]), valid.reshape(1))
    assert cmd.frame[0] == valid


@pytest.mark.parametrize("horizon", [0, -1, 1.5, True])
def test_invalid_reset_horizon_rejected(horizon):
    with pytest.raises(ValueError, match="positive integer"):
        _horizon_command(horizon)


def test_no_full_horizon_reset_state_is_reported_instead_of_silently_clamping():
    with pytest.raises(ValueError, match="125 subsequent"):
        _horizon_command(125)


@pytest.mark.parametrize("merge,fraction,horizon,version", [
    (False, .5, 1, 2), (True, .5, 1, 3),
    (True, 1., 1, 4), (True, .5, 32, 4), (False, .5, 32, 4), (True, 1., 32, 4),
])
def test_sampler_contract_versions_distinguish_boundary_semantics(merge, fraction, horizon, version):
    cmd = _horizon_command(horizon)
    cmd.cfg.merge_short_tail_bins = merge
    cmd.cfg.tail_bin_merge_fraction = fraction
    contract = cmd._make_sampler_contract()
    assert contract["version"] == version
    if version <= 3:
        assert "tail_bin_merge_fraction" not in contract
        assert "reset_min_remaining_steps" not in contract
        assert contract["reset_boundary"] == "frame_before_inclusive_last"
        if version == 2:
            assert "tail_bin_policy" not in contract
        else:
            assert contract["tail_bin_policy"] == "merge_remainder_strictly_less_than_half_reference_bin"
    else:
        assert contract["merge_short_tail_bins"] == merge
        assert contract["tail_bin_merge_fraction"] == fraction
        assert contract["reset_min_remaining_steps"] == horizon
