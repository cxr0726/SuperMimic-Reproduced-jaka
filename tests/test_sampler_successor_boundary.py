"""Clip-aware bootstrap successors without changing physical episode timing."""

import copy
from types import SimpleNamespace

import pytest
import torch

from supermimic.envs.mdp.commands import MultiClipMotionCommand, MultiClipMotionCommandCfg
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.envs.mdp.reset_sampling import WindowValueSampler, WindowValueSamplerCfg


def _library(lengths=(100, 100), *, merge=True):
    """Use the real query methods on a small, explicit clip/bin partition."""
    lib = MotionLibrary.__new__(MotionLibrary)
    lib.clip_len = torch.tensor(lengths)
    lib.clip_start = lib.clip_len.cumsum(0) - lib.clip_len
    lib.clip_last = lib.clip_start + lib.clip_len - 1
    lib.num_frames = int(lib.clip_len.sum())
    lib.num_clips = len(lengths)
    lib.frame_clip = torch.repeat_interleave(torch.arange(len(lengths)), lib.clip_len)
    lib.bin_frames, lib.merge_short_tail_bins = 50, merge
    lib.bins_per_clip = ((lib.clip_len // 50).clamp_min(1) if merge else
                         (lib.clip_len + 49) // 50)
    lib.clip_bin_offset = lib.bins_per_clip.cumsum(0) - lib.bins_per_clip
    lib.num_bins = int(lib.bins_per_clip.sum())
    lib.bin_clip = torch.repeat_interleave(torch.arange(len(lengths)), lib.bins_per_clip)
    local = torch.arange(lib.num_bins) - lib.clip_bin_offset[lib.bin_clip]
    lib.bin_start = lib.clip_start[lib.bin_clip] + local * 50
    lib.bin_end = torch.where(local == lib.bins_per_clip[lib.bin_clip] - 1,
                             lib.clip_last[lib.bin_clip] + 1, lib.bin_start + 50)
    lib.clips = [SimpleNamespace(name=f"clip{i}", length=n) for i, n in enumerate(lengths)]
    return lib


class _CapturedSampler(WindowValueSampler):
    def _update(self):
        self.scored_returns = self._returns().clone()
        super()._update()


def _sampler(lib=None, *, horizon=20, num_envs=1, clip_aware=True):
    lib = _library() if lib is None else lib
    sampler = _CapturedSampler(
        WindowValueSamplerCfg(horizon=horizon), lib.num_bins, lib.num_frames,
        num_envs, max_step_reward=.112, bin_of_frame=lib.bin_of_frame,
        last_frame_of=lib.last_frame_of if clip_aware else None,
    )
    sampler.value.copy_(torch.tensor([2., 4., 8., 9.])[:lib.num_bins])
    return sampler


def _record(sampler, frames, *, terminated=False, clip_end=False, timed_out=False):
    n = sampler.num_envs
    frame = torch.as_tensor(frames).expand(n).clone()
    flag = lambda value: torch.as_tensor(value, dtype=torch.bool).expand(n).clone()
    return sampler.record(frame, torch.full((n,), .1), flag(terminated),
                          flag(clip_end), flag(timed_out))


def _command(lib, sampler, initial=80):
    cmd = MultiClipMotionCommand.__new__(MultiClipMotionCommand)
    cmd.cfg = MultiClipMotionCommandCfg(
        npz_files=[], anchor_body_name="pelvis", body_names=("pelvis",),
        reset_sampler="window_value", return_horizon=sampler.cfg.horizon,
    )
    cmd.library, cmd.window_sampler = lib, sampler
    cmd.frame = torch.full((sampler.num_envs,), initial, dtype=torch.long)
    cmd.clip_end = torch.zeros(sampler.num_envs, dtype=torch.bool)
    cmd.bin_memory = torch.full((lib.num_bins,), 1 / lib.num_bins)
    cmd.metrics = {"sampling_entropy": torch.zeros(sampler.num_envs)}
    cmd.num_sampleable_bins = lib.num_bins
    cmd._env = SimpleNamespace(num_envs=sampler.num_envs, device="cpu")
    cmd._compute_dt = .02
    cmd._log_step_rewards = lambda *args: None
    cmd._refresh_ref = lambda: None
    cmd.update_relative_body_poses = lambda: None
    return cmd


def test_real_command_keeps_its_old_end_timing_but_h20_bootstrap_stays_in_clip():
    lib = _library()
    traces = []
    for fixed in (False, True):
        sampler = _sampler(lib, clip_aware=fixed)
        cmd = _command(lib, sampler)
        trace = []
        for _ in range(20):
            pre_frame, manager_clip_end = cmd.frame.clone(), cmd.clip_end.clone()
            cmd._update_command()
            trace.append((int(pre_frame), bool(manager_clip_end), int(cmd.frame), bool(cmd.clip_end)))
            cmd.record_window_transition(pre_frame, torch.tensor([.1]),
                                         torch.tensor([False]), torch.tensor([False]), manager_clip_end)
        traces.append(trace)
        assert trace[-2:] == [(98, False, 99, False), (99, False, 99, True)]
        expected_last = .1 + .99 * (4 if fixed else 8)
        assert sampler.scored_returns[0, -1] == pytest.approx(expected_last, abs=1e-6)
    assert traces[0] == traces[1], "The fix must not change physical end timing or reference frames"


def test_last_frame_in_middle_of_window_uses_the_following_marked_clip_end():
    fixed, original = _sampler(), _sampler(clip_aware=False)
    # pre99 is reached before window end; the next physics step sees clip_end.
    for i, frame in enumerate(range(82, 100)):
        _record(fixed, frame)
        _record(original, frame)
    for sampler in (fixed, original):
        _record(sampler, 99, clip_end=True)
        _record(sampler, 120)  # A reset can switch clips within the same window.
    torch.testing.assert_close(fixed.scored_returns, original.scored_returns, rtol=0, atol=0)
    torch.testing.assert_close(fixed.value, original.value, rtol=0, atol=0)
    torch.testing.assert_close(fixed.count, original.count, rtol=0, atol=0)


@pytest.mark.parametrize("terminated,timed_out,clip_end,expected", [
    (False, False, False, 4.06), (False, True, False, 4.06),
    (True, False, False, .1), (True, True, False, 4.06),
    (False, False, True, .1 + .99 * 5.75),
    (True, False, True, .1 + .99 * 5.75),
])
def test_command_preserves_failure_timeout_and_explicit_clip_end_tail_rules(
    terminated, timed_out, clip_end, expected,
):
    sampler = _sampler(horizon=20)
    cmd = _command(_library(), sampler)
    # End/timeout at the seventh window column, not just the window boundary.
    for _ in range(6):
        _record(sampler, 90)
    cmd.record_window_transition(torch.tensor([99]), torch.tensor([.1]),
                                 torch.tensor([terminated]), torch.tensor([timed_out]),
                                 torch.tensor([clip_end]))
    # Continue an unended trajectory at its clamped final frame. Ended rows
    # restart at another reference; done flags isolate those rewards.
    ended = terminated or timed_out or clip_end
    for _ in range(13):
        _record(sampler, 120 if ended else 99)
    if ended:
        assert sampler.scored_returns[0, 6] == pytest.approx(expected, abs=1e-6)
    else:
        assert sampler.scored_returns[0, -1] == pytest.approx(expected, abs=1e-6)


def test_partial_reset_uses_saved_source_frame_not_live_reset_reference():
    lib, sampler = _library(), _sampler(horizon=1, num_envs=3)
    cmd = _command(lib, sampler)
    pre_frames = torch.tensor([99, 49, 199])
    # All live rows may already refer to new clips when the trainer records.
    cmd.frame.copy_(torch.tensor([150, 80, 0]))
    cmd.record_window_transition(pre_frames, torch.full((3,), .1),
                                 torch.tensor([False, False, False]),
                                 torch.tensor([True, False, True]), torch.zeros(3, dtype=torch.bool))
    torch.testing.assert_close(sampler.scored_returns[:, 0], torch.tensor([4.06, 4.06, 9.01]))
    torch.testing.assert_close(sampler._frames[:, 0], pre_frames)


def test_merged_tail_and_every_frame_successor_preserve_source_clip_membership():
    lib = _library((77, 123))  # Merged tails give bin sizes 77,50,73.
    sampler = _sampler(lib, horizon=1, num_envs=lib.num_frames)
    frames = torch.arange(lib.num_frames)
    next_frames = torch.minimum(frames + 1, lib.last_frame_of(frames))
    torch.testing.assert_close(lib.frame_clip[next_frames], lib.frame_clip[frames], rtol=0, atol=0)
    assert lib.bin_of_frame(torch.tensor([76, 77, 199])).tolist() == [0, 1, 2]
    _record(sampler, frames)
    assert sampler.scored_returns[76, 0] == pytest.approx(.1 + .99 * 2)
    assert sampler.scored_returns[199, 0] == pytest.approx(.1 + .99 * 8)
    expected = .1 + .99 * torch.tensor([2., 4., 8.])[lib.bin_of_frame(next_frames)]
    torch.testing.assert_close(sampler.scored_returns[:, 0], expected)


def test_every_nonterminal_successor_matches_previous_global_behavior_exactly():
    lib = _library()
    frames = torch.arange(lib.num_frames)
    frames = frames[frames < lib.last_frame_of(frames)]
    fixed = _sampler(lib, horizon=1, num_envs=frames.numel())
    original = _sampler(lib, horizon=1, num_envs=frames.numel(), clip_aware=False)
    _record(fixed, frames)
    _record(original, frames)
    for name in fixed._TENSORS:
        torch.testing.assert_close(getattr(fixed, name), getattr(original, name), rtol=0, atol=0)


def test_snapshot_marks_successor_policy_and_roundtrips_pending_window_exactly():
    original, resumed = _sampler(horizon=3), _sampler(horizon=3)
    _record(original, 99)
    state = original.state_dict()
    assert state["version"] == 2 and state["bootstrap_next_frame"] == "clip_clamped"
    resumed.load_state_dict(state)
    for frame in (99, 120, 121, 99):
        _record(original, frame)
        _record(resumed, frame)
    for name in original._TENSORS:
        torch.testing.assert_close(getattr(resumed, name), getattr(original, name), rtol=0, atol=0)


@pytest.mark.parametrize("corruption", ["missing", "global", "old_version", "bad_version"])
def test_incompatible_successor_snapshot_is_rejected_before_any_mutation(corruption):
    sampler = _sampler(horizon=3)
    _record(sampler, 99)
    before = sampler.state_dict()
    state = copy.deepcopy(before)
    if corruption == "missing":
        del state["bootstrap_next_frame"]
    elif corruption == "global":
        state["bootstrap_next_frame"] = "global_clamped"
    elif corruption == "old_version":
        state["version"] = 1
        del state["bootstrap_next_frame"]
    else:
        state["version"] = True
    state["value"].fill_(123)
    with pytest.raises(ValueError, match="bootstrap_next_frame|version"):
        sampler.load_state_dict(state)
    for name in sampler._TENSORS:
        torch.testing.assert_close(getattr(sampler, name), before[name], rtol=0, atol=0)


def test_previous_global_only_snapshot_remains_loadable_only_by_global_runtime():
    original = _sampler(horizon=3, clip_aware=False)
    _record(original, 99)
    state = original.state_dict()
    state["version"] = 1
    del state["bootstrap_next_frame"]
    restored = _sampler(horizon=3, clip_aware=False)
    restored.load_state_dict(state)
    torch.testing.assert_close(restored._frames, original._frames, rtol=0, atol=0)
    with pytest.raises(ValueError, match="bootstrap_next_frame"):
        _sampler(horizon=3).load_state_dict(state)


def test_command_contract_identifies_fixed_successor_and_rejects_old_statistics():
    lib, sampler = _library(), _sampler()
    cmd = _command(lib, sampler)
    cmd.cfg.reset_min_remaining_steps = 32
    contract = cmd._make_sampler_contract()
    assert contract["version"] == 6
    assert contract["bootstrap_next_frame"] == "clip_clamped"
    assert contract["failure_timeout_precedence"] == "timeout"
    assert contract["frame_bin_mapping"] == "clip_local_with_merged_tail"
