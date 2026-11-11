"""Official statistical invariants, adapted from HybridRobotics/supermimic.

Upstream 5efbf114, Apache-2.0. Additional cases verify framework adaptation.
"""

import math

import torch

from supermimic.envs.mdp.reset_sampling import (
    WindowValueSampler as ReturnSampler,
)
from supermimic.envs.mdp.reset_sampling import (
    WindowValueSamplerCfg as ReturnSamplerCfg,
)

# max_step_reward 1 with gamma 0.99 gives v_max = 100.


def _sampler(num_envs: int = 1, bin_count: int = 4, frame_count: int = 40, **cfg) -> ReturnSampler:
    return ReturnSampler(
        ReturnSamplerCfg(**cfg), bin_count, frame_count, num_envs, max_step_reward=1.0
    )


def _step(s: ReturnSampler, frame, reward, terminated=False, clip_end=False, timed_out=False):
    n = s._frames.shape[0]
    return s.record(
        torch.full((n,), frame),
        torch.full((n,), float(reward)),
        torch.full((n,), terminated),
        torch.full((n,), clip_end),
        torch.full((n,), timed_out),
    )


def test_unmeasured_bins_read_v_max_and_sampling_starts_uniform() -> None:
    s = _sampler(horizon=1)
    assert torch.equal(s.value, torch.full((4,), 100.0))
    w = _step(s, 25, 1.0)  # bin 2 is measured at v_max: nothing is hard yet
    assert torch.allclose(w, torch.full((4,), 0.25))


def test_record_returns_a_distribution_every_horizon_steps() -> None:
    s = _sampler(horizon=3)
    assert _step(s, 5, 1.0) is None
    assert _step(s, 6, 1.0) is None
    w = _step(s, 7, 1.0)
    assert w is not None and abs(float(w.sum()) - 1.0) < 1e-6


def test_a_fall_stops_after_its_own_reward() -> None:
    s = _sampler(horizon=1)
    _step(s, 5, 0.3, terminated=True)
    assert abs(float(s.value[0]) - 0.3) < 1e-5


def test_a_clip_end_bootstraps_from_the_mean_value() -> None:
    s = _sampler(horizon=1)
    s.value[:] = torch.tensor([20.0, 40.0, 60.0, 80.0])
    _step(s, 39, 0.3, clip_end=True)  # frame 39 is the clip's last, in bin 3
    assert abs(float(s.value[3]) - (0.3 + 0.99 * 50.0)) < 1e-4


def test_a_time_out_and_the_buffer_end_bootstrap_from_the_next_bin() -> None:
    for flags in ({"timed_out": True}, {}):
        s = _sampler(horizon=1)
        s.value[1] = 20.0
        _step(s, 9, 0.3, **flags)  # frame 10 is in bin 1
        assert abs(float(s.value[0]) - (0.3 + 0.99 * 20.0)) < 1e-4


def test_returns_discount_back_from_a_fall() -> None:
    s = _sampler(horizon=3)
    _step(s, 0, 1.0)
    _step(s, 1, 1.0)
    _step(s, 2, 1.0, terminated=True)
    # Frames 0-2 lie in bin 0 of one episode: returns 1 + .99 (1 + .99), 1 + .99 and 1
    # average into one sample.
    expected = (1 + 0.99 * (1 + 0.99) + (1 + 0.99) + 1) / 3
    assert abs(float(s.value[0]) - expected) < 1e-5


def test_episodes_split_samples() -> None:
    s = _sampler(horizon=2)
    _step(s, 0, 1.0, terminated=True)  # episode one ends in bin 0
    _step(s, 1, 0.5, terminated=True)  # episode two, same bin: a second sample
    assert float(s.count[0]) == 2.0
    assert abs(float(s.value[0]) - 0.75) < 1e-6


def test_value_is_an_exact_mean_up_to_memory_then_an_average_over_memory() -> None:
    s = _sampler(horizon=1, memory=4)
    for reward in (1.0, 2.0, 3.0, 4.0):
        _step(s, 5, reward, terminated=True)
    assert abs(float(s.value[0]) - 2.5) < 1e-6
    _step(s, 5, 9.0, terminated=True)
    assert abs(float(s.value[0]) - (2.5 + (9.0 - 2.5) / 5)) < 1e-6  # weight 1/(memory+1)
    _step(s, 5, 9.0, terminated=True)
    assert float(s.count[0]) == 5.0  # earlier samples count at most `memory` times


def test_focus_concentrates_on_low_return() -> None:
    s = _sampler(horizon=1, explore=0.0, keep=0.0)
    s.value[1] = 70.0
    w = _step(s, 25, 1.0)  # bin 2 measured at v_max
    assert abs(float(w[1] / w[0]) - math.exp(25 * 0.3)) / math.exp(7.5) < 1e-3


def test_keep_holds_a_solved_hard_part_of_a_long_clip() -> None:
    # One clip of 64 bins whose only hard part is bins 10-11.
    s = _sampler(horizon=1, bin_count=64, frame_count=640, keep=0.1)
    s.value[10:12] = 0.0
    _step(s, 600, 1.0)  # the focus takes bins 10-11 and earns keep
    s.value[10:12] = 100.0
    w = _step(s, 600, 1.0)
    explore_and_focus = 0.5 / 64 + 0.4 / 64
    assert abs(float(w.sum()) - 1.0) < 1e-6
    assert float(w[10:12].sum()) - 2 * explore_and_focus > 0.09  # the keep stays there
    assert abs(float(w[30]) - explore_and_focus) < 1e-4


def test_focus_spread_thin_earns_almost_no_keep() -> None:
    s = _sampler(horizon=1, bin_count=1000, frame_count=10000, keep=0.1)
    s.value[7] = 0.0
    _step(s, 9000, 1.0)
    s.value[7] = 100.0
    for _ in range(50):  # then 50 updates of uniform focus
        w = _step(s, 9000, 1.0)
    # Each uniform update earns 1000 * (1/1000)**2 = 1/1000 in total, against about 1
    # for bin 7.
    assert float(s.effort.sum() - s.effort[7]) < 0.06
    assert float(w[7]) - (0.9 / 1000) > 0.09


def test_golden_stream() -> None:
    # Locks the full record -> update -> distribution pipeline on a fixed stream.
    s = _sampler(num_envs=4, bin_count=16, frame_count=160)
    gen = torch.Generator().manual_seed(99)
    weights = None
    for _ in range(60):
        frames = torch.randint(0, 160, (4,), generator=gen)
        rewards = torch.rand(4, generator=gen)
        terminated = torch.rand(4, generator=gen) < 0.15
        clip_end = (torch.rand(4, generator=gen) < 0.05) & ~terminated
        timed_out = (torch.rand(4, generator=gen) < 0.05) & ~terminated & ~clip_end
        out = s.record(frames, rewards, terminated, clip_end, timed_out)
        if out is not None:
            weights = out
    assert weights is not None
    assert torch.allclose(weights, torch.tensor(GOLDEN), atol=1e-6)


GOLDEN = [
    0.03130187,
    0.03134733,
    0.03201599,
    0.03608831,
    0.04007206,
    0.31132525,
    0.03760373,
    0.03675647,
    0.07033682,
    0.0331141,
    0.03324675,
    0.05272575,
    0.05815605,
    0.03189738,
    0.0314665,
    0.13254566,
]


# Framework-specific integration and resume invariants.


def test_count_weight_depends_on_number_of_independent_env_samples():
    for n in (1, 18, 498):
        s = _sampler(num_envs=n, horizon=1)
        s.value[0], s.count[0] = 50.0, 256.0
        _step(s, 0, 80.0, terminated=True)
        expected = 50 + (80 - 50) * n / (256 + n)
        torch.testing.assert_close(s.value[0], torch.tensor(expected))


def test_each_visited_bin_gets_updated_and_dedup_is_window_local():
    s = _sampler(horizon=2)
    _step(s, 9, 0.5)
    _step(s, 10, 0.5)
    assert s.count.tolist() == [1, 1, 0, 0]
    _step(s, 11, 0.5)
    _step(s, 12, 0.5)
    assert s.count.tolist() == [1, 2, 0, 0]
    assert s.num_updates == 2


def test_eligibility_only_restricts_reset_distribution_not_value_learning():
    s = ReturnSampler(
        ReturnSamplerCfg(horizon=1),
        4,
        40,
        2,
        max_step_reward=1,
        sampleable_bins=torch.tensor([True, True, False, False]),
    )
    _step(s, 25, 0.25, terminated=True)
    assert s.count[2] == 2 and s.value[2] == 0.25
    assert s.probabilities[2:].count_nonzero() == 0
    assert s.effort[2:].count_nonzero() == 0
    torch.testing.assert_close(s.probabilities[:2], torch.tensor([0.5, 0.5]))
    # clip_end still bootstraps from ALL values, including unresettable bins.
    s.value[:] = torch.tensor([20.0, 40.0, 60.0, 80.0])
    _step(s, 39, 0.3, clip_end=True)
    torch.testing.assert_close(s.value[3], torch.tensor(0.3 + 0.99 * 50))


def test_frame_mapping_callback_preserves_clip_local_geometry():
    # The second clip starts at frame 7, unlike global equal-width bins.
    def mapping(frames):
        return torch.where(frames < 7, 0, torch.where(frames < 17, 1, 2))

    s = ReturnSampler(ReturnSamplerCfg(horizon=1), 3, 21, 1, 1, bin_of_frame=mapping)
    _step(s, 7, 0.5, terminated=True)
    assert s.count.tolist() == [0, 1, 0]


def test_state_roundtrip_keeps_partial_window_and_continues_exactly():
    s = _sampler(num_envs=3, horizon=3)
    for t in range(4):
        _step(s, t, 0.2 + 0.01 * t)
    state = s.state_dict()
    assert state["step"] == 1
    resumed = _sampler(num_envs=3, horizon=3)
    resumed.load_state_dict(state)
    pointers = {name: getattr(resumed, name).data_ptr() for name in resumed._TENSORS}
    for t in range(4, 9):
        _step(s, t, 0.2 + 0.01 * t)
        _step(resumed, t, 0.2 + 0.01 * t)
    for name in s._TENSORS:
        torch.testing.assert_close(getattr(resumed, name), getattr(s, name), rtol=0, atol=0)
        assert pointers[name] == getattr(resumed, name).data_ptr()
    assert resumed.num_updates == s.num_updates


def test_invalid_state_rejected_before_any_buffer_changes():
    import copy

    import pytest

    s = _sampler(horizon=2)
    _step(s, 0, 0.3)
    before = s.state_dict()
    cases = []
    for key, val in (("step", 2), ("num_updates", -1), ("num_envs", 3)):
        state = copy.deepcopy(before)
        state[key] = val
        cases.append(state)
    for key, val in (("value", float("nan")), ("count", -1), ("effort", -1)):
        state = copy.deepcopy(before)
        state[key][0] = val
        cases.append(state)
    state = copy.deepcopy(before)
    state["_frames"][0, 0] = 1000000
    cases.append(state)
    for state in cases:
        with pytest.raises(ValueError):
            s.load_state_dict(state)
        for name in s._TENSORS:
            torch.testing.assert_close(getattr(s, name), before[name], rtol=0, atol=0)
        assert s._step == before["step"]


def test_discard_pending_window_preserves_learned_curriculum():
    s = _sampler(horizon=2)
    for t in range(3):
        _step(s, t, 0.5)
    before = s.state_dict()
    s.discard_pending_window()
    assert s._step == 0 and s._frames.count_nonzero() == 0
    for name in ("value", "count", "effort", "probabilities"):
        torch.testing.assert_close(getattr(s, name), before[name], rtol=0, atol=0)
    assert s.num_updates == before["num_updates"]


def test_first_cost_is_not_extrapolated_as_infinite_mean_reward():
    s = ReturnSampler(ReturnSamplerCfg(horizon=1), 4, 40, 1, max_step_reward=0.112)
    _step(s, 0, 0.012, timed_out=True)
    torch.testing.assert_close(s.value[0], torch.tensor(11.1))
