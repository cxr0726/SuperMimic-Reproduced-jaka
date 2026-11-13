"""CDF sampling preserves legal-row probabilities with fixed graph storage."""

import math

import pytest
import torch

from supermimic.data.replay_buffer import ReplayBuffer


def _sparse_buffer():
    buffer = ReplayBuffer(1, 16, 2, 1, 2, "cpu", sampling="cdf")
    buffer.valid[0, [1, 4, 10]] = True
    buffer.ptr.fill_(11)
    return buffer


def test_cdf_maps_all_uniform_legal_ranks_and_skips_plateaus(monkeypatch):
    buffer = _sparse_buffer()
    selection = buffer.prepare_starts(1)
    assert selection.cdf.dtype == torch.int64
    assert selection.count.item() == 3
    torch.testing.assert_close(selection.cdf, selection.weights.long().cumsum(0))

    def draws(n, *, dtype, device):
        assert n == 4 and dtype == torch.float64 and device == buffer.device
        # 1.0 exercises the upper-bound clamp defensively; real rand is <1.
        return torch.tensor([0., 1 / 3, 2 / 3, 1.], dtype=dtype, device=device)

    monkeypatch.setattr(torch, "rand", draws)
    env, rows = buffer._sample_from_selection(selection, 4)
    assert env.tolist() == [0, 0, 0, 0]
    assert rows.tolist() == [1, 4, 10, 10]


def test_cached_cdf_only_refreshes_on_new_replay_or_sampling_revision(monkeypatch):
    buffer = _sparse_buffer()
    original = torch.cumsum
    calls = []

    def cumsum(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "cumsum", cumsum)
    selection = buffer.prepare_starts(1)
    pointer = selection.cdf.data_ptr()
    assert len(calls) == 1
    for _ in range(3):
        assert buffer.prepare_starts(1) is selection
        buffer.sample_starts(8, 1, prepared=selection)
    assert len(calls) == 1

    buffer.valid[0, 2] = True
    buffer.prepare_starts(1)
    assert len(calls) == 2 and selection.count.item() == 4
    assert selection.cdf.data_ptr() == pointer

    # Configuration changes require new update graphs, but must not reuse a
    # stale preparation even if the replay itself has not changed.
    buffer.sampling = "multinomial"
    buffer.prepare_starts(1)
    assert len(calls) == 2
    buffer.sampling = "cdf"
    buffer.prepare_starts(1)
    assert len(calls) == 3 and selection.cdf.data_ptr() == pointer


def test_cdf_large_integer_ranks_do_not_lose_float32_tail_resolution(monkeypatch):
    buffer = ReplayBuffer(1, 16, 2, 1, 2, "cpu", sampling="cdf")
    selection = buffer._prepared_storage("arithmetic", ())
    total = 2**24 + 2
    # Synthetic cumulative counts exercise the rank arithmetic above float32's
    # exact-integer range without allocating a multi-gigabyte replay fixture.
    selection.cdf.fill_(total)
    selection.cdf[0] = total - 1
    selection.count.fill_(total)

    def draws(n, *, dtype, device):
        assert dtype == torch.float64
        return torch.tensor([(total - 1.75) / total, (total - .75) / total],
                            dtype=dtype, device=device)

    monkeypatch.setattr(torch, "rand", draws)
    _, rows = buffer._sample_from_selection(selection, 2)
    assert rows.tolist() == [0, 1]


def test_prepared_cdf_draws_need_no_host_tensor_reads_or_multinomial(monkeypatch):
    buffer = _sparse_buffer()
    selection = buffer.prepare_starts(1)

    def forbidden(*args, **kwargs):
        raise AssertionError("CDF draws must only execute device tensor operations")

    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(torch.Tensor, "__bool__", forbidden)
    monkeypatch.setattr(torch.Tensor, "tolist", forbidden)
    monkeypatch.setattr(torch, "multinomial", forbidden)
    sampled = buffer.sample_starts(32, 1, prepared=selection)
    monkeypatch.undo()
    assert sampled.state.shape == (32, 2)


def test_invalid_sampler_and_unrepresentable_rank_range_fail_before_allocation():
    with pytest.raises(ValueError, match="Replay sampling"):
        ReplayBuffer(1, 16, 2, 1, 2, "cpu", sampling="unknown")
    with pytest.raises(ValueError, match="2\\*\\*53"):
        ReplayBuffer(1, 2**53 + 1, 2, 1, 2, "cpu", sampling="cdf")


@pytest.mark.parametrize("n", [0, -1, True, 1.5])
def test_cdf_rejects_invalid_batch_size(n):
    buffer = _sparse_buffer()
    selection = buffer.prepare_starts(1)
    with pytest.raises(ValueError, match="positive integer"):
        buffer.sample_starts(n, 1, prepared=selection)


@pytest.mark.parametrize("envs, capacity", [(0, 16), (1, 0)])
def test_zero_size_replay_is_rejected_before_building_an_empty_cdf(envs, capacity):
    with pytest.raises(ValueError, match="must be positive"):
        ReplayBuffer(envs, capacity, 2, 1, 2, "cpu", sampling="cdf")


@pytest.mark.parametrize("allowed", [
    [0], [127], [0, 31, 64, 127], [1, 7, 31, 66, 95, 97, 112, 126],
])
def test_every_global_legal_rank_maps_once_including_empty_environments(monkeypatch, allowed):
    # Exhaustively enumerate legal ranks. Expected physical rows are supplied
    # independently, so leading/trailing plateaus and env boundaries cannot
    # silently receive probability mass.
    buffer = ReplayBuffer(4, 32, 1, 1, 1, "cpu", sampling="cdf")
    mask = torch.zeros(4, 32, dtype=torch.bool)
    mask.flatten()[allowed] = True
    selection = buffer._prepared_storage("rank_bijection", ())
    buffer._refresh_selection(selection, mask)

    def midpoint_ranks(n, *, dtype, device):
        assert n == len(allowed) and dtype == torch.float64
        return (torch.arange(n, dtype=dtype, device=device) + .5) / n

    monkeypatch.setattr(torch, "rand", midpoint_ranks)
    env, row = buffer._sample_from_selection(selection, len(allowed))
    assert (env * buffer.T + row).tolist() == allowed
    assert selection.count.item() == len(allowed)


def test_cdf_h32_mixed_clip_ring_can_become_empty_and_recover_in_place():
    from test_gpu_replay_sampling import _oracle_starts, _put

    buffer = ReplayBuffer(3, 48, 2, 1, 2, "cpu", sampling="cdf")
    # Two environments overwrite the ring; the middle clip is shorter than
    # H32 and must have no eligible reference start at all.
    bases, lengths = [0, 1000, 2000], [97, 29, 110]
    for env, (base, length) in enumerate(zip(bases, lengths, strict=True)):
        for serial in range(length):
            terminal = serial % (13 + env) == 12 or serial == length - 1
            _put(buffer, env, serial, terminal=terminal,
                 start=serial == 0 or (serial - 1) % (13 + env) == 12,
                 frame=base + serial, clip=env)
            if terminal:
                buffer.end_episode(torch.tensor([env]))
    ends = torch.tensor([base + length - 1 for base, length in zip(bases, lengths, strict=True)])
    selection = buffer.prepare_starts(5, horizon=32, clip_last=ends)
    pointers = (selection.weights.data_ptr(), selection.cdf.data_ptr(), selection.count.data_ptr())
    allowed = _oracle_starts(buffer, 5, horizon=32, ends=ends)
    assert allowed and all(index // buffer.T != 1 for index in allowed)
    assert set(selection.weights.nonzero().flatten().tolist()) == allowed
    assert selection.count.item() == len(allowed)
    env, row = buffer._sample_from_selection(selection, 2000)
    assert set((env * buffer.T + row).tolist()) <= allowed

    saved_ends = ends.clone()
    ends.copy_(torch.tensor(bases) + 31)
    with pytest.raises(RuntimeError, match="full reference horizon"):
        buffer.prepare_starts(5, horizon=32, clip_last=ends)
    assert selection.count.item() == 0 and selection.cdf[-1].item() == 0
    ends.copy_(saved_ends)
    assert buffer.prepare_starts(5, horizon=32, clip_last=ends) is selection
    assert pointers == (selection.weights.data_ptr(), selection.cdf.data_ptr(), selection.count.data_ptr())
    assert selection.count.item() == len(allowed)
    env, row = buffer._sample_from_selection(selection, 2000)
    assert set((env * buffer.T + row).tolist()) <= allowed


def test_cdf_and_multinomial_empirical_probabilities_match_same_sparse_global_pool():
    # Deliberately unequal counts per env (10, 0, 2, 5). The target is 1/17
    # for each listed position, independent of implementation and env count.
    allowed = torch.tensor([1, 2, 8, 30, 35, 40, 50, 55, 58, 61,
                            136, 178, 195, 198, 201, 237, 249])
    buffer = ReplayBuffer(4, 64, 1, 1, 1, "cpu")
    mask = torch.zeros(4, 64, dtype=torch.bool)
    mask.flatten()[allowed] = True
    selection = buffer._prepared_storage("distribution", ())
    n = 250_000
    counts_by_mode = []
    with torch.random.fork_rng(devices=[]):
        for seed, mode in ((179, "multinomial"), (823, "cdf")):
            buffer.sampling = mode
            buffer._refresh_selection(selection, mask)
            torch.manual_seed(seed)
            env, row = buffer._sample_from_selection(selection, n)
            counts = torch.bincount(env * buffer.T + row, minlength=buffer.E * buffer.T)
            assert counts.sum().item() == n
            assert counts[~mask.flatten()].sum().item() == 0
            counts_by_mode.append(counts[allowed].double())
    expected = n / allowed.numel()
    sigma = math.sqrt(expected * (1 - 1 / allowed.numel()))
    for counts in counts_by_mode:
        assert ((counts - expected).abs() < 6 * sigma).all()
    # Independent PRNG streams: compare distributions, not exact sampled rows.
    assert ((counts_by_mode[0] - counts_by_mode[1]).abs() < 6 * math.sqrt(2) * sigma).all()
