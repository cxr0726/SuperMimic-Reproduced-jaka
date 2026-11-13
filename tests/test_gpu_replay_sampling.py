"""Prepared GPU sampling retains the replay distribution and ring boundaries."""

import math

import pytest
import torch

from supermimic.data.replay_buffer import ReplayBuffer


def _put(buffer, env, serial, *, terminal=False, start=False, frame=None, clip=0):
    device = buffer.device
    ids = torch.tensor([env], device=device)
    value = float(1000 * env + serial)
    buffer.add(
        ids, torch.tensor([[value, float(env)]], device=device),
        torch.tensor([[value + .25]], device=device), torch.tensor([[value + .5]], device=device),
        torch.tensor([[float(env), .1]], device=device), torch.tensor([.1], device=device),
        torch.tensor([float(terminal)], device=device),
        torch.tensor([serial if frame is None else frame], device=device),
        torch.tensor([clip], device=device), torch.tensor([terminal], device=device),
        torch.tensor([start], device=device),
    )


def _ring(device="cpu"):
    buffer = ReplayBuffer(4, 16, 2, 1, 2, device)
    for env in range(4):
        start = True
        for serial in range(25 + 2 * env):
            terminal = (serial + 2 * env) % (7 + env) == 6
            _put(buffer, env, serial, terminal=terminal, start=start)
            start = terminal
            if terminal:
                buffer.end_episode(torch.tensor([env], device=device))
    return buffer


def _oracle_windows(buffer, k):
    v, terminal, ep = (x.cpu().tolist() for x in (buffer.valid, buffer.terminal, buffer.ep_id))
    ptr = buffer.ptr.cpu().tolist()
    allowed = set()
    for env in range(buffer.E):
        for row in range(buffer.T):
            if not v[env][row] or terminal[env][row] or (ptr[env] - row - 1) % buffer.T < k:
                continue
            legal = True
            for step in range(1, k + 1):
                following = (row + step) % buffer.T
                legal &= v[env][following] and ep[env][following] == ep[env][row]
                if step < k:
                    legal &= not terminal[env][following]
            if legal:
                allowed.add(env * buffer.T + row)
    return allowed


def _oracle_starts(buffer, hist, *, horizon=None, ends=None):
    v, terminal, frames, clips = (x.cpu().tolist() for x in
                                  (buffer.valid, buffer.terminal, buffer.frame, buffer.clip))
    ptr = buffer.ptr.cpu().tolist()
    ends = None if ends is None else ends.cpu().tolist()
    return {
        env * buffer.T + row
        for env in range(buffer.E) for row in range(buffer.T)
        if v[env][row] and not terminal[env][row]
        and (ptr[env] - row - 1) % buffer.T < buffer.T - hist - 1
        and (horizon is None or frames[env][row] + horizon <= ends[clips[env][row]])
    }


def test_prepared_masks_equal_independent_ring_oracles_and_storage_is_stable(monkeypatch):
    buffer = _ring()
    ends = torch.tensor([32])
    windows = buffer.prepare_windows(3)
    starts = buffer.prepare_starts(4, horizon=4, clip_last=ends)
    assert set(windows.weights.nonzero().flatten().tolist()) == _oracle_windows(buffer, 3)
    assert set(starts.weights.nonzero().flatten().tolist()) == _oracle_starts(buffer, 4, horizon=4, ends=ends)
    old_weights = windows.weights.clone()
    addresses = windows.weights.data_ptr(), starts.weights.data_ptr()
    calls = []
    original = buffer._preflight_selection
    monkeypatch.setattr(buffer, "_preflight_selection", lambda *a: (calls.append(1), original(*a))[-1])
    assert buffer.prepare_windows(3) is windows
    assert buffer.prepare_starts(4, horizon=4, clip_last=ends) is starts
    assert not calls, "unchanged replay must not redo the host preflight"
    _put(buffer, 0, 99)
    assert buffer.prepare_windows(3) is windows
    assert not torch.equal(windows.weights, old_weights)
    assert buffer.prepare_starts(4, horizon=4, clip_last=ends) is starts
    assert addresses == (windows.weights.data_ptr(), starts.weights.data_ptr())
    assert len(calls) == 2
    assert set(windows.weights.nonzero().flatten().tolist()) == _oracle_windows(buffer, 3)


def test_horizon_dependency_and_manual_bookkeeping_changes_invalidate_selection():
    buffer = ReplayBuffer(1, 16, 2, 1, 2, "cpu")
    for serial in range(8):
        _put(buffer, 0, serial, start=serial == 0)
    ends = torch.tensor([9])
    selection = buffer.prepare_starts(2, horizon=4, clip_last=ends)
    address = selection.weights.data_ptr()
    assert set(selection.weights.nonzero().flatten().tolist()) == set(range(6))
    ends.fill_(5)
    assert buffer.prepare_starts(2, horizon=4, clip_last=ends) is selection
    assert set(selection.weights.nonzero().flatten().tolist()) == {0, 1}
    buffer.terminal[0, 0] = True
    assert buffer.prepare_starts(2, horizon=4, clip_last=ends) is selection
    assert set(selection.weights.nonzero().flatten().tolist()) == {1}
    assert selection.weights.data_ptr() == address
    other_ends = torch.tensor([9])
    assert buffer.prepare_starts(2, horizon=4, clip_last=other_ends) is not selection
    assert buffer.prepare_starts(3, horizon=4, clip_last=ends) is not selection


def test_uniformity_is_over_legal_env_time_pairs_not_over_environments():
    torch.manual_seed(17)
    buffer = ReplayBuffer(2, 32, 2, 1, 2, "cpu")
    for env, count in ((0, 18), (1, 5)):
        for serial in range(count):
            _put(buffer, env, serial, start=serial == 0)
    selection = buffer.prepare_windows(2)
    allowed = sorted(_oracle_windows(buffer, 2))
    env, row = buffer._sample_from_mask(selection.weights, 100_000)
    indices = env * buffer.T + row
    counts = torch.bincount(indices, minlength=buffer.E * buffer.T)
    assert set(indices.unique().tolist()) == set(allowed)
    expected = indices.numel() / len(allowed)
    sigma = math.sqrt(expected * (1 - 1 / len(allowed)))
    assert ((counts[allowed].float() - expected).abs() < 6 * sigma).all()
    expected_env0 = sum(i < buffer.T for i in allowed) / len(allowed)
    assert abs(float((env == 0).float().mean()) - expected_env0) < .01
    assert expected_env0 > .8  # An env-uniform regression would put this at .5.


def test_prepared_start_history_backfill_and_previous_action_match_ring_contents():
    torch.manual_seed(4)
    buffer = _ring()
    selection = buffer.prepare_starts(5, horizon=4, clip_last=torch.tensor([100]))
    # Keep the exact same dependency object when consuming this prepared mask.
    starts = buffer.sample_starts(1024, 5, horizon=4, clip_last=selection.dependency, prepared=selection)
    lookup = {int(buffer.s[e, t, 0]): (e, t) for e in range(buffer.E)
              for t in range(buffer.T) if buffer.valid[e, t]}
    for column in range(starts.state.shape[0]):
        env, row = lookup[int(starts.state[column, 0])]
        rows = [(row - 4 + step) % buffer.T for step in range(5)]
        valid = [bool(buffer.valid[env, t] and buffer.ep_id[env, t] == buffer.ep_id[env, row]) for t in rows]
        first = rows[valid.index(True)]
        fixed = [t if ok else first for t, ok in zip(rows, valid, strict=True)]
        assert starts.history_valid[:, column].tolist() == valid
        torch.testing.assert_close(starts.hist_states[:, column], buffer.s[env, fixed])
        expected = []
        for t in fixed:
            previous = (t - 1) % buffer.T
            same = (buffer.valid[env, previous] and buffer.ep_id[env, previous] == buffer.ep_id[env, t]
                    and not buffer.ep_start[env, t])
            expected.append(buffer.a[env, previous] if same else torch.zeros(1))
        torch.testing.assert_close(starts.hist_actions[:, column], torch.stack(expected))
        torch.testing.assert_close(starts.prev_action[column], expected[-1])


def test_empty_selection_fails_at_preflight_and_wrong_selection_is_rejected():
    buffer = ReplayBuffer(1, 16, 2, 1, 2, "cpu")
    with pytest.raises(RuntimeError, match="no sampleable rows"):
        buffer.prepare_windows(2)
    with pytest.raises(RuntimeError, match="full reference horizon"):
        buffer.prepare_starts(2, horizon=3, clip_last=torch.tensor([10]))
    for serial in range(8):
        _put(buffer, 0, serial, start=serial == 0)
    selection = buffer.prepare_windows(2)
    with pytest.raises(ValueError, match="different buffer or sampling"):
        buffer.sample_windows(4, 3, prepared=selection)
    other = ReplayBuffer(1, 16, 2, 1, 2, "cpu")
    with pytest.raises(ValueError, match="different buffer or sampling"):
        other.sample_windows(4, 2, prepared=selection)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_graph_sampling_is_fresh_legal_and_refreshes_after_writes(monkeypatch):
    buffer = _ring("cuda:0")
    ends = torch.tensor([100], device="cuda:0")
    windows = buffer.prepare_windows(3)
    starts = buffer.prepare_starts(4, horizon=4, clip_last=ends)
    addresses = windows.weights.data_ptr(), starts.weights.data_ptr()

    def sample():
        w = buffer.sample_windows(128, 3, prepared=windows)
        s = buffer.sample_starts(128, 4, horizon=4, clip_last=ends, prepared=starts)
        return w.states, s.state, s.frame

    def no_host_preflight(*args):
        raise AssertionError("captured sampling must not run an empty-set host check")

    original_preflight = buffer._preflight_selection
    monkeypatch.setattr(buffer, "_preflight_selection", no_host_preflight)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            sample()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sampled_windows, sampled_starts, sampled_frames = sample()
    graph.replay()
    previous = sampled_starts.clone()
    graph.replay()
    assert not torch.equal(previous, sampled_starts)
    assert (sampled_frames + 4 <= ends[0]).all()
    monkeypatch.setattr(buffer, "_preflight_selection", original_preflight)

    for env in range(4):
        for serial in range(40, 48):
            _put(buffer, env, serial, start=serial == 40)
    assert buffer.prepare_windows(3) is windows
    assert buffer.prepare_starts(4, horizon=4, clip_last=ends) is starts
    assert addresses == (windows.weights.data_ptr(), starts.weights.data_ptr())
    graph.replay()
    legal_windows = {float(buffer.s[i // buffer.T, i % buffer.T, 0]) for i in _oracle_windows(buffer, 3)}
    legal_starts = {float(buffer.s[i // buffer.T, i % buffer.T, 0]) for i in
                    _oracle_starts(buffer, 4, horizon=4, ends=ends)}
    assert set(sampled_windows[0, :, 0].tolist()) <= legal_windows
    assert set(sampled_starts[:, 0].tolist()) <= legal_starts

    # A future empty buffer is rejected before graph replay and leaves the CUDA
    # context healthy, unlike relying on multinomial's device-side assertion.
    buffer.valid.zero_()
    with pytest.raises(RuntimeError, match="no sampleable rows"):
        buffer.prepare_windows(3)
    for env in range(4):
        for serial in range(60, 68):
            _put(buffer, env, serial, start=serial == 60)
    buffer.prepare_windows(3)
    buffer.prepare_starts(4, horizon=4, clip_last=ends)
    graph.replay()
    assert torch.isfinite(sampled_windows).all() and torch.isfinite(sampled_starts).all()
    torch.cuda.synchronize()
