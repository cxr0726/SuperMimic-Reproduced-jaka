"""GPU ring replay buffer laid out as [env, time].

Each row holds the state ``s_t`` at the start of a control step together with the
transition that followed it: clean action ``a_t``, executed action ``a_exec_t``, sim
context ``xi_t``, reward ``r_{t+1}``, the target frame/clip of that step and bookkeeping
flags. When an episode ends an extra *terminal* row is written that holds ``s_{T+1}``
(no transition out of it) and the termination label of that state, so k-step windows
can use it as their last target state and the termination head gets positive labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class WindowBatch:
    states: torch.Tensor  # (k+1, n, S)
    a_exec: torch.Tensor  # (k, n, A)
    a_clean: torch.Tensor  # (k, n, A)
    xi: torch.Tensor  # (n, X) context at the window start (constant inside an episode)
    reward: torch.Tensor  # (k, n)
    done: torch.Tensor  # (k, n) termination label of states 1..k
    frame: torch.Tensor  # (k, n) target frame of each transition


@dataclass
class StartBatch:
    state: torch.Tensor  # (n, S)
    xi: torch.Tensor  # (n, X)
    frame: torch.Tensor  # (n,) current target frame
    prev_action: torch.Tensor  # (n, A) clean action of the previous step (0 at episode start)
    hist_states: torch.Tensor  # (hist, n, S) states s_{t-hist+1..t} (backfilled at episode start)
    hist_actions: torch.Tensor  # (hist, n, A) clean actions a_{t-hist..t-1} aligned with hist_states' obs
    history_valid: torch.Tensor | None = None  # (hist,n); False denotes reset backfill


@dataclass
class PreparedReplaySelection:
    """Fixed-storage legal-row weights for one kind of replay sampling.

    Preparation belongs outside a captured update. Repeating preparation while
    replay is unchanged is free of device-to-host reads. After collection the
    caller refreshes the *same* storage before replaying its CUDA graph. Sampling
    itself contains only fixed-shape GPU operations and fresh multinomial RNG.
    """

    owner: int
    kind: str
    key: tuple
    weights: torch.Tensor  # flattened (E*T,), 1 for each legal start
    count: torch.Tensor  # scalar long, kept on the replay device
    dependency: torch.Tensor | None = None
    revision: tuple | None = None


class ReplayBuffer:
    def __init__(self, num_envs: int, capacity: int, state_dim: int, act_dim: int, xi_dim: int, device):
        self.E, self.T = num_envs, capacity
        self.device = torch.device(device)
        E, T = num_envs, capacity
        f = lambda *shape: torch.zeros(E, T, *shape, device=self.device)
        self.s = f(state_dim)
        self.a = f(act_dim)
        self.a_exec = f(act_dim)
        self.xi = f(xi_dim)
        self.r = f()
        self.done = f()
        self.frame = torch.zeros(E, T, dtype=torch.long, device=self.device)
        self.clip = torch.zeros(E, T, dtype=torch.long, device=self.device)
        self.ep_id = torch.full((E, T), -1, dtype=torch.long, device=self.device)
        self.terminal = torch.zeros(E, T, dtype=torch.bool, device=self.device)
        self.ep_start = torch.zeros(E, T, dtype=torch.bool, device=self.device)
        self.valid = torch.zeros(E, T, dtype=torch.bool, device=self.device)
        self.ptr = torch.zeros(E, dtype=torch.long, device=self.device)
        self.ep_counter = torch.zeros(E, dtype=torch.long, device=self.device)
        self.total_rows = 0
        self._env_ar = torch.arange(E, device=self.device)
        self._write_revision = 0
        self._prepared: dict[tuple, PreparedReplaySelection] = {}

    # ------------------------------------------------------------------ writes

    @torch.no_grad()
    def add(self, env_ids, s, a, a_exec, xi, r, done, frame, clip, terminal, ep_start) -> None:
        p = self.ptr[env_ids]
        self.s[env_ids, p] = s
        self.a[env_ids, p] = a
        self.a_exec[env_ids, p] = a_exec
        self.xi[env_ids, p] = xi
        self.r[env_ids, p] = r
        self.done[env_ids, p] = done
        self.frame[env_ids, p] = frame
        self.clip[env_ids, p] = clip
        self.ep_id[env_ids, p] = self.ep_counter[env_ids]
        self.terminal[env_ids, p] = terminal
        self.ep_start[env_ids, p] = ep_start
        self.valid[env_ids, p] = True
        self.ptr[env_ids] = (p + 1) % self.T
        self.total_rows += int(env_ids.numel())
        self._write_revision += 1

    @torch.no_grad()
    def end_episode(self, env_ids) -> None:
        self.ep_counter[env_ids] += 1
        self._write_revision += 1

    @property
    def size(self) -> int:
        return int(self.valid.sum().item())

    # ------------------------------------------------------------------ sampling helpers

    def _rolled(self, x: torch.Tensor, j: int) -> torch.Tensor:
        """x[:, (t + j) % T] for every t."""
        return torch.roll(x, shifts=-j, dims=1) if j else x

    def _selection_revision(self, *, starts: bool, dependency=None) -> tuple:
        # Tensor versions also catch explicit bookkeeping edits by diagnostics
        # and tests; reading these CPU metadata counters never synchronizes CUDA.
        tensors = (self.valid, self.terminal, self.ptr)
        tensors += (self.frame, self.clip) if starts else (self.ep_id,)
        return (self._write_revision, *(x._version for x in tensors),
                None if dependency is None else dependency._version)

    def _prepared_storage(self, kind: str, key: tuple, dependency=None) -> PreparedReplaySelection:
        cache_key = (kind, *key)
        if cache_key not in self._prepared:
            self._prepared[cache_key] = PreparedReplaySelection(
                owner=id(self), kind=kind, key=key,
                weights=torch.zeros(self.E * self.T, device=self.device),
                count=torch.zeros((), dtype=torch.long, device=self.device), dependency=dependency,
            )
        return self._prepared[cache_key]

    @staticmethod
    def _preflight_selection(selection: PreparedReplaySelection, message: str) -> None:
        # One explicit block-boundary check, never inside a captured update.
        # Failing here avoids multinomial's device assertion poisoning CUDA.
        if int(selection.count.item()) == 0:
            raise RuntimeError(message)

    @torch.no_grad()
    def prepare_windows(self, k: int) -> PreparedReplaySelection:
        """Refresh cached legal k-transition starts before a WM update block.

        All legality computation stays on the replay device. Only a changed
        replay revision performs the empty-set preflight; the ten WM batches
        can share the resulting fixed-storage selection without rescanning it.
        """
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("window length must be a positive integer")
        selection = self._prepared_storage("windows", (k,))
        revision = self._selection_revision(starts=False)
        if selection.revision != revision:
            mask = self.window_mask(k)
            selection.weights.copy_(mask.reshape(-1))
            selection.count.copy_(mask.sum())
            self._preflight_selection(selection, "replay buffer has no sampleable rows yet")
            selection.revision = revision
        return selection

    @staticmethod
    def _validate_start_arguments(hist: int, horizon, clip_last, device) -> None:
        if isinstance(hist, bool) or not isinstance(hist, int) or hist < 1:
            raise ValueError("history length must be a positive integer")
        if (horizon is None) != (clip_last is None):
            raise ValueError("horizon and clip_last must be provided together")
        if horizon is not None and (isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1):
            raise ValueError("imagination horizon must be positive")
        if clip_last is not None and (
            clip_last.ndim != 1 or clip_last.numel() == 0 or clip_last.device != device
        ):
            raise ValueError("clip_last must be a nonempty vector on the replay device")

    @torch.no_grad()
    def start_mask(self, hist: int, *, horizon=None, clip_last=None) -> torch.Tensor:
        """Legal policy starts; future reference availability is not replay length."""
        self._validate_start_arguments(hist, horizon, clip_last, self.frame.device)
        mask = self.valid & ~self.terminal
        t_idx = torch.arange(self.T, device=self.device)[None, :]
        dist_to_ptr = (self.ptr[:, None] - t_idx - 1) % self.T
        mask &= dist_to_ptr < self.T - hist - 1
        if horizon is not None:
            mask &= self.frame + horizon <= clip_last[self.clip]
        return mask

    @torch.no_grad()
    def prepare_starts(self, hist: int, *, horizon=None, clip_last=None) -> PreparedReplaySelection:
        """Refresh H-valid policy starts before an Actor/Critic update block.

        Call this again after physical collection, before replaying a graph
        that closes over this selection. Its weights keep the same address.
        """
        self._validate_start_arguments(hist, horizon, clip_last, self.frame.device)
        key = (hist, horizon, None if clip_last is None else id(clip_last))
        selection = self._prepared_storage("starts", key, clip_last)
        revision = self._selection_revision(starts=True, dependency=clip_last)
        if selection.revision != revision:
            mask = self.start_mask(hist, horizon=horizon, clip_last=clip_last)
            selection.weights.copy_(mask.reshape(-1))
            selection.count.copy_(mask.sum())
            message = ("replay buffer has no sampleable starts with a full reference horizon and valid history"
                       if horizon is not None else "replay buffer has no sampleable rows yet")
            self._preflight_selection(selection, message)
            selection.revision = revision
        return selection

    def _check_prepared(self, selection: PreparedReplaySelection, kind: str, key: tuple) -> None:
        if selection.owner != id(self) or selection.kind != kind or selection.key != key:
            raise ValueError("prepared replay selection belongs to a different buffer or sampling configuration")

    @torch.no_grad()
    def window_mask(self, k: int) -> torch.Tensor:
        """(E,T) True where rows t..t+k are valid, same episode, and t..t+k-1 are non-terminal."""
        ok = self.valid & ~self.terminal
        ep0 = self.ep_id
        for j in range(1, k + 1):
            v = self._rolled(self.valid, j) & (self._rolled(ep0, j) == ep0)
            if j < k:
                v &= ~self._rolled(self.terminal, j)
            ok &= v
        # The row currently under the write pointer may be overwritten next; the rolled window must not
        # wrap across the write pointer. Rows t with t + k crossing ptr are excluded by the ep_id test
        # (new rows carry a newer ep_id) except for same-episode wraps, which are rare; exclude explicitly.
        t = torch.arange(self.T, device=self.device)[None, :]
        dist_to_ptr = (self.ptr[:, None] - t - 1) % self.T  # rows strictly before ptr: 0 = newest
        ok &= dist_to_ptr >= k
        return ok

    @torch.no_grad()
    def _sample_from_mask(self, mask: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Empty selections have already failed at the preparation boundary.
        # In particular, no Tensor -> Python conversion is allowed here.
        idx = torch.multinomial(mask.reshape(-1).float(), n, replacement=True)
        return idx // self.T, idx % self.T

    @torch.no_grad()
    def sample_windows(
        self, n: int, k: int, *, prepared: PreparedReplaySelection | None = None,
    ) -> WindowBatch:
        if prepared is None:
            prepared = self.prepare_windows(k)
        self._check_prepared(prepared, "windows", (k,))
        e, t = self._sample_from_mask(prepared.weights, n)
        offs = torch.arange(k + 1, device=self.device)
        tt = (t[None, :] + offs[:, None]) % self.T  # (k+1, n)
        ee = e[None, :].expand(k + 1, n)
        return WindowBatch(
            states=self.s[ee, tt],
            a_exec=self.a_exec[ee[:-1], tt[:-1]],
            a_clean=self.a[ee[:-1], tt[:-1]],
            xi=self.xi[e, t],
            reward=self.r[ee[:-1], tt[:-1]],
            done=self.done[ee[1:], tt[1:]],
            frame=self.frame[ee[:-1], tt[:-1]],
        )

    @torch.no_grad()
    def sample_starts(
        self, n: int, hist: int, *, horizon: int | None = None,
        clip_last: torch.Tensor | None = None,
        prepared: PreparedReplaySelection | None = None,
    ) -> StartBatch:
        """Sample logged states, optionally requiring a full reference horizon.

        ``clip_last[c]`` is the inclusive global final frame of clip ``c``.
        The stored clip IDs avoid a large per-reference-frame lookup table.
        Eligibility concerns reference
        availability, not future replay rows: imagination may continue beyond
        a real collection timeout while the reference still has enough frames.
        """
        if prepared is None:
            prepared = self.prepare_starts(hist, horizon=horizon, clip_last=clip_last)
        self._check_prepared(prepared, "starts", (hist, horizon, None if clip_last is None else id(clip_last)))
        e, t = self._sample_from_mask(prepared.weights, n)
        offs = torch.arange(hist, device=self.device) - (hist - 1)  # -(hist-1) .. 0
        tt = (t[None, :] + offs[:, None]) % self.T  # (hist, n)
        ee = e[None, :].expand(hist, n)
        same_ep = (self.ep_id[ee, tt] == self.ep_id[e, t][None, :]) & self.valid[ee, tt]
        # Backfill: for rows outside the episode, use the first in-episode row of the window.
        first_ok = same_ep.float().argmax(dim=0)  # first True index per column (same_ep[-1] is always True)
        pos = torch.arange(hist, device=self.device)[:, None].expand(hist, n)
        tt_fixed = torch.where(same_ep, tt, tt[first_ok, torch.arange(n, device=self.device)][None, :].expand(hist, n))
        hist_states = self.s[ee, tt_fixed]
        # Actions preceding each history state: a at row-1 if same episode else 0.
        tprev = (tt_fixed - 1) % self.T
        prev_ok = (self.ep_id[ee, tprev] == self.ep_id[e, t][None, :]) & self.valid[ee, tprev] & ~self.ep_start[ee, tt_fixed]
        hist_actions = torch.where(prev_ok[..., None], self.a[ee, tprev], torch.zeros_like(self.a[ee, tprev]))
        del pos
        return StartBatch(
            state=self.s[e, t],
            xi=self.xi[e, t],
            frame=self.frame[e, t],
            prev_action=hist_actions[-1],
            hist_states=hist_states,
            hist_actions=hist_actions,
            history_valid=same_ep,
        )
