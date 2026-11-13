"""Capture complete learner updates without adding uncounted optimizer steps.

Warm-up calls are ordinary, counted training updates. Recording a CUDA graph
does not execute its kernels; its first replay is that call's single update.
Physical collection and reset remain outside these graphs.
"""

from __future__ import annotations

import gc
import math
from collections.abc import Callable, Iterable
from contextlib import contextmanager

import torch
from torch import nn

_UNSET_CONFIGURATION = object()


@contextmanager
def _recording_gc_guard():
    """Release old cycles before recording, never from inside a CUDA capture.

    Trainers own bound update methods and graph pools, so abandoned trainers
    can form Python cycles. Their CUDA resources must not be finalized by an
    automatic collection while another graph is being recorded. Replay and
    ordinary counted warm-up do not enter this scope.
    """
    enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()
        else:
            gc.disable()


def replay_sampling_key(buffer) -> tuple | None:
    """Host-only dispatch values that a recorded replay sampler specializes.

    Tensor contents (including the refreshed CDF) deliberately do not belong in
    this key: their addresses are stable and graphs must read their new values.
    Legacy diagnostic buffers without a sampler setting keep their old path.
    """
    if buffer is None:
        return None
    sampling = getattr(buffer, "sampling", "multinomial")
    if not isinstance(sampling, str) or sampling not in ("multinomial", "cdf"):
        raise ValueError("Replay sampling must be 'multinomial' or 'cdf'")
    return id(buffer), sampling


def _validate_configuration_key(key) -> None:
    """Require immutable CPU metadata; comparing a CUDA tensor can synchronize."""
    if key is None or type(key) in (bool, int, float, str, bytes):
        # NaN is not equal to itself and cannot form a stable configuration.
        if isinstance(key, float) and math.isnan(key):
            raise ValueError("Captured configuration keys cannot contain NaN")
        return
    if isinstance(key, (tuple, frozenset)):
        for value in key:
            _validate_configuration_key(value)
        return
    raise TypeError("Captured configuration must contain immutable CPU scalars/tuples, not tensors or mutable values")


def graph_state(
    modules: Iterable[nn.Module], optimizers: Iterable[torch.optim.Optimizer],
    extra: Iterable[torch.Tensor] = (),
) -> list[torch.Tensor]:
    """Tensors whose storage a captured learner must continue to read/write."""
    state = []
    for module in modules:
        state.extend(module.parameters())
        state.extend(module.buffers())
    for optimizer in optimizers:
        state.extend(value for values in optimizer.state.values()
                     for value in values.values() if isinstance(value, torch.Tensor))
        state.extend(group["lr"] for group in optimizer.param_groups
                     if isinstance(group["lr"], torch.Tensor))
    state.extend(extra)
    return state


def prepare_adam_for_capture(optimizer: torch.optim.Adam, device: torch.device | str) -> None:
    """Keep Adam math/hyperparameters and make LR/step state capture-safe.

    This also handles historical optimizer checkpoints whose parameter groups
    contain a float LR, CPU step tensors and ``capturable=False``. Call before
    warm-up/capture, not inside the recorded function. Existing CUDA LR tensors
    are kept in place, so later cosine-schedule changes are visible on replay.
    """
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Capturable Adam requires a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if not isinstance(optimizer, torch.optim.Adam):
        raise TypeError("The learner capture adapter expects torch.optim.Adam")
    for group in optimizer.param_groups:
        lr = group["lr"]
        if not isinstance(lr, torch.Tensor):
            group["lr"] = torch.tensor(float(lr), dtype=torch.float32, device=device)
        elif lr.device != device:
            group["lr"] = lr.to(device=device)
        if group["lr"].numel() != 1:
            raise ValueError("Captured Adam learning rate must be scalar")
        group["capturable"] = True
        # CUDA Adam normally chooses foreach automatically. Making that choice
        # explicit supports a tensor LR without introducing a fused algorithm.
        group["foreach"] = not bool(group.get("fused", False))
        for parameter in group["params"]:
            if parameter.device != device:
                raise ValueError("A captured optimizer must live on one CUDA device")
            values = optimizer.state.get(parameter, {})
            step = values.get("step")
            if isinstance(step, torch.Tensor) and step.device != device:
                values["step"] = step.to(device=device)
    optimizer.defaults["capturable"] = True


def make_capturable_adam(
    parameters: Iterable[nn.Parameter], lr: float, **kwargs,
) -> torch.optim.Adam:
    """Construct the same Adam with device LR and explicit foreach execution."""
    parameters = list(parameters)
    if not parameters:
        raise ValueError("Capturable Adam needs parameters")
    optimizer = torch.optim.Adam(parameters, lr=lr, **kwargs)
    prepare_adam_for_capture(optimizer, parameters[0].device)
    return optimizer


class CapturedUpdate:
    """Eager counted warm-up followed by one graph replay per call.

    ``fn`` must return scalar/small tensor metrics and have fixed tensor shapes,
    no host reads and no mutation of Python training counters. Returned metrics
    are detached and cloned out of the graph pool. CPU and distributed updates
    use an explicit eager fallback; no collectives are captured.

    ``configuration`` may return immutable CPU metadata used by Python branches
    in ``fn``. A change invalidates the old recording before any replay and
    restarts counted warm-up. Tensor values remain live without recapture.
    """

    def __init__(
        self, fn: Callable[[], dict[str, torch.Tensor]],
        state: Callable[[], Iterable[torch.Tensor]], *,
        enabled: bool = True, warmup: int = 3, label: str = "update",
        configuration: Callable[[], object] | None = None,
    ):
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 1:
            raise ValueError("CUDA Graph warmup must be a positive number of counted updates")
        self.fn, self.state = fn, state
        self.configuration = configuration
        self.enabled, self.warmup, self.label = enabled, warmup, label
        self.calls, self.phase, self.is_replay = 0, "eager", False
        self._remaining = warmup
        self._graph: torch.cuda.CUDAGraph | None = None
        self._stream: torch.cuda.Stream | None = None
        self._outputs: dict[str, torch.Tensor] = {}
        self._identity: tuple = ()
        self._fallback_reported = False
        self._configuration_key = _UNSET_CONFIGURATION
        self.configuration_invalidations = 0

    @property
    def captured(self) -> bool:
        return self._graph is not None

    def invalidate(self) -> None:
        """Drop a recording before explicitly replacing parameters/optimizer state."""
        self._graph, self._stream = None, None
        self._outputs, self._identity = {}, ()
        self._remaining = self.warmup
        self.phase, self.is_replay = "eager", False
        self._configuration_key = _UNSET_CONFIGURATION

    def check_configuration(self) -> None:
        """Invalidate changed Python dispatch before refreshing closed-over inputs.

        Trainers call this before prepare writes to a sampler's fixed storage.
        ``__call__`` checks again, so standalone callers also cannot replay an
        obsolete branch. This check itself never performs a training update.
        """
        if self.configuration is None:
            return
        key = self.configuration()
        _validate_configuration_key(key)
        if self._configuration_key is not _UNSET_CONFIGURATION and key != self._configuration_key:
            was_captured = self.captured
            if was_captured and self._stream is not None:
                # A rare hot switch may follow asynchronous replay. Complete
                # its uses of the private pool before releasing that graph.
                # The ordinary, fixed-configuration path never synchronizes.
                torch.cuda.synchronize(self._stream.device)
            self.invalidate()
            self.configuration_invalidations += 1
            if was_captured:
                print(f"[INFO] {self.label} CUDA Graph: invalidated because execution configuration changed")
        self._configuration_key = key

    @staticmethod
    def _storage_identity(state: Iterable[torch.Tensor]) -> tuple:
        return tuple((t.data_ptr(), tuple(t.shape), t.dtype, t.device) for t in state)

    @staticmethod
    def _metrics(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if any(not isinstance(value, torch.Tensor) for value in outputs.values()):
            raise TypeError("Captured learner metrics must stay as tensors")
        return {key: value.detach().clone() for key, value in outputs.items()}

    def __call__(self) -> dict[str, torch.Tensor]:
        self.check_configuration()
        self.calls += 1
        self.is_replay = False
        if not self.enabled:
            self.phase = "eager"
            return self.fn()
        state = list(self.state())
        if not state:
            raise ValueError("CapturedUpdate state must contain a model tensor")
        device = state[0].device
        distributed = (torch.distributed.is_available() and torch.distributed.is_initialized()
                       and torch.distributed.get_world_size() > 1)
        if device.type != "cuda" or distributed:
            if not self._fallback_reported:
                reason = "distributed gradient reduction" if distributed else "CPU device"
                print(f"[INFO] {self.label} CUDA Graph: eager fallback ({reason})")
                self._fallback_reported = True
            self.phase = "eager"
            return self.fn()
        with torch.cuda.device(device):
            if self._graph is not None:
                if self._storage_identity(state) != self._identity:
                    raise RuntimeError(
                        f"{self.label} captured tensor storage changed; invalidate the update before loading state"
                    )
                self.phase = "replay"
            else:
                if self._stream is None:
                    self._stream = torch.cuda.Stream(device=device)
                if self._remaining:
                    self._remaining -= 1
                    self.phase = "warmup"
                    self._stream.wait_stream(torch.cuda.current_stream(device))
                    with torch.cuda.stream(self._stream):
                        outputs = self.fn()
                    torch.cuda.current_stream(device).wait_stream(self._stream)
                    return self._metrics(outputs)
                torch.cuda.synchronize(device)
                with _recording_gc_guard():
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=self._stream):
                        self._outputs = self.fn()
                self._graph = graph
                self._identity = self._storage_identity(self.state())
                self.phase = "capture"
                print(f"[INFO] {self.label} CUDA Graph: captured complete update after {self.warmup} counted warmup calls")
            self._graph.replay()
            self.is_replay = True
            return self._metrics(self._outputs)
