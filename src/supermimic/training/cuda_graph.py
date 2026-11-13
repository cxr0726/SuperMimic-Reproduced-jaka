"""Capture complete learner updates without adding uncounted optimizer steps.

Warm-up calls are ordinary, counted training updates. Recording a CUDA graph
does not execute its kernels; its first replay is that call's single update.
Physical collection and reset remain outside these graphs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import nn


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
    """

    def __init__(
        self, fn: Callable[[], dict[str, torch.Tensor]],
        state: Callable[[], Iterable[torch.Tensor]], *,
        enabled: bool = True, warmup: int = 3, label: str = "update",
    ):
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 1:
            raise ValueError("CUDA Graph warmup must be a positive number of counted updates")
        self.fn, self.state = fn, state
        self.enabled, self.warmup, self.label = enabled, warmup, label
        self.calls, self.phase, self.is_replay = 0, "eager", False
        self._remaining = warmup
        self._graph: torch.cuda.CUDAGraph | None = None
        self._stream: torch.cuda.Stream | None = None
        self._outputs: dict[str, torch.Tensor] = {}
        self._identity: tuple = ()
        self._fallback_reported = False

    @property
    def captured(self) -> bool:
        return self._graph is not None

    def invalidate(self) -> None:
        """Drop a recording before explicitly replacing parameters/optimizer state."""
        self._graph, self._stream = None, None
        self._outputs, self._identity = {}, ()
        self._remaining = self.warmup
        self.phase, self.is_replay = "eager", False

    @staticmethod
    def _storage_identity(state: Iterable[torch.Tensor]) -> tuple:
        return tuple((t.data_ptr(), tuple(t.shape), t.dtype, t.device) for t in state)

    @staticmethod
    def _metrics(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if any(not isinstance(value, torch.Tensor) for value in outputs.values()):
            raise TypeError("Captured learner metrics must stay as tensors")
        return {key: value.detach().clone() for key, value in outputs.items()}

    def __call__(self) -> dict[str, torch.Tensor]:
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
