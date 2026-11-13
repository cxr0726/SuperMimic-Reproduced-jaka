"""World-model fitting on k-step replay windows (paper Eq. 1, Table A.5)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import ReplayBuffer, WindowBatch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.training.cuda_graph import CapturedUpdate, graph_state, prepare_adam_for_capture
from supermimic.training.distributed import allreduce_grads, world_size
from supermimic.world_model.heads import ReferenceFrame
from supermimic.world_model.world_model import WorldModel


@dataclass
class WMTrainerCfg:
    window: int = 8
    batches_per_update: int = 10
    batch_size: int = 2048
    lr: float = 2e-4
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    compile: str | None = "default"
    cuda_graph: bool = False
    cuda_graph_warmup: int = 3  # Counted complete updates, not extra minibatches.


def reference_at(lib: MotionLibrary, frame: torch.Tensor) -> ReferenceFrame:
    r = lib.ref(frame)
    b = lib.bodies(frame)
    return ReferenceFrame(b.pos_w, b.quat_w, b.lin_vel_w, b.ang_vel_w,
                          r.root_pos_w, r.root_quat_w, r.joint_pos, r.joint_vel)


class WorldModelTrainer:
    def __init__(self, wm: WorldModel, lib: MotionLibrary, cfg: WMTrainerCfg):
        self.wm = wm
        self.lib = lib
        self.cfg = cfg
        self.opt = torch.optim.Adam(wm.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self._device = next(wm.parameters()).device
        self.capture_enabled = cfg.cuda_graph and self._device.type == "cuda" and world_size() == 1
        self._buffer: ReplayBuffer | None = None
        self._prepared = None
        if self.capture_enabled:
            prepare_adam_for_capture(self.opt, self._device)
            # Full-update capture owns CUDA Graph execution. A nested
            # reduce-overhead graph would capture/reuse the same backward
            # independently and can overwrite the outer graph's saved tensors.
            wm._loss_step_fn = (torch.compile(
                wm._loss_step, dynamic=False, options={"triton.cudagraphs": False},
            ) if cfg.compile else wm._loss_step)
        else:
            wm.enable_compile(cfg.compile)
        self.update_graph = CapturedUpdate(
            self._train_update, self._capture_state, enabled=cfg.cuda_graph,
            warmup=cfg.cuda_graph_warmup, label="WM update",
        )

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            if isinstance(g["lr"], torch.Tensor):
                g["lr"].fill_(lr)
            else:
                g["lr"] = lr

    def reset_capture(self) -> None:
        """Invalidate recordings after replacing checkpoint/optimizer state."""
        self.update_graph.invalidate()
        self._prepared = None
        if self.capture_enabled:
            prepare_adam_for_capture(self.opt, self._device)

    def loss_on(self, w: WindowBatch):
        k, n = w.frame.shape
        # Only the root pose of the references is needed for the termination features.
        r = self.lib.ref(w.frame.reshape(-1))
        refs = [ReferenceFrame(None, None, None, None, r.root_pos_w[j * n : (j + 1) * n], r.root_quat_w[j * n : (j + 1) * n]) for j in range(k)]
        return self.wm.window_loss(w.states, w.a_exec, w.xi, w.done, refs)

    def _capture_state(self) -> list[torch.Tensor]:
        extra = []
        for obj in (self._buffer, self.lib):
            if obj is not None:
                extra.extend(value for value in vars(obj).values() if isinstance(value, torch.Tensor))
        if self._prepared is not None:
            extra.extend((self._prepared.weights, self._prepared.count))
        return graph_state((self.wm,), (self.opt,), extra)

    def _train_batch(self) -> dict[str, torch.Tensor]:
        if self._prepared is None:
            w = self._buffer.sample_windows(self.cfg.batch_size, self.cfg.window)
        else:
            w = self._buffer.sample_windows(self.cfg.batch_size, self.cfg.window,
                                            prepared=self._prepared)
        loss, metrics = self.loss_on(w)
        self.opt.zero_grad(set_to_none=not self.capture_enabled)
        loss.backward()
        allreduce_grads(self.wm)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.wm.parameters(), self.cfg.grad_clip)
        self.opt.step()
        return {**metrics, "loss": loss.detach(), "grad_norm": grad_norm.detach()}

    def _train_update(self) -> dict[str, torch.Tensor]:
        """One graph covers every configured minibatch and metric reduction."""
        aggregate: dict[str, torch.Tensor] = {}
        for _ in range(self.cfg.batches_per_update):
            for key, value in self._train_batch().items():
                aggregate[key] = aggregate.get(key, 0.0) + value
        return {key: value / self.cfg.batches_per_update for key, value in aggregate.items()}

    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
        if self._buffer is not None and buffer is not self._buffer and self.update_graph.captured:
            raise ValueError("Replacing replay storage requires invalidating the WM update graph")
        self._buffer = buffer
        if self.capture_enabled:
            # Restore old Adam checkpoints before the first recording as well.
            prepare_adam_for_capture(self.opt, self._device)
            self._prepared = buffer.prepare_windows(self.cfg.window)
        metrics = self.update_graph()
        # Only CPU diagnostics leave the complete update graph.
        return {key: float(value) for key, value in metrics.items()}

    @torch.no_grad()
    def online_error(self, buffer: ReplayBuffer, n: int = 1024, k: int = 8) -> torch.Tensor:
        """k-step open-loop error on (recent) replay windows; (k,) tensor."""
        w = buffer.sample_windows(n, k)
        return self.wm.rollout_error(w.states, w.a_exec, w.xi)
