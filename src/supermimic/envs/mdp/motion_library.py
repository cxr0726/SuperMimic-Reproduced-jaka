"""Multi-clip motion library held as flat GPU tensors.

All clips are concatenated along a single global frame axis.  A per-frame
``clip_id`` and per-clip ``start`` / ``length`` tables let a batch of
environments index arbitrary clips and frames with plain gathers.

Two storage modes:

* **full** (small sets, e.g. LAFAN1): per frame we keep the reference joints, the pelvis
  state and the world-frame link states of the *tracked* bodies, plus precomputed 68-d
  goal features. ~1 KB/frame.
* **compact** (BONES-SEED, 47.5M frames): per frame only the 71-d reference state
  ``[root_pos(3), root_quat(4), joint_pos(29), root_lin_vel_w(3), root_ang_vel_w(3),
  joint_vel(29)]`` in fp16/fp32 (284 B/frame in fp32); tracked-body states and goal
  features are computed on the fly with the differentiable FK (plan 1.3).

The widths above describe G1 (29 joints). For N joints the reference state has
13+2N columns and each goal frame has 10+2N columns (Khan: 67 and 64).

Goal features (per frame, 68-d for G1, plan D5):
  [joint_pos - q0 (29), joint_vel (29), projected gravity of the ref root (3),
   ref root angular velocity in the root frame (3),
   ref root linear velocity in the ref heading frame (3), ref root height (1)]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from supermimic.kinematics.rotations import projected_gravity, quat_apply_inverse, yaw_quat

GOAL_FEAT_DIM = 68
REF_DIM = 71


@dataclass
class ClipInfo:
    name: str
    start: int
    length: int

    @property
    def end(self) -> int:  # exclusive
        return self.start + self.length


@dataclass
class RefState:
    root_pos_w: torch.Tensor  # (N,3)
    root_quat_w: torch.Tensor  # (N,4) wxyz
    joint_pos: torch.Tensor  # (N,29)
    root_lin_vel_w: torch.Tensor  # (N,3)
    root_ang_vel_w: torch.Tensor  # (N,3)
    joint_vel: torch.Tensor  # (N,29)


@dataclass
class RefBodies:
    pos_w: torch.Tensor  # (N,B,3)
    quat_w: torch.Tensor  # (N,B,4)
    lin_vel_w: torch.Tensor  # (N,B,3)
    ang_vel_w: torch.Tensor  # (N,B,3)


def goal_features(ref: RefState, default_joint_pos: torch.Tensor) -> torch.Tensor:
    q = ref.root_quat_w
    grav = projected_gravity(q)
    ang_b = quat_apply_inverse(q, ref.root_ang_vel_w)
    lin_h = quat_apply_inverse(yaw_quat(q), ref.root_lin_vel_w)
    height = ref.root_pos_w[..., 2:3]
    return torch.cat([ref.joint_pos - default_joint_pos, ref.joint_vel, grav, ang_b, lin_h, height], dim=-1)


def read_npz_reference(
    path: str | Path, *, robot_name: str = "g1", root_body_name: str = "pelvis",
    joint_names: tuple[str, ...] | list[str] | None = None, body_names: tuple[str, ...] = (),
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Validate one named 50 Hz motion and return its reference and tracked bodies.

    Shared by the runtime loader and compact-cache builder so a cache cannot
    bypass the robot layout checks applied to ordinary NPZ training input.
    """
    from supermimic.robots.registry import get_robot_spec

    spec = get_robot_spec(robot_name)
    expected_joints = tuple(spec.joint_names if joint_names is None else joint_names)
    if expected_joints != tuple(spec.joint_names) or root_body_name != spec.root_body:
        raise ValueError(f"Requested joint/root layout differs from {robot_name}")
    with np.load(path, allow_pickle=False) as data:
        required = ("fps", "joint_names", "body_names", "joint_pos", "joint_vel",
                    "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
        if any(key not in data for key in required):
            raise ValueError(f"{path}: missing required motion fields")
        fps = np.asarray(data["fps"])
        if fps.size != 1 or not np.isfinite(fps).all() or not np.isclose(float(fps.reshape(-1)[0]), 50):
            raise ValueError(f"{path}: training requires 50 Hz motion data")
        if tuple(str(name) for name in data["joint_names"]) != expected_joints:
            raise ValueError(f"{path}: joint_names differ from the selected {robot_name} model")
        if "robot" in data and str(np.asarray(data["robot"]).reshape(-1)[0]) != robot_name:
            raise ValueError(f"{path}: robot metadata differs from the selected model")
        names = [str(name) for name in data["body_names"]]
        if len(set(names)) != len(names):
            raise ValueError(f"{path}: duplicate body names")
        if any(name not in names for name in (*body_names, root_body_name)):
            raise ValueError(f"{path}: missing tracked or root body")
        # Resolve every file independently; body columns may have different orders.
        body_idx = np.array([names.index(name) for name in body_names], dtype=np.int64)
        root_idx = names.index(root_body_name)
        joint_pos, joint_vel = data["joint_pos"], data["joint_vel"]
        n = int(joint_pos.shape[0])
        if n < 1:
            raise ValueError(f"{path}: empty motion")
        bp, bq = data["body_pos_w"], data["body_quat_w"]
        blv, bav = data["body_lin_vel_w"], data["body_ang_vel_w"]
        for key, value, shape in (
            ("joint_pos", joint_pos, (n, spec.num_joints)),
            ("joint_vel", joint_vel, (n, spec.num_joints)),
            ("body_pos_w", bp, (n, len(names), 3)),
            ("body_quat_w", bq, (n, len(names), 4)),
            ("body_lin_vel_w", blv, (n, len(names), 3)),
            ("body_ang_vel_w", bav, (n, len(names), 3)),
        ):
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{path}: {key} must be finite with shape {shape}")
        if not np.allclose(np.linalg.norm(bq, axis=-1), 1, atol=1e-4, rtol=0):
            raise ValueError(f"{path}: body_quat_w must contain unit WXYZ quaternions")
        ref = np.concatenate([bp[:, root_idx], bq[:, root_idx], joint_pos, blv[:, root_idx],
                              bav[:, root_idx], joint_vel], axis=1).astype(np.float32)
        bodies = {"pos": bp[:, body_idx], "quat": bq[:, body_idx],
                  "lin": blv[:, body_idx], "ang": bav[:, body_idx]}
    return ref, bodies


class MotionLibrary:
    def __init__(
        self,
        npz_files: list[str | Path],
        body_names: tuple[str, ...],
        default_joint_pos: torch.Tensor | np.ndarray,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        bin_frames: int = 50,
        compact: bool = False,
        store_dtype: torch.dtype = torch.float16,
        robot_name: str = "g1",
        root_body_name: str = "pelvis",
        joint_names: tuple[str, ...] | list[str] | None = None,
    ):
        from supermimic.robots.registry import get_robot_spec

        robot_spec = get_robot_spec(robot_name)
        self.device = torch.device(device)
        self.dtype = dtype
        self.robot_name = robot_name
        self.root_body_name = root_body_name
        self.body_names = tuple(body_names)
        self.bin_frames = int(bin_frames)
        if self.bin_frames < 1:
            raise ValueError("bin_frames must be positive")
        self.compact = compact
        if len(npz_files) == 0:
            raise ValueError("MotionLibrary needs at least one clip")
        if isinstance(default_joint_pos, torch.Tensor):
            q0 = default_joint_pos.detach().to(device=self.device, dtype=dtype).reshape(-1)
        else:
            q0 = torch.as_tensor(np.asarray(default_joint_pos), dtype=dtype, device=self.device).reshape(-1)
        self.default_joint_pos = q0
        self.num_joints = int(q0.numel())
        self.ref_dim = 13 + 2 * self.num_joints
        self.goal_feat_dim = 10 + 2 * self.num_joints
        self.joint_names = tuple(robot_spec.joint_names if joint_names is None else joint_names)
        if self.num_joints != robot_spec.num_joints or self.joint_names != tuple(robot_spec.joint_names):
            raise ValueError(f"Joint layout differs from the selected {robot_name} model")
        if root_body_name != robot_spec.root_body:
            raise ValueError(f"Root body {root_body_name!r} differs from {robot_name}: {robot_spec.root_body!r}")
        self.fps = 50.0  # The training environment advances one reference frame every 0.02 s.

        clips: list[ClipInfo] = []
        start = 0
        ref_chunks: list[np.ndarray] = []
        body_chunks = {"pos": [], "quat": [], "lin": [], "ang": []}
        # A single packed cache (build_library.py) can stand in for the npz list (compact mode only).
        if len(npz_files) == 1 and str(npz_files[0]).endswith(".pt"):
            if not compact:
                raise ValueError("packed .pt libraries are compact-only")
            pack = torch.load(npz_files[0], map_location="cpu", weights_only=False)
            packed_ref = pack["ref"]
            if packed_ref.ndim != 2 or packed_ref.shape[1] != self.ref_dim:
                raise ValueError(f"Packed reference must have {self.ref_dim} columns for {robot_name}")
            # Historical G1 caches predate metadata. Other robots must identify their layout.
            if robot_name != "g1" and any(key not in pack for key in ("robot", "joint_names", "root_body_name", "fps")):
                raise ValueError("Non-G1 packed libraries need robot/joint_names/root_body_name/fps metadata")
            if pack.get("robot", "g1") != robot_name:
                raise ValueError("Packed library robot differs from the selected model")
            if tuple(pack.get("joint_names", self.joint_names)) != self.joint_names:
                raise ValueError("Packed library joint_names differ from the selected model")
            if pack.get("root_body_name", root_body_name) != root_body_name or not np.isclose(float(pack.get("fps", 50)), 50):
                raise ValueError("Packed library root body or fps differs from training configuration")
            self._ref = pack["ref"].to(device=self.device, dtype=store_dtype)
            if not (len(pack["names"]) == len(pack["starts"]) == len(pack["lengths"])):
                raise ValueError("Packed clip metadata lengths differ")
            clips = [ClipInfo(n, int(st), int(ln)) for n, st, ln in zip(pack["names"], pack["starts"], pack["lengths"])]
            start = int(self._ref.shape[0])
            expected_start = 0
            for clip in clips:
                if clip.start != expected_start or clip.length < 1:
                    raise ValueError("Packed clips must be positive and contiguous")
                expected_start = clip.end
            if expected_start != start or not clips:
                raise ValueError("Packed clips do not cover the reference frames")
            ref_chunks = None
        for f in (npz_files if ref_chunks is not None else []):
            ref, bodies = read_npz_reference(f, robot_name=robot_name, root_body_name=root_body_name,
                                            joint_names=self.joint_names, body_names=self.body_names)
            n = len(ref)
            ref_chunks.append(ref)
            if not compact:
                for key, chunks in body_chunks.items():
                    chunks.append(bodies[key])
            clips.append(ClipInfo(Path(f).stem, start, n))
            start += n
        self.clips = clips
        self.num_clips = len(clips)
        self.num_frames = start

        if ref_chunks is not None:
            ref_np = np.concatenate(ref_chunks, 0)
            assert ref_np.shape[1] == self.ref_dim
            self._ref = torch.as_tensor(ref_np, device=self.device, dtype=store_dtype if compact else dtype)
            del ref_np, ref_chunks
        if not compact:
            t = lambda arrs: torch.as_tensor(np.concatenate(arrs, 0), dtype=dtype, device=self.device)
            self.body_pos_w = t(body_chunks["pos"])
            self.body_quat_w = t(body_chunks["quat"])
            self.body_lin_vel_w = t(body_chunks["lin"])
            self.body_ang_vel_w = t(body_chunks["ang"])
            del body_chunks
            self.goal_feat = goal_features(self._split(self._ref), q0)
        else:
            from supermimic.kinematics.g1_fk import G1Kinematics

            self._fk = G1Kinematics(robot_spec.load_model(), device=self.device, dtype=dtype)
            self._tracked_idx = self._fk.body_indices(self.body_names)

        self.clip_start = torch.tensor([c.start for c in clips], device=self.device, dtype=torch.long)
        self.clip_len = torch.tensor([c.length for c in clips], device=self.device, dtype=torch.long)
        self.clip_last = self.clip_start + self.clip_len - 1
        self.frame_clip = torch.repeat_interleave(torch.arange(self.num_clips, device=self.device), self.clip_len)
        # Bins of `bin_frames` frames inside each clip (plan D14), numbered globally.
        bins_per_clip = (self.clip_len + self.bin_frames - 1) // self.bin_frames
        self.clip_bin_offset = torch.cumsum(bins_per_clip, 0) - bins_per_clip
        self.num_bins = int(bins_per_clip.sum().item())
        bin_clip = torch.repeat_interleave(torch.arange(self.num_clips, device=self.device), bins_per_clip)
        local_bin = torch.arange(self.num_bins, device=self.device) - self.clip_bin_offset[bin_clip]
        self.bin_start = self.clip_start[bin_clip] + local_bin * self.bin_frames
        self.bin_end = torch.minimum(self.bin_start + self.bin_frames, self.clip_start[bin_clip] + self.clip_len[bin_clip])
        self.bin_clip = bin_clip

    # ------------------------------------------------------------------ per-frame access

    @staticmethod
    def _split(r: torch.Tensor) -> RefState:
        num_joints = (r.shape[-1] - 13) // 2
        j = 7 + num_joints
        return RefState(r[..., 0:3], r[..., 3:7], r[..., 7:j], r[..., j:j + 3],
                        r[..., j + 3:j + 6], r[..., j + 6:])

    def ref(self, frame: torch.Tensor) -> RefState:
        return self._split(self._ref[frame].to(self.dtype))

    def ref71(self, frame: torch.Tensor) -> torch.Tensor:
        """Stored reference state (legacy method name; width is robot dependent)."""
        return self._ref[frame].to(self.dtype)

    def bodies(self, frame: torch.Tensor) -> RefBodies:
        """Tracked-body link states (world) at the given frames."""
        if not self.compact:
            return RefBodies(self.body_pos_w[frame], self.body_quat_w[frame], self.body_lin_vel_w[frame], self.body_ang_vel_w[frame])
        qpos, qvel = self.ref_qpos_qvel(frame)
        with torch.no_grad():
            out = self._fk(qpos, qvel)
        i = self._tracked_idx
        return RefBodies(out["pos"][..., i, :], out["quat"][..., i, :], out["lin_vel"][..., i, :], out["ang_vel"][..., i, :])

    # Convenience column accessors (full fp32 copies; prefer ``ref()`` in hot paths).
    @property
    def root_pos_w(self) -> torch.Tensor:
        return self._ref[:, 0:3].to(self.dtype)

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self._ref[:, 3:7].to(self.dtype)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._ref[:, 7:7 + self.num_joints].to(self.dtype)

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        return self._ref[:, 7 + self.num_joints:10 + self.num_joints].to(self.dtype)

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        return self._ref[:, 10 + self.num_joints:13 + self.num_joints].to(self.dtype)

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._ref[:, 13 + self.num_joints:].to(self.dtype)

    # ------------------------------------------------------------------ queries

    def bin_of_frame(self, frame: torch.Tensor) -> torch.Tensor:
        clip = self.frame_clip[frame]
        return self.clip_bin_offset[clip] + (frame - self.clip_start[clip]) // self.bin_frames

    def last_frame_of(self, frame: torch.Tensor) -> torch.Tensor:
        return self.clip_last[self.frame_clip[frame]]

    def goal_window(self, frame: torch.Tensor, k: int) -> torch.Tensor:
        """(N, k*68) goal features of frames frame..frame+k-1, clamped at the clip end."""
        last = self.last_frame_of(frame)
        offs = torch.arange(k, device=self.device)
        idx = torch.minimum(frame[:, None] + offs[None, :], last[:, None])  # (N,k)
        if not self.compact:
            return self.goal_feat[idx].reshape(frame.shape[0], k * self.goal_feat_dim)
        feat = goal_features(self.ref(idx.reshape(-1)), self.default_joint_pos)
        return feat.reshape(frame.shape[0], k * self.goal_feat_dim)

    def reference_window(self, frame: torch.Tensor, k: int) -> RefState:
        """Reference state at t..t+k-1, clamped within each clip, shape (N,K,*).

        Full and compact libraries share this path. Absolute reference joint
        positions and root poses are needed by the Khan Isaac observation layout.
        """
        last = self.last_frame_of(frame)
        offsets = torch.arange(k, device=self.device)
        indices = torch.minimum(frame[:, None] + offsets, last[:, None])
        return self.ref(indices)

    def ref_qpos_qvel(self, frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """MuJoCo-layout reference state: qpos (N,36) and qvel (N,35; root ang vel in body frame)."""
        r = self.ref(frame)
        qpos = torch.cat([r.root_pos_w, r.root_quat_w, r.joint_pos], dim=-1)
        ang_b = quat_apply_inverse(r.root_quat_w, r.root_ang_vel_w)
        qvel = torch.cat([r.root_lin_vel_w, ang_b, r.joint_vel], dim=-1)
        return qpos, qvel

    def summary(self) -> str:
        hours = self.num_frames / self.fps / 3600.0
        mem = self._ref.numel() * self._ref.element_size() / 1e9
        if not self.compact:
            mem += sum(t.numel() * t.element_size() for t in (self.body_pos_w, self.body_quat_w, self.body_lin_vel_w, self.body_ang_vel_w, self.goal_feat)) / 1e9
        return f"MotionLibrary({'compact' if self.compact else 'full'}): {self.num_clips} clips, {self.num_frames} frames ({hours:.3f} h @50Hz), {self.num_bins} bins, {mem:.2f} GB on {self.device}"


def discover_npz(directory: str | Path, limit: int | None = None, names: list[str] | None = None, manifest: str | Path | None = None) -> list[Path]:
    """List npz clips; ``manifest`` (csv with name,keep columns) restricts to keep==1."""
    directory = Path(directory)
    files = sorted(directory.glob("*.npz"))
    if manifest is not None:
        import csv

        with open(manifest) as f:
            keep = {row["name"] for row in csv.DictReader(f) if row.get("keep", "1") in ("1", "True", "true")}
        files = [f for f in files if f.stem in keep]
    if names is not None:
        wanted = set(names)
        files = [f for f in files if f.stem in wanted]
    if limit is not None:
        files = files[:limit]
    return files
