"""Materialize compact references as ordinary, precomputed training NPZ clips.

No resampling, finite differencing, clearance correction or joint clamping is
performed. Float16 source values are promoted to float32, not reconstructed to
an unavailable pre-quantization motion. The optional reference_state extension
preserves quantized root quaternions/velocities separately from normalized FK
body outputs, so full and compact training receive the same reference inputs.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from supermimic.envs.mdp.motion_library import PACKED_REFERENCE_FORMAT, read_npz_reference
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.kinematics.rotations import quat_apply_inverse
from supermimic.robots.registry import get_robot_spec


def _integer(value, label):
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must contain integers")  # noqa: TRY004
    return int(value)


def _safe_name(value):
    if (not isinstance(value, str) or not value or value in (".", "..")
            or Path(value).name != value or "/" in value or "\\" in value
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise ValueError(f"Unsafe clip name: {value!r}")
    return value


def _load_pack(path, spec):
    pack = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(pack, dict) or any(key not in pack for key in ("ref", "names", "starts", "lengths")):
        raise ValueError("Packed library needs ref/names/starts/lengths")
    ref = pack["ref"]
    if (not isinstance(ref, torch.Tensor) or ref.dtype not in (torch.float16, torch.float32)
            or ref.ndim != 2 or ref.shape[1] != spec.state_dim or ref.shape[0] < 1):
        raise ValueError(f"Packed ref must be nonempty float16/float32 with {spec.state_dim} columns")
    if spec.name != "g1" and any(key not in pack for key in ("robot", "joint_names", "root_body_name", "fps")):
        raise ValueError("Non-G1 libraries require robot/joint_names/root_body_name/fps metadata")
    if pack.get("robot", "g1") != spec.name:
        raise ValueError("Packed library robot differs from --robot")
    if tuple(pack.get("joint_names", spec.joint_names)) != spec.joint_names:
        raise ValueError("Packed library joint_names differ from the selected robot")
    if pack.get("root_body_name", spec.root_body) != spec.root_body:
        raise ValueError("Packed library root_body_name differs from the selected robot")
    fps = np.asarray(pack.get("fps", 50.0))
    if fps.size != 1 or not np.isfinite(fps).all() or not np.isclose(float(fps.reshape(-1)[0]), 50., atol=1e-6, rtol=0):
        raise ValueError("Packed library must contain 50 Hz references")
    if pack.get("format", f"supermimic_compact_q_{spec.state_dim}_v1") != f"supermimic_compact_q_{spec.state_dim}_v1":
        raise ValueError("Unsupported packed library format")
    layout = ["root_pos_w:3", "root_quat_wxyz:4", f"joint_pos:{spec.num_joints}",
              "root_lin_vel_w:3", "root_ang_vel_w:3", f"joint_vel:{spec.num_joints}"]
    if tuple(pack.get("reference_layout", layout)) != tuple(layout):
        raise ValueError("Packed reference_layout differs from the selected robot")
    names = [_safe_name(value) for value in pack["names"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("Packed clip names must be nonempty and unique")
    if not (len(names) == len(pack["starts"]) == len(pack["lengths"])):
        raise ValueError("Packed clip metadata lengths differ")
    clips, offset = [], 0
    for name, start, length in zip(names, pack["starts"], pack["lengths"], strict=True):
        start, length = _integer(start, "starts"), _integer(length, "lengths")
        if start != offset or length < 1:
            raise ValueError("Packed clips must be positive and contiguous")
        offset += length
        clips.append((name, start, length))
    if offset != len(ref):
        raise ValueError("Packed clip metadata must cover all reference frames")
    return pack, clips


def _write_atomic(path, writer):
    """Publish a completed file without replacing even a racing destination."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
                                         mode="wb", delete=False) as stream:
            temporary = Path(stream.name)
            writer(stream)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_library(library: Path, out_dir: Path, *, robot: str = "khan_mini",
                   device: str = "cpu", chunk_size: int = 8192,
                   clips: list[str] | None = None, limit: int | None = None,
                   all_bodies: bool = False) -> dict:
    """Export selected clips in source-pack order; ``limit`` follows filtering.

    The output directory must be new or empty. Default outputs contain tracked
    bodies only (Khan:15); --all-bodies includes the complete authored body tree.
    Computation uses eager float32 FK on CPU or CUDA, with bounded chunk size.
    """
    started = time.perf_counter()
    library, out_dir = Path(library).resolve(), Path(out_dir).resolve()
    if chunk_size < 1 or (limit is not None and limit < 1):
        raise ValueError("chunk_size and limit must be positive")
    if not library.is_file():
        raise FileNotFoundError(library)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise FileExistsError(f"Use a new or empty output directory: {out_dir}")
    spec = get_robot_spec(robot)
    pack, metadata = _load_pack(library, spec)
    if clips is not None:
        if not clips or len(set(clips)) != len(clips):
            raise ValueError("--clips must contain unique names")
        unknown = set(clips) - {name for name, _, _ in metadata}
        if unknown:
            raise ValueError(f"Unknown clips: {sorted(unknown)}")
        selected = [item for item in metadata if item[0] in set(clips)]
    else:
        selected = metadata
    if limit is not None:
        selected = selected[:limit]
    # Validate every selected frame before publishing any output file.
    ref = pack["ref"].detach()
    for name, start, length in selected:
        for local in range(0, length, chunk_size):
            chunk = ref[start+local:start+min(local+chunk_size, length)].float()
            if not bool(torch.isfinite(chunk).all()):
                raise ValueError(f"{name}: packed reference contains NaN/Inf")
            norm = chunk[:, 3:7].norm(dim=-1)
            if not bool(((norm-1).abs() <= .01).all()):
                raise ValueError(f"{name}: root quaternion must be near unit length")
    compute_device = torch.device(device)
    if compute_device.type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    model = spec.load_model()
    fk = G1Kinematics(model, device=compute_device)
    body_names = tuple(fk.body_names) if all_bodies else spec.tracked_bodies
    body_indices = tuple(fk.body_index(name) for name in body_names)
    if spec.root_body not in body_names:
        raise ValueError("The selected body layout must include the reference root")
    with library.open("rb") as stream:
        source_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    j = 7 + spec.num_joints
    with torch.no_grad():
        for ordinal, (name, start, length) in enumerate(selected, 1):
            exact = ref[start:start+length].float().contiguous().numpy()
            body = {"pos": np.empty((length, len(body_names), 3), np.float32),
                    "quat": np.empty((length, len(body_names), 4), np.float32),
                    "lin_vel": np.empty((length, len(body_names), 3), np.float32),
                    "ang_vel": np.empty((length, len(body_names), 3), np.float32)}
            for local in range(0, length, chunk_size):
                end = min(local+chunk_size, length)
                chunk = torch.from_numpy(exact[local:end]).to(compute_device)
                qpos = chunk[:, :j]
                ang_b = quat_apply_inverse(chunk[:, 3:7], chunk[:, j+3:j+6])
                qvel = torch.cat((chunk[:, j:j+3], ang_b, chunk[:, j+6:]), dim=-1)
                result = fk(qpos, qvel, body_indices=body_indices)
                for key, value in body.items():
                    value[local:end] = result[key].cpu().numpy()
            if not all(np.isfinite(value).all() for value in body.values()):
                raise ValueError(f"{name}: FK produced nonfinite body states")
            fields = {
                "fps": np.array(50., dtype=np.float32),
                "joint_names": np.asarray(spec.joint_names), "body_names": np.asarray(body_names),
                "joint_pos": exact[:, 7:j], "joint_vel": exact[:, j+6:],
                "body_pos_w": body["pos"], "body_quat_w": body["quat"],
                "body_lin_vel_w": body["lin_vel"], "body_ang_vel_w": body["ang_vel"],
                "robot": np.array(spec.name), "source": np.array(str(library)),
                "source_clip": np.array(name), "source_library_sha256": np.array(source_sha),
                "source_storage_dtype": np.array(str(ref.dtype)),
                "source_start_frame": np.array(start, dtype=np.int64),
                "reference_state": exact, "reference_state_format": np.array(PACKED_REFERENCE_FORMAT),
            }
            target = out_dir/f"{name}.npz"
            _write_atomic(target, lambda stream, fields=fields: np.savez_compressed(stream, **fields))
            # Exercise the actual training validator, including exact-ref/body consistency.
            loaded, _ = read_npz_reference(target, robot_name=spec.name, root_body_name=spec.root_body,
                                             joint_names=spec.joint_names, body_names=spec.tracked_bodies)
            if not np.array_equal(loaded, exact):
                raise AssertionError(f"{name}: exact reference was not preserved")
            reports.append({"name": name, "frames": length, "source_start_frame": start,
                            "file": target.name, "bytes": target.stat().st_size})
            print(f"[{ordinal}/{len(selected)}] {target.name}: {length} frames", flush=True)
    report = {"status": "passed", "source_library": str(library), "source_sha256": source_sha,
              "robot": spec.name, "fps": 50., "source_clips": len(metadata),
              "source_frames": len(ref), "exported_clips": len(reports),
              "exported_frames": sum(row["frames"] for row in reports),
              "source_storage_dtype": str(ref.dtype), "output_dtype": "float32",
              "body_names": list(body_names), "joint_names": list(spec.joint_names),
              "reference_state_format": PACKED_REFERENCE_FORMAT,
              "conversion": "Exact stored references promoted to float32; eager FK, no resampling, finite differencing, height offset or joint clamping",
              "device": str(compute_device), "chunk_size": chunk_size,
              "clips": reports, "wall_s": time.perf_counter()-started}
    _write_atomic(out_dir/"conversion_report.json", lambda stream: stream.write(
        (json.dumps(report, indent=2, ensure_ascii=False)+"\n").encode()))
    manifest = io.StringIO()
    writer = csv.DictWriter(manifest, fieldnames=("name", "keep", "frames", "fps", "file", "source_start_frame"))
    writer.writeheader()
    for row in reports:
        writer.writerow({"name": row["name"], "keep": 1, "frames": row["frames"], "fps": 50,
                         "file": row["file"], "source_start_frame": row["source_start_frame"]})
    _write_atomic(out_dir/"manifest.csv", lambda stream: stream.write(manifest.getvalue().encode()))
    return report
