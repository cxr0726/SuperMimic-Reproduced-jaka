"""Extract named motions from a compact PT without changing stored references.

The built-in mixed480 selection is a motion-name match to the old NPZ set. Its
LAFAN references are the versions in the supplied PT, not reconstructed old NPZ
values. No FK, resampling, quantization, grounding or joint clamping is done.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch

from supermimic.robots.registry import get_robot_spec


def _integer(value, label):
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must contain integers")  # noqa: TRY004
    return int(value)


def _name(value):
    if (not isinstance(value, str) or not value or value in (".", "..")
            or Path(value).name != value or "/" in value or "\\" in value
            or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise ValueError(f"Unsafe clip name: {value!r}")
    return value


def _load_pack(path):
    pack = torch.load(path, mmap=True, map_location="cpu", weights_only=False)
    if not isinstance(pack, dict) or not all(k in pack for k in ("ref", "names", "starts", "lengths")):
        raise ValueError("Packed library needs ref/names/starts/lengths")
    robot = pack.get("robot", "g1")
    spec = get_robot_spec(robot)
    ref = pack["ref"]
    if (not isinstance(ref, torch.Tensor) or ref.ndim != 2 or len(ref) < 1
            or ref.shape[1] != spec.state_dim or ref.dtype not in (torch.float16, torch.float32)):
        raise ValueError(f"Packed ref must be nonempty float16/float32 with {spec.state_dim} columns")
    if robot != "g1" and any(k not in pack for k in ("joint_names", "root_body_name", "fps")):
        raise ValueError("Non-G1 libraries require robot/joint_names/root_body_name/fps metadata")
    if tuple(pack.get("joint_names", spec.joint_names)) != spec.joint_names:
        raise ValueError("Packed joint_names differ from the robot")
    if pack.get("root_body_name", spec.root_body) != spec.root_body:
        raise ValueError("Packed root_body_name differs from the robot")
    fps = np.asarray(pack.get("fps", 50.))
    if (fps.size != 1 or not np.isfinite(fps).all()
            or not np.isclose(float(fps.reshape(-1)[0]), 50., atol=1e-6, rtol=0)):
        raise ValueError("Packed library must contain 50 Hz references")
    if pack.get("format", f"supermimic_compact_q_{spec.state_dim}_v1") != f"supermimic_compact_q_{spec.state_dim}_v1":
        raise ValueError("Unsupported packed format")
    layout = ["root_pos_w:3", "root_quat_wxyz:4", f"joint_pos:{spec.num_joints}",
              "root_lin_vel_w:3", "root_ang_vel_w:3", f"joint_vel:{spec.num_joints}"]
    if tuple(pack.get("reference_layout", layout)) != tuple(layout):
        raise ValueError("Packed reference_layout differs from the robot")
    names = [_name(n) for n in pack["names"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("Packed clip names must be nonempty and unique")
    if not len(names) == len(pack["starts"]) == len(pack["lengths"]):
        raise ValueError("Packed clip metadata lengths differ")
    clips, offset = [], 0
    for name, start, length in zip(names, pack["starts"], pack["lengths"], strict=True):
        start, length = _integer(start, "starts"), _integer(length, "lengths")
        if start != offset or length < 1:
            raise ValueError("Packed clips must have positive lengths and contiguous starts")
        clips.append((name, start, length))
        offset += length
    if offset != len(ref):
        raise ValueError("Packed metadata must cover every reference frame")
    return pack, clips


def mixed480_names():
    """Audited 444 mixed + 36 LAFAN packed names; no source NPZ needed."""
    manifest = files("supermimic.data").joinpath("motion_subsets/khan_mixed480.txt")
    names = [line for line in manifest.read_text().splitlines() if line and not line.startswith("#")]
    if len(names) != 480 or len(set(names)) != 480:
        raise ValueError("The built-in mixed480 manifest must contain 480 unique names")
    return tuple(_name(n) for n in names)


def _select(clips, *, subset, motion_list_dir, names):
    if sum(v is not None for v in (subset, motion_list_dir, names)) != 1:
        raise ValueError("Choose exactly one of subset, motion_list_dir or names")
    available = {name for name, _, _ in clips}
    if subset is not None:
        if subset != "mixed480":
            raise ValueError(f"Unknown subset: {subset}")
        requested = list(mixed480_names())
    elif motion_list_dir is not None:
        directory = Path(motion_list_dir)
        if not directory.is_dir():
            raise ValueError(f"Motion-list directory does not exist: {directory}")
        stems = sorted(p.stem for p in directory.glob("*.npz") if p.is_file())
        if not stems:
            raise ValueError("Motion-list directory contains no NPZ clips")
        requested = []
        for stem in stems:
            candidates = {n for n in (stem, f"mixed__{stem}", f"lafan1__{stem}") if n in available}
            if len(candidates) != 1:
                raise ValueError(f"Missing or ambiguous motion {stem!r}: {sorted(candidates)}")
            requested.append(candidates.pop())
    else:
        requested = [_name(n) for n in names]
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Subset names must be nonempty and unique")
    missing = set(requested)-available
    if missing:
        raise ValueError(f"Motions missing from packed library: {sorted(missing)}")
    selected = set(requested)
    return [clip for clip in clips if clip[0] in selected]


def _sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def subset_library(library: Path, out: Path, *, subset: str | None = None,
                   motion_list_dir: Path | None = None, names: list[str] | None = None,
                   reuse_existing: bool = False, chunk_size: int = 8192) -> dict:
    """Copy selected clips in packed order, preserving exact tensor dtype/values.

    Existing output is protected. Opt-in reuse validates source hash, selection,
    metadata and *all* output frames against the current source before accepting
    it, so launchers can safely repeat a subset preparation command.
    """
    library, out = Path(library).resolve(), Path(out).resolve()
    if library == out:
        raise ValueError("Subset output must differ from the source library")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if out.suffix != ".pt":
        raise ValueError("Subset output must be a .pt file")
    if out.exists() and not reuse_existing:
        raise FileExistsError(f"Output exists; choose a new path or --reuse-existing: {out}")
    pack, clips = _load_pack(library)
    if subset == "mixed480" and pack.get("robot", "g1") != "khan_mini":
        raise ValueError("The built-in mixed480 subset requires Khan_mini")
    selected = _select(clips, subset=subset, motion_list_dir=motion_list_dir, names=names)
    starts, lengths, offset = [], [], 0
    for _, _, length in selected:
        starts.append(offset)
        lengths.append(length)
        offset += length
    source_sha = _sha256(library)
    selected_names = [name for name, _, _ in selected]
    provenance = {
        "version": 1, "parent_library": str(library), "parent_sha256": source_sha,
        "parent_clips": len(clips), "parent_frames": len(pack["ref"]),
        "selection": subset or ("motion_list_dir" if motion_list_dir is not None else "exact_names"),
        "motion_list_dir": str(Path(motion_list_dir).resolve()) if motion_list_dir is not None else None,
        "selected_names_sha256": hashlib.sha256(json.dumps(selected_names, separators=(",", ":")).encode()).hexdigest(),
        "parent_source_inventory_sha256": pack.get("source_inventory_sha256"),
        "operation": "Exact stored frame copy; no FK/resampling/quantization/grounding/clamping",
        "note": "mixed480 identifies motions by name; LAFAN values remain the versions in the supplied PT, not the old NPZ values.",
    }
    existing = None
    if out.exists():
        existing, existing_clips = _load_pack(out)
        if (existing.get("robot", "g1") != pack.get("robot", "g1")
                or existing["ref"].dtype != pack["ref"].dtype
                or existing_clips != list(zip(selected_names, starts, lengths, strict=True))):
            raise ValueError("Existing subset has incompatible robot/dtype/clip layout")
        old = existing.get("subset_provenance", {})
        if any(old.get(k) != provenance[k] for k in ("version", "parent_sha256", "selected_names_sha256")):
            raise ValueError("Existing subset provenance differs from the requested source/selection")
        target_ref = existing["ref"]
    else:
        target_ref = torch.empty((offset, pack["ref"].shape[1]), dtype=pack["ref"].dtype)
    with torch.no_grad():
        for (_, source_start, length), target_start in zip(selected, starts, strict=True):
            for local in range(0, length, chunk_size):
                n = min(chunk_size, length-local)
                chunk = pack["ref"][source_start+local:source_start+local+n]
                if not bool(torch.isfinite(chunk).all()):
                    raise ValueError("Selected packed references contain NaN/Inf")
                if not bool(((chunk[:, 3:7].float().norm(dim=-1)-1).abs() <= .01).all()):
                    raise ValueError("Selected packed root quaternions must be near unit length")
                destination = target_ref[target_start+local:target_start+local+n]
                if existing is not None:
                    if not torch.equal(destination, chunk):
                        raise ValueError("Existing subset frames differ from the source library")
                else:
                    destination.copy_(chunk)
    if existing is None:
        result = {k: v for k, v in pack.items()
                  if k not in ("ref", "names", "starts", "lengths", "source_inventory_sha256", "subset_provenance")}
        result.update(ref=target_ref, names=selected_names, starts=starts, lengths=lengths,
                      subset_provenance=provenance)
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=out.parent, prefix=f".{out.name}.", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                torch.save(result, stream)
            # Atomic publication; never overwrite an output created concurrently.
            os.link(temporary, out)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return {"status": "reused_verified" if existing is not None else "created", "library": str(library),
            "out": str(out), "source_sha256": source_sha, "selected_clips": len(selected),
            "selected_frames": offset, "dtype": str(target_ref.dtype), "subset": provenance["selection"],
            "names": selected_names}
