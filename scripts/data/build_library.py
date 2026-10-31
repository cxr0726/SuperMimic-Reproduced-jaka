"""Pack named 50 Hz NPZ clips into a compact robot-specific fp16 library.

  uv run python scripts/data/build_library.py --npz-dir data/npz/bones_seed --manifest data/npz/bones_seed/manifest_filtered.csv --out data/library/bones_seed_compact.pt
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from tqdm import tqdm

from supermimic.envs.mdp.motion_library import discover_npz, read_npz_reference
from supermimic.robots.registry import get_robot_spec


def main(npz_dir: Path, out: Path, manifest: Path | None = None, workers: int = 16,
         robot: Literal["g1", "khan_mini"] = "g1") -> None:
    spec = get_robot_spec(robot)
    files = discover_npz(npz_dir, manifest=manifest)
    if not files:
        raise ValueError(f"No NPZ clips found under {npz_dir}")
    if workers < 1:
        raise ValueError("workers must be positive")
    print(f"{len(files)} clips")
    from concurrent.futures import ThreadPoolExecutor

    def load(f):
        ref, _ = read_npz_reference(f, robot_name=robot, root_body_name=spec.root_body,
                                    joint_names=spec.joint_names, body_names=spec.tracked_bodies)
        return f.stem, ref.astype(np.float16)

    with ThreadPoolExecutor(workers) as ex:
        items = list(tqdm(ex.map(load, files), total=len(files)))
    names = [n for n, _ in items]
    lengths = [int(r.shape[0]) for _, r in items]
    starts = np.cumsum([0] + lengths[:-1]).tolist()
    ref = torch.from_numpy(np.concatenate([r for _, r in items], 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ref": ref, "names": names, "starts": starts, "lengths": lengths,
                "robot": robot, "joint_names": spec.joint_names,
                "root_body_name": spec.root_body, "fps": 50.0}, out)
    print(f"saved {out}: {ref.shape[0]} frames ({ref.shape[0]/50/3600:.2f} h), {ref.numel()*2/1e9:.2f} GB fp16")


if __name__ == "__main__":
    tyro.cli(main)
