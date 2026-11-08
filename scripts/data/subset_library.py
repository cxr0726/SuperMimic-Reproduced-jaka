"""Extract named clips from a compact PT while preserving exact stored frames."""

from __future__ import annotations

import argparse
from pathlib import Path

from supermimic.data.packed_subset import subset_library


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--subset", choices=("mixed480",), help="Audited 444 mixed + 36 LAFAN names; source NPZ not needed")
    selection.add_argument("--motion-list-dir", type=Path, help="Select motions matching NPZ filenames; NPZ contents are not read")
    selection.add_argument("--clips", nargs="+", help="Exact packed motion names")
    parser.add_argument("--reuse-existing", action="store_true", help="Accept existing output only after full source/provenance/frame verification")
    parser.add_argument("--chunk-size", type=int, default=8192)
    args = parser.parse_args()
    report = subset_library(args.library, args.out, subset=args.subset,
                            motion_list_dir=args.motion_list_dir, names=args.clips,
                            reuse_existing=args.reuse_existing, chunk_size=args.chunk_size)
    print(f"[PASS] {report['status']}: {report['selected_clips']} clips / "
          f"{report['selected_frames']} frames -> {report['out']}")
    if args.subset == "mixed480":
        print("[INFO] Motion-name subset only: LAFAN frames are the supplied PT versions, not the old mixed NPZ values.")


if __name__ == "__main__":
    main()
