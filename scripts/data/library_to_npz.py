"""Expand a compact .pt library into full 50 Hz training NPZ clips."""

from __future__ import annotations

import argparse
from pathlib import Path

from supermimic.data.library_to_npz import export_library


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--robot", choices=("g1", "khan_mini"), default="khan_mini")
    parser.add_argument("--device", default="cpu", help="cpu (portable default) or cuda:0")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--clips", nargs="+", help="Exact clip names; selected in packed order")
    parser.add_argument("--limit", type=int, help="Export only the first N clips after name filtering")
    parser.add_argument("--all-bodies", action="store_true", help="Store all robot links instead of tracked links")
    args = parser.parse_args()
    report = export_library(args.library, args.out_dir, robot=args.robot, device=args.device,
                            chunk_size=args.chunk_size, clips=args.clips, limit=args.limit,
                            all_bodies=args.all_bodies)
    print(f"[PASS] {report['exported_clips']} clips / {report['exported_frames']} frames -> {args.out_dir}")


if __name__ == "__main__":
    main()
