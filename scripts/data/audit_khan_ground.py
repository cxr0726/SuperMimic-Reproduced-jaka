"""Audit converted Khan foot collision/visual geometry against the world z=0 plane."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from supermimic.data.foot_clearance import audit_npz

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--npz", type=Path, help="Audit one motion")
    source.add_argument("--npz-dir", type=Path, help="Audit every NPZ in a directory")
    parser.add_argument("--asset-dir", type=Path)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--tolerance-m", type=float, default=0.001)
    parser.add_argument("--output", type=Path, default=ROOT / "logs/khan_replay/ground_audit.json")
    parser.add_argument("--per-frame-csv", type=Path, help="Optional CSV with one row per evaluated frame")
    args = parser.parse_args()
    if args.stride < 1 or args.tolerance_m < 0:
        parser.error("stride must be positive and tolerance-m nonnegative")
    directory = args.npz_dir or ROOT / "data/npz/khan_mini/lafan1"
    paths = [args.npz] if args.npz is not None else sorted(directory.glob("*.npz"))
    if not paths:
        parser.error(f"No NPZ motions found in {directory}")
    stream = None
    writer = None
    rows = []
    try:
        if args.per_frame_csv is not None:
            args.per_frame_csv.parent.mkdir(parents=True, exist_ok=True)
            stream = args.per_frame_csv.open("w", newline="")
        for i, path in enumerate(paths, 1):
            report = audit_npz(path, args.asset_dir, tolerance_m=args.tolerance_m, stride=args.stride)
            values = report.pop("_per_frame")
            if stream is not None:
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=["clip", *values])
                    writer.writeheader()
                for frame in range(report["evaluated_frames"]):
                    writer.writerow({"clip": report["clip"],
                                     **{key: value[frame] for key, value in values.items()}})
            rows.append(report)
            bottom = report["combined"]["collision"]
            print(f"[{i}/{len(paths)}] {report['clip']}: collision min {bottom['min_z_m']:.6f} m, "
                  f"below tolerance {bottom['below_tolerance_frames']}/{report['evaluated_frames']}", flush=True)
    finally:
        if stream is not None:
            stream.close()
    report = {
        "status": "audited", "ground_z_m": 0.0, "tolerance_m": args.tolerance_m,
        "stride": args.stride, "clip_count": len(rows),
        "total_frames": sum(row["total_frames"] for row in rows),
        "evaluated_frames": sum(row["evaluated_frames"] for row in rows),
        "per_frame_csv": str(args.per_frame_csv.resolve()) if args.per_frame_csv is not None else None,
        "summary": {}, "clips": rows,
    }
    for kind in ("collision", "visual"):
        worst = min(rows, key=lambda row: row["combined"][kind]["min_z_m"])
        count = sum(row["combined"][kind]["below_tolerance_frames"] for row in rows)
        report["summary"][kind] = {
            "min_z_m": worst["combined"][kind]["min_z_m"], "worst_clip": worst["clip"],
            "worst_frame": worst["combined"][kind]["min_frame"],
            "worst_time_s": worst["combined"][kind]["min_time_s"],
            "below_tolerance_frames": count,
            "below_tolerance_fraction": count / report["evaluated_frames"],
            "clips_with_below_tolerance_frames": sum(row["combined"][kind]["below_tolerance_frames"] > 0 for row in rows),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"[DONE] ground audit: {args.output}")


if __name__ == "__main__":
    main()
