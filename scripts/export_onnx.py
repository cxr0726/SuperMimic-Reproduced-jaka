"""Export the current Khan actor and robot metadata, then verify with ONNX Runtime."""

from pathlib import Path

import tyro

from supermimic.deployment.onnx_export import InputLayout, export_policy


def main(ckpt: Path, out: Path, input_layout: InputLayout = "packed",
         validation_samples: int = 64, motion: Path | None = None,
         overwrite: bool = False) -> None:
    """--motion adds NPZ-derived parity checks; it does not embed the motion."""
    report = export_policy(ckpt, out, input_layout, validation_samples, motion, overwrite)
    validation = report["validation"]
    print(f"Saved {report['onnx']} (iteration {report['iteration']}, {input_layout}, FP32)")
    print(f"ONNX/PyTorch max action error {validation['max_action_abs_error']:.3e}; "
          f"PD target error {validation['max_pd_target_abs_error_rad']:.3e} rad")
    print(f"Metadata and validation report: {Path(report['onnx']).with_suffix('.json')}")


if __name__ == "__main__":
    tyro.cli(main)
