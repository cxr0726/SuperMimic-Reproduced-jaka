"""Time actual prepared replay samplers without allocating full robot trajectories.

The buffer has scalar state/action/context fields; legality and sampling use the
real implementation and configured E x T storage. This is not a simulator or
end-to-end training benchmark.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from supermimic.data.replay_buffer import ReplayBuffer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--capacity", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        parser.error("This benchmark requires CUDA")
    if args.num_envs < 1 or args.capacity < 32 or args.samples < 1:
        parser.error("num-envs/samples must be positive, capacity at least 32")
    torch.cuda.set_device(torch.device(args.device))
    torch.manual_seed(0)
    buffer = ReplayBuffer(args.num_envs, args.capacity, 1, 1, 1, args.device)
    buffer.valid.fill_(True)
    buffer.valid[:, ::127] = False  # Some windows encounter holes.
    buffer.ep_id.zero_()
    buffer.frame.copy_(torch.arange(args.capacity, device=args.device)[None])
    clip_last = torch.tensor([args.capacity + 100], device=args.device)

    def prepare():
        windows = buffer.prepare_windows(8)
        starts = buffer.prepare_starts(5, horizon=32, clip_last=clip_last)
        return windows, starts

    # Allocate both fixed selections before timing; exclude cold allocations.
    prepare()
    report = {"gpu": torch.cuda.get_device_name(torch.device(args.device)),
              "scope": "sampler only; actual mask preparation and sampling; scalar replay fields",
              "num_envs": args.num_envs, "capacity": args.capacity,
              "wm_draws": 10, "wm_batch": 2048, "actor_batch": 4096,
              "cuda_graph": True, "modes": []}
    for sampling in ("multinomial", "cdf"):
        buffer.sampling = sampling
        # Warm both mask generation and the respective CDF/reduction kernels.
        for _ in range(3):
            buffer._write_revision += 1
            windows, starts = prepare()
        preparation = []
        for _ in range(10):
            buffer._write_revision += 1
            torch.cuda.synchronize()
            begin = time.perf_counter()
            windows, starts = prepare()
            torch.cuda.synchronize()
            preparation.append((time.perf_counter() - begin) * 1000)

        def draw(windows=windows, starts=starts):
            for _ in range(10):
                buffer._sample_from_selection(windows, 2048)
            return buffer._sample_from_selection(starts, 4096)

        stream = torch.cuda.Stream(device=args.device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                draw()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            env, slot = draw()
        for _ in range(5):
            graph.replay()
        timings = []
        for _ in range(args.samples):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            graph.replay()
            end.record()
            end.synchronize()
            timings.append(begin.elapsed_time(end))
        assert bool((starts.weights[env * buffer.T + slot] > 0).all())
        row = {"sampling": sampling, "prepare_both_masks_median_ms": statistics.median(preparation),
               "draw_block_median_ms": statistics.median(timings), "draw_samples_ms": timings,
               "valid_windows": int(windows.count), "valid_starts": int(starts.count),
               "cdf_storage_bytes": windows.cdf.nbytes + starts.cdf.nbytes}
        report["modes"].append(row)
        print(json.dumps(row), flush=True)
        del graph, env, slot
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
