#!/usr/bin/env python3
"""
benchmark_video_reader.py

Benchmarks sequential video decode speed and estimates whether the VideoReader
chunk-prefetch cache can keep up with real-time playback without freezing.

Usage:
    python utils/benchmark_video_reader.py <video_path> [--chunk-size N] [--start-frame N] [--frames N]

Example:
    python utils/benchmark_video_reader.py /path/to/video.mp4
    python utils/benchmark_video_reader.py /path/to/video.mp4 --chunk-size 16 --frames 120
"""

import argparse
import sys
import time

try:
    import cv2
except ImportError:
    print("ERROR: opencv-contrib-python is required. Run: pip install opencv-contrib-python")
    sys.exit(1)


def benchmark(video_path: str, chunk_size: int, start_frame: int, num_frames: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        sys.exit(1)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo : {video_path}")
    print(f"Size  : {width}x{height}  |  FPS: {fps:.2f}  |  Total frames: {total}")
    print(f"Test  : {num_frames} frames from frame {start_frame}  |  Chunk size: {chunk_size}\n")

    frame_ms   = 1000.0 / fps
    budget_ms  = (chunk_size // 2) * frame_ms   # time available while consuming first half of chunk

    # ── Single-frame random-access cost (seek overhead) ──────────────────────
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    seeks = min(20, total - start_frame)
    t0 = time.perf_counter()
    for i in range(seeks):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i * 5)
        cap.read()
    seek_ms = (time.perf_counter() - t0) * 1000 / seeks

    # ── Sequential chunk decode ───────────────────────────────────────────────
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    chunk_times = []
    decoded = 0
    while decoded < num_frames:
        n = min(chunk_size, num_frames - decoded)
        t0 = time.perf_counter()
        for _ in range(n):
            ok, _ = cap.read()
            if not ok:
                break
        chunk_ms = (time.perf_counter() - t0) * 1000
        chunk_times.append(chunk_ms)
        decoded += n

    cap.release()

    avg_chunk_ms = sum(chunk_times) / len(chunk_times)
    max_chunk_ms = max(chunk_times)
    min_chunk_ms = min(chunk_times)
    per_frame_ms = avg_chunk_ms / chunk_size

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"{'Metric':<40} {'Value':>10}")
    print("-" * 52)
    print(f"{'Single-frame seek+decode (avg)':<40} {seek_ms:>9.1f}ms")
    print(f"{'Sequential decode per frame (avg)':<40} {per_frame_ms:>9.1f}ms")
    print(f"{'Chunk decode time — avg':<40} {avg_chunk_ms:>9.1f}ms")
    print(f"{'Chunk decode time — min':<40} {min_chunk_ms:>9.1f}ms")
    print(f"{'Chunk decode time — max':<40} {max_chunk_ms:>9.1f}ms")
    print(f"{'Playback frame interval':<40} {frame_ms:>9.1f}ms")
    print(f"{'Prefetch budget (half-chunk display time)':<40} {budget_ms:>9.1f}ms")
    print()

    ok_avg = avg_chunk_ms < budget_ms
    ok_max = max_chunk_ms < budget_ms
    status_avg = "✓  OK" if ok_avg else "✗  TOO SLOW"
    status_max = "✓  OK" if ok_max else "✗  RISK OF FREEZE"

    print(f"Prefetch vs budget (avg): {status_avg}  ({avg_chunk_ms:.0f}ms vs {budget_ms:.0f}ms budget)")
    print(f"Prefetch vs budget (max): {status_max}  ({max_chunk_ms:.0f}ms vs {budget_ms:.0f}ms budget)")

    if not ok_avg:
        needed = int(avg_chunk_ms / frame_ms * 2) + 2
        print(f"\n  → Suggested minimum chunk_size to avoid freezes: {needed} frames")
        print(f"    Memory for 3 chunks of {needed} frames at {width}x{height}:")
        mb_per_frame = width * height * 3 / 1024 / 1024
        print(f"    ~{mb_per_frame * needed * 3:.0f} MB")
    elif not ok_max:
        print("\n  → Average is fine but occasional slow decodes may cause brief stutters.")
        print("    Consider increasing chunk_size slightly for headroom.")
    else:
        print(f"\n  → chunk_size={chunk_size} is sufficient for smooth playback on this machine.")
        mb_per_frame = width * height * 3 / 1024 / 1024
        print(f"    3-chunk cache memory usage: ~{mb_per_frame * chunk_size * 3:.0f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark VideoReader decode speed vs real-time playback budget."
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--chunk-size", type=int, default=30,
                        help="Frames per chunk (default: 30, same as VideoReader default)")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Frame index to start benchmarking from (default: 0)")
    parser.add_argument("--frames", type=int, default=150,
                        help="Total frames to benchmark (default: 150 = 5s at 30fps)")
    args = parser.parse_args()

    benchmark(args.video, args.chunk_size, args.start_frame, args.frames)


if __name__ == "__main__":
    main()
