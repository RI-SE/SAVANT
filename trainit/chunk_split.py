"""Chunked train/val split for frames extracted from video.

Random per-frame splitting puts near-identical neighbouring frames in both train and
val. Instead, each sequence is cut into fixed chunks of frame numbers; whole chunks go
to val, and train frames within a gap of a val chunk are excluded from both sets.
Frame numbers come from the trailing number in the filename (e.g. frame_000120.jpg,
M0101_00042.jpg), so chunk and gap sizes are in those units.
"""

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# 20 s chunks and 4 s gaps at 30 fps (frame numbers are video frames for extracted data)
DEFAULT_CHUNK_FRAMES = 600
DEFAULT_GAP_FRAMES = 120

_TRAILING_NUMBER = re.compile(r"(\d+)$")


@dataclass
class ChunkSplit:
    train: list[Path] = field(default_factory=list)
    val: list[Path] = field(default_factory=list)
    excluded: list[Path] = field(default_factory=list)  # gap frames, in neither set


def frame_number(path: Path) -> Optional[int]:
    """Return the trailing number of the filename stem, or None."""
    match = _TRAILING_NUMBER.search(Path(path).stem)
    return int(match.group(1)) if match else None


def _frame_numbers(files: list[Path]) -> dict[Path, int]:
    """Frame number per file; falls back to sorted position if any name lacks a number."""
    numbers = {f: frame_number(f) for f in files}
    if None in numbers.values():
        return {f: i for i, f in enumerate(sorted(files))}
    return numbers


def _split_sequence(
    files: list[Path], val_ratio: float, rng: random.Random, chunk_frames: int, gap_frames: int
) -> ChunkSplit:
    numbers = _frame_numbers(files)
    chunks: dict[int, list[Path]] = {}
    for f in sorted(files):
        chunks.setdefault(numbers[f] // chunk_frames, []).append(f)
    if len(chunks) < 2:
        # A single chunk can't be held out without leaking; keep it all in train
        return ChunkSplit(train=sorted(files))

    target = max(1, round(len(files) * val_ratio))
    order = sorted(chunks)
    rng.shuffle(order)
    val_chunks: list[int] = []
    val_count = 0
    for chunk_id in order[:-1]:  # always leave at least one chunk for train
        if val_count >= target:
            break
        val_chunks.append(chunk_id)
        val_count += len(chunks[chunk_id])

    val_ranges = [(c * chunk_frames, (c + 1) * chunk_frames - 1) for c in val_chunks]
    result = ChunkSplit()
    for chunk_id, chunk_files in chunks.items():
        if chunk_id in val_chunks:
            result.val.extend(chunk_files)
            continue
        for f in chunk_files:
            n = numbers[f]
            near_val = any(start - gap_frames <= n <= end + gap_frames for start, end in val_ranges)
            (result.excluded if near_val else result.train).append(f)
    return result


def chunked_split(
    sequences: dict[str, list[Path]],
    train_ratio: float,
    seed: int,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
    gap_frames: int = DEFAULT_GAP_FRAMES,
) -> Optional[ChunkSplit]:
    """Split each sequence by chunks; return None if no sequence spans two chunks."""
    rng = random.Random(seed)
    combined = ChunkSplit()
    for seq_id in sorted(sequences):
        part = _split_sequence(sequences[seq_id], 1.0 - train_ratio, rng, chunk_frames, gap_frames)
        combined.train.extend(part.train)
        combined.val.extend(part.val)
        combined.excluded.extend(part.excluded)
    return combined if combined.val else None
