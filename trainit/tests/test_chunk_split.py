"""Tests for the chunked train/val split."""

from pathlib import Path

from trainit.chunk_split import chunked_split, frame_number


def _frames(seq, numbers):
    return [Path(f"/ds/{seq}/images/train/frame_{n:06d}.jpg") for n in numbers]


def test_frame_number_from_trailing_digits():
    assert frame_number(Path("frame_000120.jpg")) == 120
    assert frame_number(Path("M0101_00042.png")) == 42
    assert frame_number(Path("noframe.jpg")) is None


def test_val_is_whole_chunks_with_gap_excluded():
    files = _frames("a", range(0, 1000, 10))  # 100 images, 10 chunks of 100 frames
    split = chunked_split({"a": files}, train_ratio=0.9, seed=3, chunk_frames=100, gap_frames=20)

    val_chunks = {frame_number(f) // 100 for f in split.val}
    assert len(val_chunks) == 1 and len(split.val) == 10
    (chunk,) = val_chunks
    lo, hi = chunk * 100, chunk * 100 + 99
    for f in split.train:
        n = frame_number(f)
        assert n < lo - 20 or n > hi + 20
    for f in split.excluded:
        n = frame_number(f)
        assert lo - 20 <= n < lo or hi < n <= hi + 20
    assert len(split.train) + len(split.val) + len(split.excluded) == len(files)


def test_split_is_reproducible_and_seed_dependent():
    files = _frames("a", range(0, 2000, 10))
    a = chunked_split({"a": files}, 0.8, seed=1, chunk_frames=100)
    b = chunked_split({"a": list(reversed(files))}, 0.8, seed=1, chunk_frames=100)
    c = chunked_split({"a": files}, 0.8, seed=2, chunk_frames=100)
    assert sorted(a.val) == sorted(b.val)
    assert sorted(a.val) != sorted(c.val)


def test_every_sequence_contributes_val():
    sequences = {s: _frames(s, range(0, 1000, 10)) for s in ("a", "b")}
    split = chunked_split(sequences, 0.9, seed=0, chunk_frames=100)
    assert {f.parts[2] for f in split.val} == {"a", "b"}


def test_single_chunk_sequences_stay_in_train_or_fall_back():
    short = {"a": _frames("a", range(0, 50, 10))}
    assert chunked_split(short, 0.9, seed=0, chunk_frames=100) is None

    mixed = {"a": _frames("a", range(0, 50, 10)), "b": _frames("b", range(0, 1000, 10))}
    split = chunked_split(mixed, 0.9, seed=0, chunk_frames=100)
    assert all(f.parts[2] == "b" for f in split.val)
    assert set(mixed["a"]) <= set(split.train)
