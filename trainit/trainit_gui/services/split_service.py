"""Service for train/val splitting with stratification."""

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """Result of a train/val split operation."""

    train_files: list[Path] = field(default_factory=list)
    val_files: list[Path] = field(default_factory=list)
    method: str = "random"  # "stratified" or "random"
    sequences_found: int = 0


class SplitService:
    """Service for stratified train/val splitting.

    Adapts the algorithm from split_train_val.py for use in the GUI.
    Tries stratified splitting by sequence when possible, falls back to random.
    """

    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def extract_sequence_id(self, filename: str) -> str:
        """Extract sequence ID from filename.

        Uses prefix-underscore pattern: "M0101_00001.jpg" -> "M0101"

        Args:
            filename: Image filename

        Returns:
            Sequence ID (part before first underscore, or whole stem if no underscore)
        """
        stem = Path(filename).stem
        parts = stem.split('_')
        if len(parts) >= 2:
            return parts[0]
        return stem

    def collect_files_from_dataset(
        self,
        dataset_path: Path,
        include_val: bool
    ) -> list[Path]:
        """Collect all image files from a dataset.

        Args:
            dataset_path: Path to dataset root
            include_val: Whether to include val folder in pool

        Returns:
            List of image file paths
        """
        files = []

        # Always include train
        train_dir = dataset_path / "images" / "train"
        if train_dir.exists():
            files.extend(self._list_images(train_dir))
            logger.debug(f"Collected {len(files)} images from {train_dir}")

        # Optionally include val
        if include_val:
            val_dir = dataset_path / "images" / "val"
            if val_dir.exists():
                val_files = self._list_images(val_dir)
                files.extend(val_files)
                logger.debug(f"Collected {len(val_files)} images from {val_dir}")

        return files

    def _list_images(self, directory: Path) -> list[Path]:
        """List all image files in directory."""
        return [
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in self.IMAGE_EXTENSIONS
        ]

    def split_files(
        self,
        files: list[Path],
        train_ratio: float,
        seed: int
    ) -> SplitResult:
        """Split files into train/val sets.

        Attempts stratified split by sequence first, falls back to random.

        Args:
            files: List of image file paths
            train_ratio: Fraction for training (e.g., 0.9)
            seed: Random seed for reproducibility

        Returns:
            SplitResult with train/val file lists and method used
        """
        if not files:
            return SplitResult()

        random.seed(seed)

        # Group by sequence
        sequences = self._group_by_sequence(files)
        num_sequences = len(sequences)

        logger.debug(f"Found {num_sequences} sequences in {len(files)} files")

        # Determine if stratification is meaningful
        # (at least 2 sequences with reasonable distribution)
        if num_sequences >= 2:
            return self._stratified_split(sequences, train_ratio, seed)
        else:
            return self._random_split(files, train_ratio, seed)

    def _group_by_sequence(self, files: list[Path]) -> dict[str, list[Path]]:
        """Group files by sequence ID."""
        sequences = defaultdict(list)
        for f in files:
            seq_id = self.extract_sequence_id(f.name)
            sequences[seq_id].append(f)
        return dict(sequences)

    def _stratified_split(
        self,
        sequences: dict[str, list[Path]],
        train_ratio: float,
        seed: int
    ) -> SplitResult:
        """Split maintaining representation from each sequence.

        For each sequence, samples proportionally to ensure both train and val
        contain images from all sequences.
        """
        random.seed(seed)

        train_files = []
        val_files = []

        for seq_id, files in sequences.items():
            shuffled = list(files)
            random.shuffle(shuffled)

            # Calculate split, ensuring at least 1 in train if possible
            n_train = max(1, int(len(files) * train_ratio))
            if n_train >= len(files):
                n_train = len(files) - 1 if len(files) > 1 else len(files)

            train_files.extend(shuffled[:n_train])
            val_files.extend(shuffled[n_train:])

        logger.info(
            f"Stratified split: {len(train_files)} train, {len(val_files)} val "
            f"across {len(sequences)} sequences"
        )

        return SplitResult(
            train_files=train_files,
            val_files=val_files,
            method="stratified",
            sequences_found=len(sequences)
        )

    def _random_split(
        self,
        files: list[Path],
        train_ratio: float,
        seed: int
    ) -> SplitResult:
        """Pure random split when stratification is not applicable."""
        random.seed(seed)

        shuffled = list(files)
        random.shuffle(shuffled)

        n_train = int(len(shuffled) * train_ratio)

        logger.info(
            f"Random split: {n_train} train, {len(shuffled) - n_train} val "
            f"(no sequence pattern detected)"
        )

        return SplitResult(
            train_files=shuffled[:n_train],
            val_files=shuffled[n_train:],
            method="random",
            sequences_found=1
        )
