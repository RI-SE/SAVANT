import threading
import cv2
import numpy as np
from .exceptions import VideoLoadError, VideoFrameIndexError, VideoReadError
from .project_state import ProjectState


class VideoReader:
    """Wrapper around OpenCV VideoCapture for frame iteration and random access.

    Frames are loaded in chunks and two chunks are kept in memory.  A background
    thread prefetches the next chunk so that sequential playback (and tracking)
    never blocks waiting for a decode.
    """

    DEFAULT_CHUNK_SIZE = 30  # frames per chunk (~1 second at 30fps)

    def __init__(self, project_state: ProjectState, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
        """Initialize the video reader."""
        self.capture: cv2.VideoCapture = None
        self._prefetch_capture: cv2.VideoCapture = None  # dedicated to prefetch thread
        self._video_path: str | None = None
        self.project_state = project_state
        self.chunk_size = chunk_size
        # Three chunk slots: previous, current, next (prefetched)
        self._chunks: list = [None, None, None]
        self._chunk_ages: list = [0, 0, 0]
        self._age_counter: int = 0
        self._last_index: int = -1
        # Prefetch state
        self._prefetch_lock = threading.Lock()
        self._prefetch_result: tuple | None = None  # (chunk_start, frames_dict)
        self._prefetch_thread: threading.Thread | None = None

    def load_video(self, path: str) -> None:
        self._cancel_prefetch()
        if self.capture and self.capture.isOpened():
            self.release()
        self._video_path = path
        self.capture = cv2.VideoCapture(path)
        if not self.capture.isOpened():
            raise VideoLoadError(f"Could not open video file from path: {path}")
        self._prefetch_capture = cv2.VideoCapture(path)
        self._chunks = [None, None, None]
        self._chunk_ages = [0, 0, 0]
        self._age_counter = 0
        self._last_index = -1
        self._prefetch_result = None

        if self.project_state:
            self.project_state.video_metadata.frame_count = int(
                self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
            )
            self.project_state.video_metadata.width = int(
                self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            )
            self.project_state.video_metadata.height = int(
                self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )
            self.project_state.video_metadata.fps = self.capture.get(cv2.CAP_PROP_FPS)

    @property
    def current_index(self) -> int:
        """Index of the most recently returned frame, or -1 if no frame has been read yet."""
        return self._last_index

    def __iter__(self):
        """Return self as iterator."""
        return self

    def __next__(self) -> np.ndarray:
        """Return the next frame via the chunk cache, or raise StopIteration at end."""
        self._validate_video_loaded()
        next_index = self._last_index + 1
        frame_count = self.project_state.video_metadata.frame_count
        if next_index >= frame_count:
            raise StopIteration
        return self.get_frame(next_index)

    def skip_frames(self, delta: int) -> np.ndarray:
        """
        Move delta frames from current position, clamped to [0, frame_count-1]
        Returns the frame at the new position.
        """
        self._validate_video_loaded()
        cur = max(self.current_index, 0)
        last = self.project_state.video_metadata.frame_count - 1
        target = min(max(cur + delta, 0), last)
        return self.get_frame(target)

    def get_frame(self, index: int) -> np.ndarray:
        """
        Retrieve a specific frame by index.

        Frames are loaded in chunks of ``chunk_size`` and three chunks are kept
        in memory (previous, current, prefetched next).  Sequential access
        therefore avoids per-frame seeks after the initial seek at the start of
        each chunk, and the next chunk is decoded in the background so there is
        no freeze at chunk boundaries.

        Args:
            index (int): Frame index (0-based).

        Returns:
            np.ndarray: The requested frame.

        Raises:
            VideoFrameIndexError: If the index is out of range.
            VideoReadError: If the frame cannot be decoded.
        """
        self._validate_video_loaded()
        frame_count = self.project_state.video_metadata.frame_count
        if not (0 <= index < frame_count):
            raise VideoFrameIndexError("Frame index out of range")

        # Absorb any finished prefetch into the chunk slots
        self._absorb_prefetch()

        # Check existing chunks
        for slot, entry in enumerate(self._chunks):
            if entry is not None:
                start, frames_dict = entry
                if index in frames_dict:
                    self._chunk_ages[slot] = self._age_counter
                    self._age_counter += 1
                    self._last_index = index
                    self._maybe_prefetch(index)
                    return frames_dict[index]

        # Cache miss — if the prefetch thread is loading exactly this chunk, wait for it
        chunk_start = (index // self.chunk_size) * self.chunk_size
        with self._prefetch_lock:
            prefetch_thread = self._prefetch_thread
            _ = self._prefetch_result[0] if self._prefetch_result else None

        if prefetch_thread and prefetch_thread.is_alive():
            # Check if it's loading the chunk we need by inspecting what chunk_start it was given.
            # We store the target in _prefetch_target for this purpose.
            if getattr(self, '_prefetch_target', None) == chunk_start:
                prefetch_thread.join()
                self._absorb_prefetch()
                # Now check cache again
                for slot, entry in enumerate(self._chunks):
                    if entry is not None and index in entry[1]:
                        self._chunk_ages[slot] = self._age_counter
                        self._age_counter += 1
                        self._last_index = index
                        self._maybe_prefetch(index)
                        return entry[1][index]

        # Synchronous load — also read the next chunk while the capture is
        # already positioned at chunk_end (avoids I/O contention with the
        # background prefetch thread which uses a second VideoCapture handle).
        frames_dict = self._load_chunk(self.capture, chunk_start, frame_count)

        if index not in frames_dict:
            raise VideoReadError(f"Failed to read frame {index}")

        # Replace the oldest chunk slot
        oldest_slot = self._chunk_ages.index(min(self._chunk_ages))
        self._chunks[oldest_slot] = (chunk_start, frames_dict)
        self._age_counter += 1
        self._chunk_ages[oldest_slot] = self._age_counter

        # Read ahead: the capture is now at chunk_end, read next chunk too
        next_start = chunk_start + self.chunk_size
        if next_start < frame_count:
            ahead_dict = self._load_chunk(self.capture, next_start, frame_count)
            if ahead_dict:
                oldest_slot = self._chunk_ages.index(min(self._chunk_ages))
                self._chunks[oldest_slot] = (next_start, ahead_dict)
                self._age_counter += 1
                self._chunk_ages[oldest_slot] = self._age_counter

        # Position prefetch capture where the main capture left off so
        # the next background prefetch can read sequentially without seeking.
        read_end = next_start + self.chunk_size if next_start < frame_count else chunk_start + self.chunk_size
        if self._prefetch_capture and read_end < frame_count:
            self._prefetch_capture.set(cv2.CAP_PROP_POS_FRAMES, read_end)

        self._last_index = index
        self._maybe_prefetch(index)
        return frames_dict[index]

    def _load_chunk(self, cap: cv2.VideoCapture, chunk_start: int, frame_count: int) -> dict:
        """Seek cap to chunk_start and read chunk_size frames. Returns {idx: frame}.

        Seeks only when the capture isn't already at the correct position,
        avoiding expensive keyframe-based seeking during sequential reads.
        """
        chunk_end = min(chunk_start + self.chunk_size, frame_count)
        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_pos != chunk_start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, chunk_start)
        frames = {}
        for i in range(chunk_start, chunk_end):
            ok, frame = cap.read()
            if not ok:
                break
            frames[i] = frame
        return frames

    def _maybe_prefetch(self, current_index: int) -> None:
        """Kick off a background load of the next *uncached* chunk.

        With read-ahead the immediate next chunk is usually already cached, so
        we walk forward through cached chunks and prefetch the first gap.
        Triggered when the current index is past the midpoint of the chunk
        immediately before the gap.
        """
        frame_count = self.project_state.video_metadata.frame_count
        chunk_start = (current_index // self.chunk_size) * self.chunk_size

        # Walk forward to find the first uncached chunk
        target = chunk_start + self.chunk_size
        while target < frame_count:
            cached = False
            for entry in self._chunks:
                if entry is not None and entry[0] == target:
                    cached = True
                    break
            if not cached:
                break
            target += self.chunk_size

        if target >= frame_count:
            return  # nothing to prefetch

        # The chunk before the gap — trigger at its midpoint
        trigger_chunk = target - self.chunk_size
        midpoint = trigger_chunk + self.chunk_size // 2
        if current_index < midpoint:
            return  # too early

        # Already prefetched or in-progress?
        with self._prefetch_lock:
            if self._prefetch_result is not None and self._prefetch_result[0] == target:
                return
            if self._prefetch_thread and self._prefetch_thread.is_alive():
                return
        self._start_prefetch(target)

    def _start_prefetch(self, chunk_start: int) -> None:
        """Start a background thread to load chunk_start into _prefetch_result."""
        frame_count = self.project_state.video_metadata.frame_count
        cap = self._prefetch_capture
        self._prefetch_target = chunk_start

        def _worker():
            frames = self._load_chunk(cap, chunk_start, frame_count)
            with self._prefetch_lock:
                self._prefetch_result = (chunk_start, frames)

        t = threading.Thread(target=_worker, daemon=True)
        self._prefetch_thread = t
        t.start()

    def _absorb_prefetch(self) -> None:
        """Move a completed prefetch result into the chunk slots."""
        with self._prefetch_lock:
            if self._prefetch_result is None:
                return
            # The worker sets _prefetch_result only after _load_chunk returns,
            # so if it's not None the data is complete — no need to check
            # is_alive() (which can race with thread teardown).
            chunk_start, frames_dict = self._prefetch_result
            self._prefetch_result = None

        # Check it's not already in a slot
        for entry in self._chunks:
            if entry is not None and entry[0] == chunk_start:
                return

        oldest_slot = self._chunk_ages.index(min(self._chunk_ages))
        self._chunks[oldest_slot] = (chunk_start, frames_dict)
        self._age_counter += 1
        self._chunk_ages[oldest_slot] = self._age_counter

    def _cancel_prefetch(self) -> None:
        """Signal that any running prefetch result should be discarded."""
        with self._prefetch_lock:
            self._prefetch_result = None
        # Thread is daemon so it will die on its own; we just ignore the result

    def previous_frame(self) -> np.ndarray:
        """
        Retrieve the previous frame relative to the current index.

        Returns:
            np.ndarray: The previous frame

        Raises:
            IndexError: If already at the first frame.
        """
        self._validate_video_loaded()
        target = self.current_index - 1
        if target < 0:
            raise VideoFrameIndexError("Already at the first frame")
        return self.get_frame(target)

    def release(self) -> None:
        """Release the video capture resource."""
        self._cancel_prefetch()
        if self.capture:
            self.capture.release()
        if self._prefetch_capture:
            self._prefetch_capture.release()
            self._prefetch_capture = None
        self._chunks = [None, None, None]
        self._chunk_ages = [0, 0, 0]
        self._age_counter = 0
        self._last_index = -1

        if self.project_state and self.project_state.video_metadata:
            self.project_state.video_metadata.frame_count = 0
            self.project_state.video_metadata.width = 0
            self.project_state.video_metadata.height = 0
            self.project_state.video_metadata.fps = 0.0

    def _validate_video_loaded(self):
        """Ensure a video is loaded before performing operations."""
        if not self.capture or not self.capture.isOpened():
            raise VideoLoadError(
                "No video loaded - please ensure you have imported a project."
            )
