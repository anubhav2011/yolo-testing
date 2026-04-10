"""
Video processing utilities for proctoring system.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Generator, Optional, Tuple
import os

from config import CONFIG


@dataclass
class FrameData:
    """Container for frame information."""
    frame: np.ndarray
    frame_number: int
    timestamp: float
    original_size: Tuple[int, int]
    processed_size: Tuple[int, int]


class VideoProcessor:
    """Handles video file loading and frame extraction."""

    def __init__(self, video_path: str):
        """
        Initialize video processor.

        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 0.0
        self._total_frames: int = 0
        self._width: int = 0
        self._height: int = 0
        self._duration: float = 0.0

        self._validate_and_open()

    def _validate_and_open(self) -> None:
        """Validate video file and open capture."""
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self._cap = cv2.VideoCapture(self.video_path)

        if not self._cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self._fps > 0:
            self._duration = self._total_frames / self._fps

        if CONFIG.VERBOSE:
            print(f"Video loaded: {self.video_path}")
            print(f"  Resolution: {self._width}x{self._height}")
            print(f"  FPS: {self._fps:.2f}")
            print(f"  Total frames: {self._total_frames}")
            print(f"  Duration: {self._duration:.2f}s")

    @property
    def fps(self) -> float:
        """Get video FPS."""
        return self._fps

    @property
    def total_frames(self) -> int:
        """Get total frame count."""
        return self._total_frames

    @property
    def duration(self) -> float:
        """Get video duration in seconds."""
        return self._duration

    @property
    def resolution(self) -> Tuple[int, int]:
        """Get original video resolution (width, height)."""
        return (self._width, self._height)

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize frame if needed while maintaining aspect ratio.

        Args:
            frame: Input frame

        Returns:
            Resized frame
        """
        h, w = frame.shape[:2]
        max_w = CONFIG.video.MAX_FRAME_WIDTH
        max_h = CONFIG.video.MAX_FRAME_HEIGHT

        if w <= max_w and h <= max_h:
            return frame

        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def get_frames(self,
                   skip_frames: int = 0,
                   max_frames: Optional[int] = None) -> Generator[FrameData, None, None]:
        """
        Generator that yields frames from the video.

        Args:
            skip_frames: Number of frames to skip between processed frames
            max_frames: Maximum number of frames to process (None for all)

        Yields:
            FrameData objects containing frame and metadata
        """
        if self._cap is None:
            return

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_count = 0
        processed_count = 0

        while True:
            if max_frames is not None and processed_count >= max_frames:
                break

            ret, frame = self._cap.read()

            if not ret:
                break

            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                frame_count += 1
                continue

            timestamp = frame_count / self._fps if self._fps > 0 else 0.0
            original_size = (frame.shape[1], frame.shape[0])

            processed_frame = self._resize_frame(frame)
            processed_size = (processed_frame.shape[1], processed_frame.shape[0])

            yield FrameData(
                frame=processed_frame,
                frame_number=frame_count,
                timestamp=timestamp,
                original_size=original_size,
                processed_size=processed_size
            )

            frame_count += 1
            processed_count += 1

    def get_frame_at(self, frame_number: int) -> Optional[FrameData]:
        """
        Get a specific frame by number.

        Args:
            frame_number: Frame index to retrieve

        Returns:
            FrameData or None if frame not found
        """
        if self._cap is None or frame_number >= self._total_frames:
            return None

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self._cap.read()

        if not ret:
            return None

        timestamp = frame_number / self._fps if self._fps > 0 else 0.0
        original_size = (frame.shape[1], frame.shape[0])

        processed_frame = self._resize_frame(frame)
        processed_size = (processed_frame.shape[1], processed_frame.shape[0])

        return FrameData(
            frame=processed_frame,
            frame_number=frame_number,
            timestamp=timestamp,
            original_size=original_size,
            processed_size=processed_size
        )

    def release(self) -> None:
        """Release video capture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False