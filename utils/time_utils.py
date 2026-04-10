"""
Time formatting utilities for proctoring system.
"""

from typing import Union


class TimeFormatter:
    """Handles time format conversions for the proctoring system."""

    @staticmethod
    def seconds_to_mmss(seconds: float) -> str:
        """
        Convert seconds to MM:SS:ms format.

        Args:
            seconds: Time in seconds (float)

        Returns:
            Formatted string in MM:SS:ms format (e.g., "02:34:567")
        """
        if seconds < 0:
            seconds = 0

        total_ms = int(seconds * 1000)
        minutes = total_ms // 60000
        remaining_ms = total_ms % 60000
        secs = remaining_ms // 1000
        ms = remaining_ms % 1000

        return f"{minutes:02d}:{secs:02d}:{ms:03d}"

    @staticmethod
    def mmss_to_seconds(time_str: str) -> float:
        """
        Convert MM:SS:ms format to seconds.

        Args:
            time_str: Time string in MM:SS:ms format

        Returns:
            Time in seconds (float)
        """
        parts = time_str.split(":")
        if len(parts) == 3:
            minutes, secs, ms = parts
            return int(minutes) * 60 + int(secs) + int(ms) / 1000
        elif len(parts) == 2:
            minutes, secs = parts
            return int(minutes) * 60 + float(secs)
        else:
            return float(time_str)

    @staticmethod
    def format_duration(duration_seconds: float) -> int:
        """
        Format duration to integer seconds.

        Args:
            duration_seconds: Duration in seconds (float)

        Returns:
            Duration rounded to nearest integer
        """
        return max(1, round(duration_seconds))

    @staticmethod
    def frame_to_timestamp(frame_number: int, fps: float) -> float:
        """
        Convert frame number to timestamp in seconds.

        Args:
            frame_number: Frame index
            fps: Frames per second

        Returns:
            Timestamp in seconds
        """
        if fps <= 0:
            return 0.0
        return frame_number / fps