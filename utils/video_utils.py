"""
Compatibility shim for video utilities.

The project originally used `vedio_utils.py` (typo) but other modules import
`utils.video_utils`. This module re-exports the public API.
"""

from .vedio_utils import FrameData, VideoProcessor

__all__ = ["VideoProcessor", "FrameData"]

