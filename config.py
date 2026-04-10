"""
Configuration settings for the proctoring system.
"""

import os
from dataclasses import dataclass, field
from typing import List


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def resolve_project_path(path: str) -> str:
    """
    Resolve a possibly-relative path against the project root.

    This keeps config portable when running from different working directories.
    """
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


@dataclass
class ModelConfig:
    """Model paths and settings."""
    # Prefer ONNX weights when present in `models/`.
    # `DetectionService` will still fall back to auto-downloading a YOLO model if needed.
    YOLO_FACE_MODEL_PATH: str = "models/model.onnx"
    WHENET_MODEL_PATH: str = "models/WHENet.onnx"
    YOLO_INPUT_SIZE: int = 416
    YOLO_CONF_THRESHOLD: float = 0.45


@dataclass
class DetectionConfig:
    """Detection thresholds and parameters."""
    # Face detection
    MIN_FACE_CONFIDENCE: float = 0.5
    FACE_PADDING_RATIO: float = 0.2

    # Head pose thresholds (degrees)
    HEAD_YAW_THRESHOLD: float = 25.0
    HEAD_PITCH_THRESHOLD: float = 20.0
    HEAD_ROLL_THRESHOLD: float = 15.0

    # Gaze thresholds (normalized displacement)
    GAZE_HORIZONTAL_THRESHOLD: float = 0.15
    GAZE_VERTICAL_THRESHOLD: float = 0.12

    # Eye aspect ratio for blink detection
    EAR_THRESHOLD: float = 0.2

    # MediaPipe settings
    MEDIAPIPE_MAX_FACES: int = 1
    MEDIAPIPE_MIN_DETECTION_CONF: float = 0.5
    MEDIAPIPE_MIN_TRACKING_CONF: float = 0.5


@dataclass
class TrackingConfig:
    """Gesture tracking parameters."""
    # Minimum duration (seconds) for an event to be recorded
    MIN_EVENT_DURATION: float = 0.3

    # Maximum gap (seconds) to merge consecutive events
    EVENT_MERGE_GAP: float = 0.5

    # Face missing threshold (seconds)
    FACE_MISSING_THRESHOLD: float = 0.5


@dataclass
class VideoConfig:
    """Video processing settings."""
    TARGET_FPS: int = 18
    MAX_FRAME_WIDTH: int = 640
    MAX_FRAME_HEIGHT: int = 480


@dataclass
class ProctoringConfig:
    """Main configuration class combining all configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    video: VideoConfig = field(default_factory=VideoConfig)

    # Output settings
    OUTPUT_JSON_INDENT: int = 2
    VERBOSE: bool = True


# Global config instance
CONFIG = ProctoringConfig()