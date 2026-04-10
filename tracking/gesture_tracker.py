"""
Gesture tracking and event aggregation.
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum

from config import CONFIG
from utils.time_utils import TimeFormatter


class GestureType(Enum):
    """Types of tracked gestures."""
    HEAD_MOVEMENT = "head_movement"
    EYE_MOVEMENT = "eye_movement"
    FACE_MISSING = "face_missing"


@dataclass
class GestureEvent:
    """Represents a single gesture occurrence."""
    gesture_type: GestureType
    start_time: float  # Start timestamp in seconds
    end_time: float  # End timestamp in seconds
    direction: str  # Movement direction
    intensity: float  # Rotation angle or displacement magnitude

    @property
    def duration(self) -> int:
        """Get duration in seconds (rounded)."""
        return TimeFormatter.format_duration(self.end_time - self.start_time)

    @property
    def timestamp(self) -> str:
        """Get formatted timestamp string."""
        return TimeFormatter.seconds_to_mmss(self.start_time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "timestamp": self.timestamp,
            "duration": self.duration,
            "direction": self.direction,
            "intensity": f"{abs(self.intensity):.1f}" if self.intensity else ""
        }


@dataclass
class GestureReport:
    """Final gesture analysis report."""
    gestures: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {"gestures": self.gestures}

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class GestureTracker:
    """
    Tracks and aggregates gesture events from frame-by-frame detections.

    Handles:
    - Head movements (yaw/pitch beyond thresholds)
    - Eye movements (gaze beyond thresholds)
    - Face missing events
    """

    def __init__(self):
        """Initialize gesture tracker."""
        self._events: List[GestureEvent] = []

        # Current tracking state
        self._current_head_event: Optional[Dict] = None
        self._current_eye_event: Optional[Dict] = None
        self._current_face_missing_event: Optional[Dict] = None

        # Last known states
        self._last_head_direction: Optional[str] = None
        self._last_eye_direction: Optional[str] = None
        self._last_face_present: bool = True
        self._last_timestamp: float = 0.0

    def update(self,
               timestamp: float,
               face_detected: bool,
               head_direction: Optional[str] = None,
               head_intensity: float = 0.0,
               eye_direction: Optional[str] = None,
               eye_intensity: float = 0.0):
        """
        Update tracker with frame detection results.

        Args:
            timestamp: Current frame timestamp in seconds
            face_detected: Whether face was detected in frame
            head_direction: Head movement direction ('left', 'right', 'up', 'down') or None
            head_intensity: Head rotation angle in degrees
            eye_direction: Eye gaze direction or None
            eye_intensity: Eye gaze displacement (will be converted to degrees)
        """
        self._last_timestamp = timestamp

        # Track face missing
        self._track_face_missing(timestamp, face_detected)

        # Only track head/eye if face is present
        if face_detected:
            self._track_head_movement(timestamp, head_direction, head_intensity)
            self._track_eye_movement(timestamp, eye_direction, eye_intensity)

    def _track_face_missing(self, timestamp: float, face_detected: bool):
        """Track face missing events."""
        if not face_detected:
            if self._current_face_missing_event is None:
                # Start new face missing event
                self._current_face_missing_event = {
                    "start_time": timestamp,
                    "end_time": timestamp
                }
            else:
                # Continue existing event
                self._current_face_missing_event["end_time"] = timestamp
        else:
            if self._current_face_missing_event is not None:
                # End face missing event
                duration = (self._current_face_missing_event["end_time"] -
                            self._current_face_missing_event["start_time"])

                if duration >= CONFIG.tracking.FACE_MISSING_THRESHOLD:
                    self._events.append(GestureEvent(
                        gesture_type=GestureType.FACE_MISSING,
                        start_time=self._current_face_missing_event["start_time"],
                        end_time=self._current_face_missing_event["end_time"],
                        direction="",
                        intensity=0.0
                    ))

                self._current_face_missing_event = None

        self._last_face_present = face_detected

    def _track_head_movement(self,
                             timestamp: float,
                             direction: Optional[str],
                             intensity: float):
        """Track head movement events."""
        if direction is not None:
            if self._current_head_event is None:
                # Start new head movement event
                self._current_head_event = {
                    "start_time": timestamp,
                    "end_time": timestamp,
                    "direction": direction,
                    "max_intensity": intensity
                }
            elif (self._current_head_event["direction"] == direction and
                  timestamp - self._current_head_event["end_time"] <= CONFIG.tracking.EVENT_MERGE_GAP):
                # Continue existing event with same direction
                self._current_head_event["end_time"] = timestamp
                self._current_head_event["max_intensity"] = max(
                    self._current_head_event["max_intensity"], intensity
                )
            else:
                # Different direction or gap too large - end current and start new
                self._finalize_head_event()
                self._current_head_event = {
                    "start_time": timestamp,
                    "end_time": timestamp,
                    "direction": direction,
                    "max_intensity": intensity
                }
        else:
            # No movement detected - check if we should finalize current event
            if self._current_head_event is not None:
                if timestamp - self._current_head_event["end_time"] > CONFIG.tracking.EVENT_MERGE_GAP:
                    self._finalize_head_event()

        self._last_head_direction = direction

    def _finalize_head_event(self):
        """Finalize and store current head movement event."""
        if self._current_head_event is None:
            return

        duration = self._current_head_event["end_time"] - self._current_head_event["start_time"]

        if duration >= CONFIG.tracking.MIN_EVENT_DURATION:
            self._events.append(GestureEvent(
                gesture_type=GestureType.HEAD_MOVEMENT,
                start_time=self._current_head_event["start_time"],
                end_time=self._current_head_event["end_time"],
                direction=self._current_head_event["direction"],
                intensity=self._current_head_event["max_intensity"]
            ))

        self._current_head_event = None

    def _track_eye_movement(self,
                            timestamp: float,
                            direction: Optional[str],
                            intensity: float):
        """Track eye movement events."""
        # Convert gaze displacement to approximate degrees
        intensity_degrees = intensity * 60  # Approximate conversion

        if direction is not None:
            if self._current_eye_event is None:
                # Start new eye movement event
                self._current_eye_event = {
                    "start_time": timestamp,
                    "end_time": timestamp,
                    "direction": direction,
                    "max_intensity": intensity_degrees
                }
            elif (self._current_eye_event["direction"] == direction and
                  timestamp - self._current_eye_event["end_time"] <= CONFIG.tracking.EVENT_MERGE_GAP):
                # Continue existing event
                self._current_eye_event["end_time"] = timestamp
                self._current_eye_event["max_intensity"] = max(
                    self._current_eye_event["max_intensity"], intensity_degrees
                )
            else:
                # Different direction - finalize and start new
                self._finalize_eye_event()
                self._current_eye_event = {
                    "start_time": timestamp,
                    "end_time": timestamp,
                    "direction": direction,
                    "max_intensity": intensity_degrees
                }
        else:
            # No gaze deviation - check if we should finalize
            if self._current_eye_event is not None:
                if timestamp - self._current_eye_event["end_time"] > CONFIG.tracking.EVENT_MERGE_GAP:
                    self._finalize_eye_event()

        self._last_eye_direction = direction

    def _finalize_eye_event(self):
        """Finalize and store current eye movement event."""
        if self._current_eye_event is None:
            return

        duration = self._current_eye_event["end_time"] - self._current_eye_event["start_time"]

        if duration >= CONFIG.tracking.MIN_EVENT_DURATION:
            self._events.append(GestureEvent(
                gesture_type=GestureType.EYE_MOVEMENT,
                start_time=self._current_eye_event["start_time"],
                end_time=self._current_eye_event["end_time"],
                direction=self._current_eye_event["direction"],
                intensity=self._current_eye_event["max_intensity"]
            ))

        self._current_eye_event = None

    def finalize(self):
        """Finalize all pending events at end of video."""
        # Finalize any open head movement
        self._finalize_head_event()

        # Finalize any open eye movement
        self._finalize_eye_event()

        # Finalize any open face missing event
        if self._current_face_missing_event is not None:
            duration = (self._current_face_missing_event["end_time"] -
                        self._current_face_missing_event["start_time"])

            if duration >= CONFIG.tracking.FACE_MISSING_THRESHOLD:
                self._events.append(GestureEvent(
                    gesture_type=GestureType.FACE_MISSING,
                    start_time=self._current_face_missing_event["start_time"],
                    end_time=self._current_face_missing_event["end_time"],
                    direction="",
                    intensity=0.0
                ))

            self._current_face_missing_event = None

    def generate_report(self) -> GestureReport:
        """
        Generate final gesture report.

        Returns:
            GestureReport object with all detected gestures
        """
        self.finalize()

        # Group events by type
        head_movements = [e for e in self._events if e.gesture_type == GestureType.HEAD_MOVEMENT]
        eye_movements = [e for e in self._events if e.gesture_type == GestureType.EYE_MOVEMENT]
        face_missing = [e for e in self._events if e.gesture_type == GestureType.FACE_MISSING]

        # Sort events by start time
        head_movements.sort(key=lambda e: e.start_time)
        eye_movements.sort(key=lambda e: e.start_time)
        face_missing.sort(key=lambda e: e.start_time)

        # Build report structure
        gestures = []

        if head_movements:
            gestures.append({
                "name": "head_movement",
                "occurrence": [e.to_dict() for e in head_movements]
            })

        if eye_movements:
            gestures.append({
                "name": "eye_movement",
                "occurrence": [e.to_dict() for e in eye_movements]
            })

        if face_missing:
            gestures.append({
                "name": "face_missing",
                "occurrence": [e.to_dict() for e in face_missing]
            })

        return GestureReport(gestures=gestures)

    def reset(self):
        """Reset tracker state for processing a new video."""
        self._events = []
        self._current_head_event = None
        self._current_eye_event = None
        self._current_face_missing_event = None
        self._last_head_direction = None
        self._last_eye_direction = None
        self._last_face_present = True
        self._last_timestamp = 0.0