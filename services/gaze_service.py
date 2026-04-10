"""
Gaze estimation service using MediaPipe Iris landmarks.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
import math

from config import CONFIG


@dataclass
class GazeData:
    """Container for gaze estimation results."""
    horizontal: float  # Normalized horizontal displacement (-1 to 1)
    vertical: float  # Normalized vertical displacement (-1 to 1)
    confidence: float
    left_ear: float  # Left eye aspect ratio
    right_ear: float  # Right eye aspect ratio
    is_blinking: bool

    @property
    def is_looking_left(self) -> bool:
        """Check if gaze is directed left."""
        return self.horizontal < -CONFIG.detection.GAZE_HORIZONTAL_THRESHOLD

    @property
    def is_looking_right(self) -> bool:
        """Check if gaze is directed right."""
        return self.horizontal > CONFIG.detection.GAZE_HORIZONTAL_THRESHOLD

    @property
    def is_looking_up(self) -> bool:
        """Check if gaze is directed up."""
        return self.vertical < -CONFIG.detection.GAZE_VERTICAL_THRESHOLD

    @property
    def is_looking_down(self) -> bool:
        """Check if gaze is directed down."""
        return self.vertical > CONFIG.detection.GAZE_VERTICAL_THRESHOLD

    def get_primary_direction(self) -> Optional[str]:
        """
        Get the primary gaze direction.

        Returns:
            Direction string or None if centered
        """
        max_deviation = 0
        direction = None

        h_thresh = CONFIG.detection.GAZE_HORIZONTAL_THRESHOLD
        v_thresh = CONFIG.detection.GAZE_VERTICAL_THRESHOLD

        if abs(self.horizontal) > h_thresh:
            if abs(self.horizontal) > max_deviation:
                max_deviation = abs(self.horizontal)
                direction = "left" if self.horizontal < 0 else "right"

        if abs(self.vertical) > v_thresh:
            if abs(self.vertical) > max_deviation:
                max_deviation = abs(self.vertical)
                direction = "up" if self.vertical < 0 else "down"

        return direction

    def get_intensity_degrees(self) -> float:
        """
        Convert gaze displacement to approximate degrees.

        Returns:
            Approximate gaze angle in degrees
        """
        # Approximate conversion: max normalized value (~0.5) ≈ 30 degrees
        max_disp = max(abs(self.horizontal), abs(self.vertical))
        return max_disp * 60  # Scale factor


class GazeService:
    """
    Gaze estimation using MediaPipe Iris landmarks.

    Calculates iris displacement relative to eye corners
    to determine gaze direction.
    """

    # MediaPipe landmark indices
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
    RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

    # Eye corner indices for displacement calculation
    LEFT_EYE_LEFT_CORNER = 33
    LEFT_EYE_RIGHT_CORNER = 133
    RIGHT_EYE_LEFT_CORNER = 362
    RIGHT_EYE_RIGHT_CORNER = 263

    # Vertical eye landmarks for EAR
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374

    def __init__(self):
        """Initialize gaze service."""
        self._face_mesh = None
        self._initialized = False
        self.last_init_error = None  # type: Optional[str]

    def initialize(self) -> bool:
        """
        Initialize MediaPipe FaceMesh.

        Returns:
            True if initialization successful
        """
        self.last_init_error = None
        try:
            import mediapipe as mp

            if not hasattr(mp, "solutions"):
                self.last_init_error = (
                    "This MediaPipe install has no 'mediapipe.solutions' (common with "
                    "mediapipe>=0.10.15 wheels or running outside the project venv). "
                    "Use: python -m venv venv, activate it, then pip install -r requirements.txt "
                    "(e.g. mediapipe==0.10.14)."
                )
                print(f"Error initializing MediaPipe: {self.last_init_error}")
                return False

            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=CONFIG.detection.MEDIAPIPE_MAX_FACES,
                refine_landmarks=True,  # Required for iris landmarks
                min_detection_confidence=CONFIG.detection.MEDIAPIPE_MIN_DETECTION_CONF,
                min_tracking_confidence=CONFIG.detection.MEDIAPIPE_MIN_TRACKING_CONF
            )
            self._initialized = True

            if CONFIG.VERBOSE:
                print("MediaPipe gaze estimation initialized successfully")

            return True

        except ImportError:
            self.last_init_error = "mediapipe is not installed. Run: pip install mediapipe"
            print("Error: mediapipe package not installed")
            print("Install with: pip install mediapipe")
            return False
        except Exception as e:
            self.last_init_error = f"MediaPipe gaze init failed: {e}"
            print(f"Error initializing MediaPipe: {e}")
            return False

    def _calculate_ear(self,
                       landmarks: list,
                       eye_top_idx: int,
                       eye_bottom_idx: int,
                       eye_left_idx: int,
                       eye_right_idx: int,
                       width: int,
                       height: int) -> float:
        """
        Calculate Eye Aspect Ratio (EAR).

        Args:
            landmarks: MediaPipe face landmarks
            eye_*_idx: Landmark indices for eye corners
            width: Image width
            height: Image height

        Returns:
            Eye aspect ratio value
        """

        def get_point(idx):
            return (landmarks[idx].x * width, landmarks[idx].y * height)

        top = get_point(eye_top_idx)
        bottom = get_point(eye_bottom_idx)
        left = get_point(eye_left_idx)
        right = get_point(eye_right_idx)

        vertical_dist = math.sqrt((top[0] - bottom[0]) ** 2 + (top[1] - bottom[1]) ** 2)
        horizontal_dist = math.sqrt((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2)

        if horizontal_dist == 0:
            return 0.0

        return vertical_dist / horizontal_dist

    def _calculate_iris_displacement(self,
                                     landmarks: list,
                                     iris_indices: List[int],
                                     eye_left_idx: int,
                                     eye_right_idx: int,
                                     eye_top_idx: int,
                                     eye_bottom_idx: int,
                                     width: int,
                                     height: int) -> Tuple[float, float]:
        """
        Calculate iris displacement from eye center.

        Args:
            landmarks: MediaPipe face landmarks
            iris_indices: Iris landmark indices
            eye_*_idx: Eye corner landmark indices
            width: Image width
            height: Image height

        Returns:
            Tuple of (horizontal, vertical) displacement (-1 to 1)
        """

        def get_point(idx):
            return (landmarks[idx].x * width, landmarks[idx].y * height)

        # Get iris center
        iris_points = [get_point(i) for i in iris_indices]
        iris_center = (
            sum(p[0] for p in iris_points) / len(iris_points),
            sum(p[1] for p in iris_points) / len(iris_points)
        )

        # Get eye corners
        left_corner = get_point(eye_left_idx)
        right_corner = get_point(eye_right_idx)
        top_corner = get_point(eye_top_idx)
        bottom_corner = get_point(eye_bottom_idx)

        # Calculate eye center
        eye_center_x = (left_corner[0] + right_corner[0]) / 2
        eye_center_y = (top_corner[1] + bottom_corner[1]) / 2

        # Calculate eye dimensions
        eye_width = abs(right_corner[0] - left_corner[0])
        eye_height = abs(bottom_corner[1] - top_corner[1])

        if eye_width == 0 or eye_height == 0:
            return 0.0, 0.0

        # Calculate normalized displacement
        horizontal = (iris_center[0] - eye_center_x) / (eye_width / 2)
        vertical = (iris_center[1] - eye_center_y) / (eye_height / 2)

        # Clamp to [-1, 1]
        horizontal = max(-1.0, min(1.0, horizontal))
        vertical = max(-1.0, min(1.0, vertical))

        return horizontal, vertical

    def estimate_gaze(self,
                      frame: np.ndarray,
                      face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Optional[GazeData]:
        """
        Estimate gaze direction from a frame.

        Args:
            frame: BGR image
            face_bbox: Optional face bounding box to crop

        Returns:
            GazeData object or None if estimation fails
        """
        if not self._initialized:
            if not self.initialize():
                return None

        # Use face crop if bbox provided
        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            h, w = frame.shape[:2]

            # Add padding
            pad = CONFIG.detection.FACE_PADDING_RATIO
            pad_x = int((x2 - x1) * pad)
            pad_y = int((y2 - y1) * pad)

            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            frame = frame[y1:y2, x1:x2]

        if frame.size == 0:
            return None

        h, w = frame.shape[:2]

        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb_frame)

            if not results.multi_face_landmarks:
                return None

            landmarks = results.multi_face_landmarks[0].landmark

            # Calculate EAR for both eyes
            left_ear = self._calculate_ear(
                landmarks,
                self.LEFT_EYE_TOP, self.LEFT_EYE_BOTTOM,
                self.LEFT_EYE_LEFT_CORNER, self.LEFT_EYE_RIGHT_CORNER,
                w, h
            )

            right_ear = self._calculate_ear(
                landmarks,
                self.RIGHT_EYE_TOP, self.RIGHT_EYE_BOTTOM,
                self.RIGHT_EYE_LEFT_CORNER, self.RIGHT_EYE_RIGHT_CORNER,
                w, h
            )

            is_blinking = (left_ear < CONFIG.detection.EAR_THRESHOLD and
                           right_ear < CONFIG.detection.EAR_THRESHOLD)

            # Calculate iris displacement for both eyes
            left_h, left_v = self._calculate_iris_displacement(
                landmarks,
                self.LEFT_IRIS_INDICES,
                self.LEFT_EYE_LEFT_CORNER, self.LEFT_EYE_RIGHT_CORNER,
                self.LEFT_EYE_TOP, self.LEFT_EYE_BOTTOM,
                w, h
            )

            right_h, right_v = self._calculate_iris_displacement(
                landmarks,
                self.RIGHT_IRIS_INDICES,
                self.RIGHT_EYE_LEFT_CORNER, self.RIGHT_EYE_RIGHT_CORNER,
                self.RIGHT_EYE_TOP, self.RIGHT_EYE_BOTTOM,
                w, h
            )

            # Average both eyes
            horizontal = (left_h + right_h) / 2
            vertical = (left_v + right_v) / 2

            # Estimate confidence based on landmark visibility
            visibility_sum = sum(
                landmarks[i].visibility if hasattr(landmarks[i], 'visibility') else 1.0
                for i in self.LEFT_IRIS_INDICES + self.RIGHT_IRIS_INDICES
            )
            confidence = min(1.0, visibility_sum / 10)

            return GazeData(
                horizontal=horizontal,
                vertical=vertical,
                confidence=confidence,
                left_ear=left_ear,
                right_ear=right_ear,
                is_blinking=is_blinking
            )

        except Exception as e:
            if CONFIG.VERBOSE:
                print(f"Gaze estimation error: {e}")
            return None

    def close(self):
        """Release resources."""
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None
            self._initialized = False