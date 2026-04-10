"""
Head pose estimation service using WHENet ONNX model.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import os

from config import CONFIG


@dataclass
class HeadPose:
    """Container for head pose estimation results."""
    yaw: float  # Rotation around vertical axis (left/right)
    pitch: float  # Rotation around lateral axis (up/down)
    roll: float  # Rotation around longitudinal axis (tilt)
    confidence: float

    @property
    def is_looking_left(self) -> bool:
        """Check if head is turned left beyond threshold."""
        return self.yaw < -CONFIG.detection.HEAD_YAW_THRESHOLD

    @property
    def is_looking_right(self) -> bool:
        """Check if head is turned right beyond threshold."""
        return self.yaw > CONFIG.detection.HEAD_YAW_THRESHOLD

    @property
    def is_looking_up(self) -> bool:
        """Check if head is tilted up beyond threshold."""
        return self.pitch < -CONFIG.detection.HEAD_PITCH_THRESHOLD

    @property
    def is_looking_down(self) -> bool:
        """Check if head is tilted down beyond threshold."""
        return self.pitch > CONFIG.detection.HEAD_PITCH_THRESHOLD

    def get_primary_direction(self) -> Optional[str]:
        """
        Get the primary direction the head is turned.

        Returns:
            Direction string ('left', 'right', 'up', 'down') or None
        """
        max_deviation = 0
        direction = None

        # Check horizontal movement
        if abs(self.yaw) > CONFIG.detection.HEAD_YAW_THRESHOLD:
            if abs(self.yaw) > max_deviation:
                max_deviation = abs(self.yaw)
                direction = "left" if self.yaw < 0 else "right"

        # Check vertical movement
        if abs(self.pitch) > CONFIG.detection.HEAD_PITCH_THRESHOLD:
            if abs(self.pitch) > max_deviation:
                max_deviation = abs(self.pitch)
                direction = "up" if self.pitch < 0 else "down"

        return direction

    def get_intensity(self) -> float:
        """
        Get the maximum rotation angle (intensity).

        Returns:
            Maximum absolute rotation in degrees
        """
        return max(abs(self.yaw), abs(self.pitch), abs(self.roll))


class PoseService:
    """
    Head pose estimation service using WHENet.

    WHENet directly regresses yaw, pitch, roll from face crops,
    providing faster and more stable results than solvePnP approaches.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize pose estimation service.

        Args:
            model_path: Path to WHENet ONNX model
        """
        self.model_path = model_path or CONFIG.model.WHENET_MODEL_PATH
        self._session = None
        self._input_name = None
        self._input_size = (224, 224)  # WHENet input size
        self._initialized = False
        self._use_fallback = False
        self._resolved_model_path: Optional[str] = None

    def _resolve_model_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(project_root, path)

    def initialize(self) -> bool:
        """
        Initialize the ONNX inference session.

        Returns:
            True if initialization successful
        """
        try:
            import onnxruntime as ort

            requested = self._resolve_model_path(self.model_path)
            # Keep a fallback name in case config differs from the actual file name.
            fallback = self._resolve_model_path(os.path.join("models", "WHENet.onnx"))

            if os.path.exists(requested):
                self._resolved_model_path = requested
                self._session = ort.InferenceSession(
                    requested,
                    providers=["CPUExecutionProvider"]
                )
                self._input_name = self._session.get_inputs()[0].name
                self._initialized = True

                if CONFIG.VERBOSE:
                    print("WHENet pose estimation initialized successfully")

                return True
            elif os.path.exists(fallback):
                self._resolved_model_path = fallback
                self._session = ort.InferenceSession(
                    fallback,
                    providers=["CPUExecutionProvider"]
                )
                self._input_name = self._session.get_inputs()[0].name
                self._initialized = True

                if CONFIG.VERBOSE:
                    print("WHENet pose estimation initialized successfully")

                return True
            else:
                print(f"WHENet model not found at {requested}")
                print("Falling back to solvePnP-based pose estimation")
                self._use_fallback = True
                self._initialized = True
                return True

        except ImportError:
            print("Error: onnxruntime package not installed")
            print("Install with: pip install onnxruntime")
            print("Falling back to solvePnP-based pose estimation")
            self._use_fallback = True
            self._initialized = True
            return True
        except Exception as e:
            print(f"Error initializing WHENet: {e}")
            print("Falling back to solvePnP-based pose estimation")
            self._use_fallback = True
            self._initialized = True
            return True

    def _estimate_pose_whenet(self, face_crop: np.ndarray) -> Optional[HeadPose]:
        """
        Estimate head pose using WHENet model.

        Args:
            face_crop: Cropped face image (BGR)

        Returns:
            HeadPose object or None
        """
        try:
            # Preprocess for WHENet
            inp = cv2.resize(face_crop, self._input_size)
            inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
            inp = inp.astype(np.float32) / 255.0
            inp = np.expand_dims(inp.transpose(2, 0, 1), 0)  # NCHW format

            # Run inference
            output = self._session.run(None, {self._input_name: inp})

            # Parse output (yaw, pitch, roll)
            yaw, pitch, roll = output[0][0]

            return HeadPose(
                yaw=float(yaw),
                pitch=float(pitch),
                roll=float(roll),
                confidence=0.85  # WHENet generally reliable
            )

        except Exception as e:
            if CONFIG.VERBOSE:
                print(f"WHENet inference error: {e}")
            return None

    def _estimate_pose_solvepnp(self,
                                face_crop: np.ndarray,
                                face_keypoints: Optional[list] = None) -> Optional[HeadPose]:
        """
        Fallback pose estimation using solvePnP.

        Args:
            face_crop: Cropped face image
            face_keypoints: Facial keypoints if available

        Returns:
            HeadPose object or None
        """
        try:
            import mediapipe as mp

            h, w = face_crop.shape[:2]

            # Use MediaPipe to get facial landmarks
            face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5
            )

            rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_crop)
            face_mesh.close()

            if not results.multi_face_landmarks:
                return None

            landmarks = results.multi_face_landmarks[0].landmark

            # 6 key points for pose estimation
            # Nose tip, chin, left eye corner, right eye corner, left mouth corner, right mouth corner
            landmark_indices = [1, 152, 33, 263, 61, 291]

            # 3D model points (generic face model)
            model_points = np.array([
                (0.0, 0.0, 0.0),  # Nose tip
                (0.0, -330.0, -65.0),  # Chin
                (-225.0, 170.0, -135.0),  # Left eye corner
                (225.0, 170.0, -135.0),  # Right eye corner
                (-150.0, -150.0, -125.0),  # Left mouth corner
                (150.0, -150.0, -125.0)  # Right mouth corner
            ], dtype=np.float64)

            # 2D image points
            image_points = np.array([
                (landmarks[i].x * w, landmarks[i].y * h)
                for i in landmark_indices
            ], dtype=np.float64)

            # Camera matrix (approximate)
            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)

            dist_coeffs = np.zeros((4, 1))

            # Solve PnP
            success, rotation_vec, translation_vec = cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return None

            # Convert rotation vector to Euler angles
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            pose_mat = cv2.hconcat([rotation_mat, translation_vec])
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

            yaw = float(euler_angles[1][0])
            pitch = float(euler_angles[0][0])
            roll = float(euler_angles[2][0])

            return HeadPose(
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                confidence=0.7  # solvePnP less reliable than WHENet
            )

        except Exception as e:
            if CONFIG.VERBOSE:
                print(f"solvePnP pose estimation error: {e}")
            return None

    def estimate_pose(self,
                      frame: np.ndarray,
                      bbox: Tuple[int, int, int, int],
                      keypoints: Optional[list] = None) -> Optional[HeadPose]:
        """
        Estimate head pose from a face bounding box.

        Args:
            frame: Full BGR image
            bbox: Face bounding box (x1, y1, x2, y2)
            keypoints: Optional facial keypoints

        Returns:
            HeadPose object or None if estimation fails
        """
        if not self._initialized:
            self.initialize()

        # Extract face crop with padding
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        pad_ratio = CONFIG.detection.FACE_PADDING_RATIO
        pad_x = int((x2 - x1) * pad_ratio)
        pad_y = int((y2 - y1) * pad_ratio)

        x1_padded = max(0, x1 - pad_x)
        y1_padded = max(0, y1 - pad_y)
        x2_padded = min(w, x2 + pad_x)
        y2_padded = min(h, y2 + pad_y)

        face_crop = frame[y1_padded:y2_padded, x1_padded:x2_padded]

        if face_crop.size == 0:
            return None

        # Use WHENet if available, fallback to solvePnP
        if self._use_fallback:
            return self._estimate_pose_solvepnp(face_crop, keypoints)
        else:
            pose = self._estimate_pose_whenet(face_crop)
            if pose is None:
                return self._estimate_pose_solvepnp(face_crop, keypoints)
            return pose