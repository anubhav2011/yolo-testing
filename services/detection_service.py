"""
YOLO-based face detection service.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import os

from config import CONFIG


@dataclass
class FaceDetection:
    """Container for face detection results."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    keypoints: Optional[List[Tuple[float, float]]] = None  # 5 facial keypoints
    center: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        """Calculate center if not provided."""
        if self.center is None:
            x1, y1, x2, y2 = self.bbox
            self.center = ((x1 + x2) // 2, (y1 + y2) // 2)


class DetectionService:
    """
    Face detection service using YOLOv8n-face.

    Provides robust face detection with bounding boxes and facial keypoints.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the detection service.

        Args:
            model_path: Path to YOLOv8n-face model file
        """
        self.model_path = model_path or CONFIG.model.YOLO_FACE_MODEL_PATH
        self._model = None
        self._initialized = False
        self._resolved_model_path: Optional[str] = None
        self.last_init_error: Optional[str] = None

    def _resolve_model_path(self, path: str) -> str:
        """
        Resolve model path relative to project root.

        This allows running from any working directory.
        """
        if os.path.isabs(path):
            return path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(project_root, path)

    def initialize(self) -> bool:
        """
        Initialize the YOLO model.

        Returns:
            True if initialization successful, False otherwise
        """
        self.last_init_error = None
        try:
            from ultralytics import YOLO

            requested = self._resolve_model_path(self.model_path)
            onnx_fallback = self._resolve_model_path(os.path.join("models", "model.onnx"))
            pt_fallback = self._resolve_model_path(os.path.join("models", "yolov8n-face.pt"))

            if os.path.exists(requested):
                self._resolved_model_path = requested
                self._model = YOLO(requested)
            elif os.path.exists(onnx_fallback):
                # Use the bundled ONNX file if present
                self._resolved_model_path = onnx_fallback
                self._model = YOLO(onnx_fallback)
            elif os.path.exists(pt_fallback):
                self._resolved_model_path = pt_fallback
                self._model = YOLO(pt_fallback)
            else:
                # Download YOLO model if not present locally
                print(
                    f"Model not found at {requested}. "
                    f"Also missing {onnx_fallback} and {pt_fallback}; attempting to download..."
                )
                self._resolved_model_path = None
                self._model = YOLO("yolov8n-face.pt")

            # fuse() only for PyTorch checkpoints; ONNX/exported formats must not call fuse()
            if self._resolved_model_path is None or str(
                self._resolved_model_path
            ).lower().endswith(".pt"):
                self._model.fuse()
            self._initialized = True

            if CONFIG.VERBOSE:
                model_desc = self._resolved_model_path or "auto-downloaded yolov8n-face.pt"
                print(f"YOLO face detection initialized successfully ({model_desc})")

            return True

        except ImportError:
            self.last_init_error = (
                "ultralytics is not installed. Run: pip install ultralytics"
            )
            print("Error: ultralytics package not installed")
            print("Install with: pip install ultralytics")
            return False
        except Exception as e:
            self.last_init_error = f"YOLO face model failed to load: {e}"
            print(f"Error initializing YOLO model: {e}")
            return False

    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in a frame.

        Args:
            frame: BGR image (numpy array)

        Returns:
            List of FaceDetection objects
        """
        if not self._initialized:
            if not self.initialize():
                return []

        if frame is None or frame.size == 0:
            return []

        try:
            results = self._model(
                frame,
                imgsz=CONFIG.model.YOLO_INPUT_SIZE,
                conf=CONFIG.model.YOLO_CONF_THRESHOLD,
                verbose=False
            )

            faces = []

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes

                for i in range(len(boxes)):
                    # Get bounding box
                    bbox = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = map(int, bbox)

                    # Get confidence
                    conf = float(boxes.conf[i].cpu().numpy())

                    # Get keypoints if available
                    keypoints = None
                    if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                        try:
                            kps = results[0].keypoints.xy[i].cpu().numpy()
                            keypoints = [(float(kp[0]), float(kp[1])) for kp in kps]
                        except:
                            pass

                    faces.append(FaceDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        keypoints=keypoints
                    ))

            # Sort by confidence (highest first)
            faces.sort(key=lambda f: f.confidence, reverse=True)

            return faces

        except Exception as e:
            if CONFIG.VERBOSE:
                print(f"Error during face detection: {e}")
            return []

    def get_primary_face(self, frame: np.ndarray) -> Optional[FaceDetection]:
        """
        Get the primary (highest confidence) face in a frame.

        Args:
            frame: BGR image

        Returns:
            FaceDetection object or None if no face found
        """
        faces = self.detect_faces(frame)
        return faces[0] if faces else None

    def get_face_count(self, frame: np.ndarray) -> int:
        """
        Get the number of faces detected in a frame.

        Args:
            frame: BGR image

        Returns:
            Number of faces detected
        """
        return len(self.detect_faces(frame))

    def extract_face_crop(self,
                          frame: np.ndarray,
                          bbox: Tuple[int, int, int, int],
                          padding_ratio: float = 0.2) -> Optional[np.ndarray]:
        """
        Extract face crop from frame with optional padding.

        Args:
            frame: BGR image
            bbox: Face bounding box (x1, y1, x2, y2)
            padding_ratio: Padding ratio to add around bbox

        Returns:
            Cropped face image or None if invalid
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Add padding
        pad_x = int((x2 - x1) * padding_ratio)
        pad_y = int((y2 - y1) * padding_ratio)

        x1_padded = max(0, x1 - pad_x)
        y1_padded = max(0, y1 - pad_y)
        x2_padded = min(w, x2 + pad_x)
        y2_padded = min(h, y2 + pad_y)

        crop = frame[y1_padded:y2_padded, x1_padded:x2_padded]

        if crop.size == 0:
            return None

        return crop