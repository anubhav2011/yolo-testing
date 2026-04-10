"""
Main entry point for proctoring video analysis.
Designed for execution in Google Colab.
"""

import os
import sys
import json
import time
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG, ProctoringConfig
from utils.video_utils import VideoProcessor
from utils.time_utils import TimeFormatter
from services.detection_service import DetectionService
from services.pose_service import PoseService
from services.gaze_service import GazeService
from tracking.gesture_tracker import GestureTracker


def install_dependencies():
    """Install required dependencies in Colab environment."""
    import subprocess

    print("Installing dependencies...")
    packages = [
        "ultralytics",
        "onnxruntime",
        "mediapipe",
        "opencv-python-headless"
    ]

    for package in packages:
        try:
            __import__(package.replace("-", "_").split("[")[0])
            print(f"  ✓ {package} already installed")
        except ImportError:
            print(f"  Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                package, "-q"
            ])
            print(f"  ✓ {package} installed")

    print("All dependencies installed!\n")


def download_models():
    """Download required model files."""
    os.makedirs("models", exist_ok=True)

    # YOLOv8n-face will be auto-downloaded by ultralytics
    print("Models will be downloaded automatically on first use.\n")


class ProctoringAnalyzer:
    """
    Main proctoring analyzer class.

    Processes video files and generates gesture reports.
    """

    def __init__(self, config: Optional[ProctoringConfig] = None):
        """
        Initialize analyzer.

        Args:
            config: Configuration object (uses global CONFIG if None)
        """
        self.config = config or CONFIG

        # Initialize services
        self.detection_service = DetectionService()
        self.pose_service = PoseService()
        self.gaze_service = GazeService()
        self.gesture_tracker = GestureTracker()

        self._initialized = False
        self._init_failure_detail: Optional[str] = None

    def initialize(self) -> bool:
        """
        Initialize all services.

        Returns:
            True if all services initialized successfully
        """
        self._init_failure_detail = None
        print("Initializing services...")

        # Detection service
        print("  Initializing face detection (YOLO)...")
        if not self.detection_service.initialize():
            self._init_failure_detail = (
                getattr(self.detection_service, "last_init_error", None)
                or "Face detection initialization failed"
            )
            print("  ✗ Face detection initialization failed")
            return False
        print("  ✓ Face detection ready")

        # Pose service
        print("  Initializing head pose estimation...")
        if not self.pose_service.initialize():
            self._init_failure_detail = (
                getattr(self.pose_service, "last_init_error", None)
                or "Head pose estimation initialization failed"
            )
            print("  ✗ Pose estimation initialization failed")
            return False
        print("  ✓ Head pose estimation ready")

        # Gaze service
        print("  Initializing gaze estimation...")
        if not self.gaze_service.initialize():
            self._init_failure_detail = (
                getattr(self.gaze_service, "last_init_error", None)
                or "Gaze estimation initialization failed"
            )
            print("  ✗ Gaze estimation initialization failed")
            return False
        print("  ✓ Gaze estimation ready")

        self._initialized = True
        print("All services initialized!\n")
        return True

    def analyze_video(self,
                      video_path: str,
                      progress_callback: Optional[callable] = None) -> dict:
        """
        Analyze a video file and generate gesture report.

        Args:
            video_path: Path to video file
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            Dictionary containing gesture analysis results
        """
        if not self._initialized:
            if not self.initialize():
                return {
                    "error": self._init_failure_detail or "Failed to initialize services",
                }

        # Reset tracker for new video
        self.gesture_tracker.reset()

        print(f"Analyzing video: {video_path}")
        start_time = time.time()

        try:
            with VideoProcessor(video_path) as video:
                total_frames = video.total_frames
                processed_frames = 0

                # Calculate frame skip for target FPS
                skip_frames = max(0, int(video.fps / self.config.video.TARGET_FPS) - 1)

                print(f"Processing at ~{self.config.video.TARGET_FPS} FPS "
                      f"(skip every {skip_frames} frames)")
                print("-" * 50)

                for frame_data in video.get_frames(skip_frames=skip_frames):
                    frame = frame_data.frame
                    timestamp = frame_data.timestamp

                    # Detect faces
                    face = self.detection_service.get_primary_face(frame)
                    face_detected = face is not None

                    head_direction = None
                    head_intensity = 0.0
                    eye_direction = None
                    eye_intensity = 0.0

                    if face_detected:
                        # Estimate head pose
                        pose = self.pose_service.estimate_pose(
                            frame,
                            face.bbox,
                            face.keypoints
                        )

                        if pose is not None:
                            head_direction = pose.get_primary_direction()
                            head_intensity = pose.get_intensity()

                        # Estimate gaze
                        gaze = self.gaze_service.estimate_gaze(frame, face.bbox)

                        if gaze is not None and not gaze.is_blinking:
                            eye_direction = gaze.get_primary_direction()
                            eye_intensity = max(abs(gaze.horizontal), abs(gaze.vertical))

                    # Update gesture tracker
                    self.gesture_tracker.update(
                        timestamp=timestamp,
                        face_detected=face_detected,
                        head_direction=head_direction,
                        head_intensity=head_intensity,
                        eye_direction=eye_direction,
                        eye_intensity=eye_intensity
                    )

                    processed_frames += 1

                    # Progress update
                    if progress_callback:
                        progress_callback(frame_data.frame_number, total_frames)
                    elif processed_frames % 100 == 0:
                        progress = (frame_data.frame_number / total_frames) * 100
                        print(f"  Progress: {progress:.1f}% "
                              f"({frame_data.frame_number}/{total_frames} frames)")

                print("-" * 50)

        except FileNotFoundError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Processing error: {str(e)}"}

        # Generate report
        report = self.gesture_tracker.generate_report()

        elapsed = time.time() - start_time
        print(f"Analysis complete in {elapsed:.2f} seconds")
        print(f"Processed {processed_frames} frames")

        return report.to_dict()

    def analyze_video_json(self,
                           video_path: str,
                           output_path: Optional[str] = None,
                           indent: int = 2) -> str:
        """
        Analyze video and return JSON string.

        Args:
            video_path: Path to video file
            output_path: Optional path to save JSON file
            indent: JSON indentation level

        Returns:
            JSON string with analysis results
        """
        result = self.analyze_video(video_path)
        json_str = json.dumps(result, indent=indent)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(json_str)
            print(f"Results saved to: {output_path}")

        return json_str


def run_analysis(video_path: str,
                 output_path: Optional[str] = None,
                 verbose: bool = True) -> dict:
    """
    Convenience function to run complete analysis.

    Args:
        video_path: Path to video file
        output_path: Optional path to save JSON results
        verbose: Whether to print progress information

    Returns:
        Analysis results dictionary
    """
    CONFIG.VERBOSE = verbose

    analyzer = ProctoringAnalyzer()
    result = analyzer.analyze_video(video_path)

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=CONFIG.OUTPUT_JSON_INDENT)
        if verbose:
            print(f"\nResults saved to: {output_path}")

    return result


# ============================================================================
# COLAB EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Check if running in Colab
    try:
        import google.colab

        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    print("=" * 60)
    print("PROCTORING VIDEO ANALYZER")
    print("YOLO + WHENet + MediaPipe Pipeline")
    print("=" * 60 + "\n")

    # Install dependencies if in Colab
    if IN_COLAB:
        install_dependencies()

    # Download models
    download_models()

    # ========================================================================
    # USAGE EXAMPLE - Modify this section for your video
    # ========================================================================

    # Option 1: Upload video in Colab
    if IN_COLAB:
        from google.colab import files

        print("Please upload a video file:")
        uploaded = files.upload()

        if uploaded:
            video_path = list(uploaded.keys())[0]
            print(f"\nAnalyzing uploaded video: {video_path}\n")

            # Run analysis
            result = run_analysis(video_path, output_path="analysis_result.json")

            # Print JSON result
            print("\n" + "=" * 60)
            print("ANALYSIS RESULT:")
            print("=" * 60)
            print(json.dumps(result, indent=2))

            # Download result
            print("\n\nDownloading result file...")
            files.download("analysis_result.json")
        else:
            print("No file uploaded.")

    # Option 2: Local execution with command line argument
    else:
        if len(sys.argv) > 1:
            video_path = sys.argv[1]
            output_path = sys.argv[2] if len(sys.argv) > 2 else "analysis_result.json"

            result = run_analysis(video_path, output_path=output_path)

            print("\n" + "=" * 60)
            print("ANALYSIS RESULT:")
            print("=" * 60)
            print(json.dumps(result, indent=2))
        else:
            print("Usage: python main.py <video_path> [output_path]")
            print("\nExample:")
            print("  python main.py test_video.mp4")
            print("  python main.py test_video.mp4 results.json")


# ============================================================================
# COLAB NOTEBOOK HELPER FUNCTIONS
# ============================================================================

def colab_analyze(video_path: str) -> dict:
    """
    Helper function for Colab notebook cells.

    Usage in Colab:
        from main import colab_analyze
        result = colab_analyze("/content/my_video.mp4")
    """
    return run_analysis(video_path)


def colab_analyze_with_display(video_path: str):
    """
    Analyze and display results in Colab with formatting.
    """
    result = run_analysis(video_path)

    print("\n" + "=" * 60)
    print("📊 GESTURE ANALYSIS REPORT")
    print("=" * 60 + "\n")

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return result

    gestures = result.get("gestures", [])

    if not gestures:
        print("✅ No suspicious gestures detected!")
        return result

    for gesture in gestures:
        name = gesture["name"]
        occurrences = gesture["occurrence"]

        emoji = {"head_movement": "🔄", "eye_movement": "👁️", "face_missing": "👤"}.get(name, "📌")
        print(f"{emoji} {name.upper().replace('_', ' ')}: {len(occurrences)} occurrence(s)")

        for i, occ in enumerate(occurrences, 1):
            timestamp = occ["timestamp"]
            duration = occ["duration"]
            direction = occ.get("direction", "N/A")
            intensity = occ.get("intensity", "N/A")

            print(f"   {i}. Time: {timestamp} | Duration: {duration}s | "
                  f"Direction: {direction} | Intensity: {intensity}°")
        print()

    return result