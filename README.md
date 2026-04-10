# Proctoring YOLO + Pose (Colab-only)

This project processes a video and returns a JSON summary of **head movement**, **eye movement**, and **face missing** events in the required format.

## Folder structure

```
proctoring_yolo_pose/
├── models/
│   ├── model.onnx
│   └── WHENet.onnx
├── services/
│   ├── detection_service.py
│   ├── gaze_service.py
│   └── pose_service.py
├── tracking/
│   └── gesture_tracker.py
├── utils/
│   ├── time_utils.py
│   └── video_utils.py
├── main.py
└── config.py
```

## Colab setup

Install dependencies:

```bash
pip install ultralytics onnxruntime mediapipe opencv-python numpy
```

Upload your models to `models/`:
- `models/model.onnx` (YOLO face detector; OR you can omit it and let Ultralytics auto-download)
- `models/WHENet.onnx`

Run:

```bash
python main.py --video /path/to/video.mp4 --output_json
```

## Output JSON

`main.py` prints JSON like:

```json
{
  "gestures": [
    { "name": "head_movement", "occurrence": [ ... ] },
    { "name": "eye_movement",  "occurrence": [ ... ] },
    { "name": "face_missing",  "occurrence": [ ... ] }
  ]
}
```

