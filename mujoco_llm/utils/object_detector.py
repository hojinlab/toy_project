# utils/object_detector.py
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import numpy as np

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    YOLO = None  # type: ignore
    _YOLO_AVAILABLE = False


@dataclass
class DetectedBox:
    confidence: float
    bbox: list  # [x1, y1, x2, y2]
    center: list  # [cx, cy]


class ObjectDetector:
    """
    YOLO detector adapter.

    - Keeps ultralytics dependency inside utils layer
    - Provides structured dict output for downstream logic
    - Thread-safe inference with internal lock
    """

    def __init__(self, weight_path: str, conf: float = 0.5):
        if not _YOLO_AVAILABLE:
            raise ImportError("ultralytics is not installed. Install it to use ObjectDetector.")
        self.weight_path = weight_path
        self.conf = conf
        self.model = YOLO(weight_path)
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return _YOLO_AVAILABLE and self.model is not None

    def detect_dict(self, frame_bgr: np.ndarray) -> Dict[str, List[dict]]:
        """
        Parameters
        - frame_bgr: np.ndarray (H, W, 3), BGR image

        Returns
        - dict[label] = list of {confidence, bbox, center}
        """
        if frame_bgr is None:
            return {}

        with self._lock:
            result = self.model(frame_bgr, verbose=False, conf=self.conf)[0]

        out: Dict[str, List[dict]] = {}
        for box in result.boxes:
            cls = int(box.cls[0].item())
            label = result.names.get(cls)
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            out.setdefault(label, []).append({
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2],
                "center": [(x1 + x2) / 2, (y1 + y2) / 2],
            })
        return out

    def detect_image(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns
        - np.ndarray: plotted BGR image with bounding boxes, or None
        """
        if frame_bgr is None:
            return None

        with self._lock:
            result = self.model(frame_bgr, verbose=False, conf=self.conf)[0]
        return result.plot()
