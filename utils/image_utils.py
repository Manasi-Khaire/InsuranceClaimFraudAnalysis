"""
Purpose:
    Utility functions for image validation and a lightweight damage-assessment proxy.

Related PRD:
    - FR-02: MVP image validation through a pretrained YOLOv8n-based proxy.
    - Architecture: Damage Assessment Agent uses a pretrained YOLOv8n proxy.

Important design note:
    This module deliberately does not claim true dent/scratch/crack detection.
    It is intended only as a workflow validation proxy until a future VehiDE-based
    damage-classification model is integrated.

Inputs:
    - vehicle_image_path: path to a local image file

Outputs:
    - Dictionary with:
        * damage_detected
        * damage_confidence
        * damage_severity_score
        * damage_assessment_summary

Dependencies:
    - ultralytics (pretrained YOLOv8n)
    - Python standard library only for file checks

Integration:
    The helper is consumed by the damage assessment agent without affecting the
    existing LangGraph workflow or shared ClaimState contract.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
VEHIDE_DATASET_PATH = os.environ.get(
    "VEHIDE_DATASET_PATH",
    r"C:\Users\myogeshk\Downloads\archive (1)",
)


# -----------------------------------------------------------------------------
# Public helpers
# -----------------------------------------------------------------------------
def get_vehide_dataset_path() -> str:
    """Return the configured VehiDE dataset path for future integration."""
    return VEHIDE_DATASET_PATH


def validate_vehicle_image_path(vehicle_image_path: str) -> str:
    """Validate and normalize a local vehicle image path."""
    if not vehicle_image_path or not str(vehicle_image_path).strip():
        raise ValueError("vehicle_image_path is required")

    normalized_path = os.path.abspath(os.path.expanduser(str(vehicle_image_path)))

    if not os.path.exists(normalized_path):
        raise FileNotFoundError(f"Vehicle image not found: {normalized_path}")

    if not os.path.isfile(normalized_path):
        raise ValueError(f"Vehicle image path is not a file: {normalized_path}")

    return normalized_path


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _estimate_region_label(image_shape: Tuple[int, int, int], centroid_x: float, centroid_y: float) -> str:
    """Estimate which part of the car the damage is concentrated in."""
    height, width = image_shape[:2]

    if centroid_x < width * 0.33:
        side = "left-side"
    elif centroid_x > width * 0.66:
        side = "right-side"
    else:
        side = "front bumper"

    if centroid_y < height * 0.35:
        location = "upper"
    elif centroid_y < height * 0.7:
        location = "mid"
    else:
        location = "lower"

    return f"{side} {location} region"


def _severity_label_from_percentage(damage_area_percentage: float) -> str:
    if damage_area_percentage < 5:
        return "Minor"
    if damage_area_percentage < 20:
        return "Moderate"
    if damage_area_percentage < 45:
        return "Severe"
    return "Critical"


def _extract_opencv_damage_features(image_path: str, vehicle_box: Optional[list] = None) -> Dict[str, float]:
    """Compute image-structure signals for damage estimation without training."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return {
            "edge_density": 0.0,
            "contour_density": 0.0,
            "contour_coverage_percentage": 0.0,
            "damaged_area_ratio": 0.0,
            "texture_irregularity": 0.0,
            "brightness_variance": 0.0,
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edge_density = float(np.mean(edges > 0) / 255.0)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_density = min(1.0, len(contours) / max(1, (gray.shape[0] * gray.shape[1]) / 1500.0))

    laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    texture_irregularity = _clip01(float(np.std(laplacian) / 80.0))
    brightness_variance = _clip01(float(np.std(gray) / 80.0))

    contour_coverage_percentage = 0.0
    if contours:
        contour_areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 25]
        total_contour_area = float(sum(contour_areas))
        vehicle_area = float(gray.shape[0] * gray.shape[1])
        if vehicle_box and len(vehicle_box) >= 4:
            x1, y1, x2, y2 = [max(0.0, float(v)) for v in vehicle_box[:4]]
            vehicle_w = max(1.0, abs(x2 - x1))
            vehicle_h = max(1.0, abs(y2 - y1))
            vehicle_area = max(1.0, vehicle_w * vehicle_h)
        contour_coverage_percentage = min(100.0, (total_contour_area / max(vehicle_area, 1.0)) * 100.0)

    damaged_area_ratio = _clip01(contour_coverage_percentage / 100.0)

    return {
        "edge_density": edge_density,
        "contour_density": contour_density,
        "contour_coverage_percentage": contour_coverage_percentage,
        "damaged_area_ratio": damaged_area_ratio,
        "texture_irregularity": texture_irregularity,
        "brightness_variance": brightness_variance,
    }


def _extract_yolo_vehicle_proxy(image_path: str) -> Dict[str, Any]:
    """Use YOLOv8n only to localize the vehicle and estimate confidence."""
    try:
        from ultralytics import YOLO
    except Exception:
        return {"vehicle_confidence": 0.0, "vehicle_detected": False, "vehicle_label": "unknown", "vehicle_box": None}

    model = YOLO("yolov8n.pt")
    results = model(image_path, verbose=False, conf=0.20, device="cpu")

    best_confidence = 0.0
    detected_label = "unknown"
    vehicle_box = None

    for result in results:
        for box in getattr(result, "boxes", []):
            if hasattr(box, "cls") and hasattr(box, "conf") and hasattr(box, "xyxy"):
                cls_ids = getattr(box, "cls")
                confs = getattr(box, "conf")
                xyxy = getattr(box, "xyxy")
                if hasattr(cls_ids, "cpu"):
                    cls_ids = cls_ids.cpu().numpy().tolist()
                if hasattr(confs, "cpu"):
                    confs = confs.cpu().numpy().tolist()
                if hasattr(xyxy, "cpu"):
                    xyxy = xyxy.cpu().numpy().tolist()
                if not isinstance(cls_ids, list):
                    cls_ids = [cls_ids]
                if not isinstance(confs, list):
                    confs = [confs]
                if not isinstance(xyxy, list):
                    xyxy = [xyxy]

                for cls_id, conf, box_xy in zip(cls_ids, confs, xyxy):
                    label = result.names.get(int(cls_id), "vehicle") if hasattr(result, "names") else "vehicle"
                    if any(keyword in str(label).lower() for keyword in ["car", "truck", "bus", "motorcycle", "bicycle", "vehicle"]):
                        if float(conf) > best_confidence:
                            best_confidence = float(conf)
                            detected_label = str(label)
                            vehicle_box = [float(v) for v in box_xy]

    return {
        "vehicle_confidence": float(best_confidence),
        "vehicle_detected": best_confidence >= 0.20,
        "vehicle_label": detected_label,
        "vehicle_box": vehicle_box,
    }


def _build_hybrid_summary(
    damage_severity_label: str,
    damage_detected: bool,
    damaged_area_percent: float,
    region_label: str,
    vehicle_confidence: float,
    contour_density: float,
    internal_components_visible: bool,
) -> str:
    """Generate a summary that explains the driver behind the assigned severity."""
    damaged_area_pct = max(0.0, round(float(damaged_area_percent), 1))

    if not damage_detected:
        return (
            "Limited visual irregularities detected in the vehicle image. "
            f"Estimated damaged area: {damaged_area_pct}%. Contour density: {contour_density:.2f}. "
            f"Severity classified as {damage_severity_label}. This is a YOLO proxy assessment only and does not claim true dent/scratch/crack detection."
        )

    if damaged_area_pct >= 45:
        description = (
            f"Large damaged area detected around {region_label}. Internal vehicle components visible. "
            f"Estimated damaged area {damaged_area_pct}%. Contour density: {contour_density:.2f}. "
            f"Severity classified as {damage_severity_label} because the damaged region is extensive and structurally significant."
        )
    elif damaged_area_pct >= 20:
        description = (
            f"Substantial damaged area detected around {region_label}. Estimated damaged area {damaged_area_pct}%. "
            f"Contour density: {contour_density:.2f}. Severity classified as {damage_severity_label} based on the visible damaged region and high contour concentration."
        )
    elif damaged_area_pct >= 5:
        description = (
            f"Noticeable damaged area detected around {region_label}. Estimated damaged area {damaged_area_pct}%. "
            f"Contour density: {contour_density:.2f}. Severity classified as {damage_severity_label} due to a moderate but visible damaged zone."
        )
    else:
        description = (
            f"Minor irregularities detected around {region_label}. Estimated damaged area {damaged_area_pct}%. "
            f"Contour density: {contour_density:.2f}. Severity classified as {damage_severity_label}."
        )

    if internal_components_visible:
        description += " Internal vehicle components are visible, which increases the damage severity estimate."

    description += (
        f" YOLO confidence was {vehicle_confidence:.2f}; this is a YOLO proxy assessment and not a true dent/scratch/crack detector."
    )
    return description


def assess_vehicle_damage(
    vehicle_image_path: str,
    model: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Assess vehicle condition using a hybrid MVP approach:
        1. YOLOv8n vehicle detection for localization/confidence
        2. OpenCV image-structure metrics for edge and contour irregularity analysis

    This is a proxy and not a true damage classifier. It is intended for demo-ready
    local evaluation without training or fine-tuning.
    """
    image_path = validate_vehicle_image_path(vehicle_image_path)

    try:
        yolo_signal = _extract_yolo_vehicle_proxy(image_path)
        vehicle_box = yolo_signal.get("vehicle_box")
        opencv_signal = _extract_opencv_damage_features(image_path, vehicle_box=vehicle_box)
    except Exception:
        return {
            "damage_detected": False,
            "damage_confidence": 0.0,
            "damage_severity_score": 0.0,
            "damage_assessment_summary": (
                "YOLO + OpenCV proxy could not process the image. This is a local "
                "demo placeholder and not a true VehiDE damage classification model."
            ),
        }

    vehicle_confidence = float(yolo_signal.get("vehicle_confidence", 0.0))
    edge_density = float(opencv_signal.get("edge_density", 0.0))
    contour_density = float(opencv_signal.get("contour_density", 0.0))
    texture_irregularity = float(opencv_signal.get("texture_irregularity", 0.0))
    brightness_variance = float(opencv_signal.get("brightness_variance", 0.0))
    contour_coverage_percentage = float(opencv_signal.get("contour_coverage_percentage", 0.0))
    damaged_area_ratio = float(opencv_signal.get("damaged_area_ratio", contour_coverage_percentage / 100.0))

    damage_score = 0.20 * vehicle_confidence + 0.50 * damaged_area_ratio + 0.30 * contour_density
    damage_severity_score = _clip01(float(damage_score))

    damage_detected = damage_severity_score >= 0.08 or contour_coverage_percentage >= 3.0
    damage_confidence = _clip01(0.55 * vehicle_confidence + 0.45 * damage_severity_score)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    region_label = "vehicle body region"
    if image is not None:
        contours, _ = cv2.findContours(cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0), 50, 150), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            merged = np.concatenate([cnt.reshape(-1, 2) for cnt in contours[:10]], axis=0)
            centroid_x = float(np.mean(merged[:, 0]))
            centroid_y = float(np.mean(merged[:, 1]))
            region_label = _estimate_region_label(image.shape, centroid_x, centroid_y)

    damage_area_percentage = round(float(contour_coverage_percentage), 2)
    damage_severity_label = _severity_label_from_percentage(damage_area_percentage)
    internal_components_visible = damage_area_percentage >= 10 and (texture_irregularity > 0.35 or brightness_variance > 0.35)

    damage_summary = _build_hybrid_summary(
        damage_severity_label=damage_severity_label,
        damage_detected=damage_detected,
        damaged_area_percent=damage_area_percentage,
        region_label=region_label,
        vehicle_confidence=vehicle_confidence,
        contour_density=contour_density,
        internal_components_visible=internal_components_visible,
    )

    return {
        "damage_detected": bool(damage_detected),
        "damage_confidence": round(float(damage_confidence), 4),
        "damage_severity_score": round(float(damage_severity_score), 4),
        "damage_area_percentage": round(float(damage_area_percentage), 2),
        "contour_coverage_percentage": round(float(contour_coverage_percentage), 2),
        "edge_density": round(float(edge_density), 4),
        "contour_density": round(float(contour_density), 4),
        "damaged_area_ratio": round(float(damaged_area_ratio), 4),
        "damage_severity_label": damage_severity_label,
        "damage_assessment_summary": damage_summary,
    }
