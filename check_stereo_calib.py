"""Offline calibration-quality check: detects ArUco markers shared by both
cameras in a captured frame pair and measures how far each marker corner
falls from its predicted epipolar line under the current calibration data.
No LiveKit/RTSP/OpenDIBR needed — same input shape as
scratch_test_capture*.py (a DibrDepthTestCaptures/<pair>/ directory with
<cam>.png + calibration.json).

This directly measures calibration quality (are the two cameras' extrinsics
actually consistent with what the cameras see?), independent of
StereoDepthComputer's own code — use it to validate a calibration/orientation
fix BEFORE re-testing live depth. Per the peer review that flagged the
calibration/orientation mismatch as dibr-bridge's real root cause: target
~2px median epipolar error before trusting depth output again.

Board config (dictionary, layout) is hardcoded to match the project's actual
ChArUco board -- confirmed empirically against real capture data, NOT
Master-Thesis-Client's CameraCalibrationData.cs default (which says
DICT_5X5_250; the real board the registration-service currently configures
uses DICT_6X6_250 -- only the dictionary matters here, not square/marker
size, since we only need marker-corner pixel correspondences, not a fresh
calibration).
"""
from __future__ import annotations

import json
import sys

import cv2
import numpy as np

from bridge.registration_client import CameraCalibration

ARUCO_DICT = cv2.aruco.DICT_6X6_250
BOARD_SQUARES = (5, 7)  # (squaresX, squaresY), from CameraCalibrationData.cs


def _camera_matrix(intr) -> np.ndarray:
    return np.array([
        [intr.fx, 0, intr.cx],
        [0, intr.fy, intr.cy],
        [0, 0, 1],
    ], dtype=np.float64)


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def _detect_markers(img: np.ndarray) -> dict[int, np.ndarray]:
    """Returns {marker_id: (4,2) corner array}, in ArUco's fixed
    top-left/top-right/bottom-right/bottom-left order per marker."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _rejected = detector.detectMarkers(img)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for c, i in zip(corners, ids.ravel())}


def _fundamental_matrix(calib_a: CameraCalibration, calib_b: CameraCalibration,
                         size_a: tuple[int, int], size_b: tuple[int, int]) -> np.ndarray:
    """F such that pt_b^T @ F @ pt_a ~= 0 for a true correspondence
    (pt_a in camera A's image, pt_b in camera B's). Scales each camera's
    intrinsics from ITS OWN calibration resolution to the ACTUAL loaded
    image resolution first -- same per-camera scaling stereo_depth.py uses,
    since captured frames may be at a different resolution than calibration
    (e.g. LiveKit's published resolution vs. the calibration-time capture)."""
    Ka = _camera_matrix(calib_a.intrinsics)
    Kb = _camera_matrix(calib_b.intrinsics)
    calib_size_a = (calib_a.intrinsics.image_width, calib_a.intrinsics.image_height)
    calib_size_b = (calib_b.intrinsics.image_width, calib_b.intrinsics.image_height)
    for K, calib_size, actual_size in ((Ka, calib_size_a, size_a), (Kb, calib_size_b, size_b)):
        if actual_size != calib_size:
            sx = actual_size[0] / calib_size[0]
            sy = actual_size[1] / calib_size[1]
            K[0, 0] *= sx; K[0, 2] *= sx
            K[1, 1] *= sy; K[1, 2] *= sy

    Ra, _ = cv2.Rodrigues(np.array(calib_a.extrinsics.rvec, dtype=np.float64))
    ta = np.array(calib_a.extrinsics.tvec, dtype=np.float64).reshape(3, 1)
    Rb, _ = cv2.Rodrigues(np.array(calib_b.extrinsics.rvec, dtype=np.float64))
    tb = np.array(calib_b.extrinsics.tvec, dtype=np.float64).reshape(3, 1)
    R_rel = Rb @ Ra.T
    t_rel = tb - R_rel @ ta

    E = _skew(t_rel.ravel()) @ R_rel
    F = np.linalg.inv(Kb).T @ E @ np.linalg.inv(Ka)
    return F, Ka, Kb, np.linalg.norm(t_rel)


def _point_to_line_dist(line: np.ndarray, pt: np.ndarray) -> float:
    a, b, c = line
    return abs(a * pt[0] + b * pt[1] + c) / np.hypot(a, b)


def check(capture_dir: str) -> None:
    with open(f"{capture_dir}/calibration.json") as f:
        raw = json.load(f)
    calib_a = CameraCalibration.from_json(raw["cam1"])
    calib_b = CameraCalibration.from_json(raw["cam2"])

    img_a = cv2.imread(f"{capture_dir}/{calib_a.camera_name}.png")
    img_b = cv2.imread(f"{capture_dir}/{calib_b.camera_name}.png")
    size_a = (img_a.shape[1], img_a.shape[0])
    size_b = (img_b.shape[1], img_b.shape[0])

    print(f"Pair: {calib_a.camera_name} (calib {calib_a.intrinsics.image_width}x"
          f"{calib_a.intrinsics.image_height}, actual {size_a[0]}x{size_a[1]})  <->  "
          f"{calib_b.camera_name} (calib {calib_b.intrinsics.image_width}x"
          f"{calib_b.intrinsics.image_height}, actual {size_b[0]}x{size_b[1]})")

    markers_a = _detect_markers(img_a)
    markers_b = _detect_markers(img_b)
    shared_ids = sorted(set(markers_a) & set(markers_b))
    print(f"Detected {len(markers_a)} markers in {calib_a.camera_name}, "
          f"{len(markers_b)} in {calib_b.camera_name}, {len(shared_ids)} shared IDs: {shared_ids}")
    if len(shared_ids) < 3:
        print("Too few shared markers for a meaningful epipolar check "
              "(need the board visible to both cameras at once).")
        return

    F, _Ka, _Kb, baseline_m = _fundamental_matrix(calib_a, calib_b, size_a, size_b)

    dists = []
    for marker_id in shared_ids:
        for corner_a, corner_b in zip(markers_a[marker_id], markers_b[marker_id]):
            pt_a = np.array([corner_a[0], corner_a[1], 1.0])
            pt_b = np.array([corner_b[0], corner_b[1], 1.0])
            line_in_b = F @ pt_a
            line_in_a = F.T @ pt_b
            dists.append(_point_to_line_dist(line_in_b, pt_b))
            dists.append(_point_to_line_dist(line_in_a, pt_a))

    dists = np.array(dists)
    median_err = float(np.median(dists))
    max_err = float(np.max(dists))
    print(f"baseline (from calibration): {baseline_m:.3f} m")
    print(f"epipolar error over {len(dists)} corner correspondences "
          f"({len(shared_ids)} markers x 4 corners x 2 directions):")
    print(f"  median = {median_err:.1f} px, max = {max_err:.1f} px")
    verdict = "PASS" if median_err <= 2.0 else "FAIL"
    print(f"  [{verdict}] target: median <= ~2px (per review threshold)")


if __name__ == "__main__":
    capture_dir = sys.argv[1] if len(sys.argv) > 1 else \
        "../DibrDepthTestCaptures/cam3_cam2_20260921_181148"
    check(capture_dir)
