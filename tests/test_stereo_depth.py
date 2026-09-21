"""Tests for StereoDepthComputer.

Fixture: a fronto-parallel textured plane at known depth Z, two cameras
separated by baseline b along x with no rotation.  With pure-x translation
and zero distortion, rectification is near-identity and the two views differ
by a pure horizontal pixel shift d = fx*b/Z.

No LiveKit or RTSP dependency — pure cv2/numpy.
"""
import numpy as np
import pytest

from bridge.stereo_depth import StereoDepthComputer
from bridge.registration_client import CameraCalibration, Intrinsics, Extrinsics


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_calib(fx: float, fy: float, cx: float, cy: float,
                W: int, H: int, tvec: list[float]) -> CameraCalibration:
    return CameraCalibration(
        camera_name="test",
        intrinsics=Intrinsics(
            fx=fx, fy=fy, cx=cx, cy=cy,
            dist_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
            image_width=W, image_height=H,
            reprojection_error=0.0,
        ),
        extrinsics=Extrinsics(rvec=[0.0, 0.0, 0.0], tvec=tvec),
    )


def _shifted_pair(W: int, H: int, shift: int, seed: int = 42):
    """Return (frame_a, frame_b) where frame_b is frame_a shifted left by
    `shift` pixels — simulating a fronto-parallel plane seen from a camera
    offset to the right by baseline b (disparity = fx*b/Z)."""
    rng = np.random.default_rng(seed)
    frame_a = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    frame_b = np.roll(frame_a, -shift, axis=1)
    return frame_a, frame_b


def _median_depth(depth: np.ndarray) -> float:
    valid = depth[depth > 0]
    return float(np.median(valid))


# ── Test 1 — both cameras must have non-empty depth (Bug 1 regression) ──────

def test_both_cameras_have_valid_depth():
    """Both depth_a and depth_b must be mostly non-zero.

    Before the Bug-1 fix, StereoSGBM.compute(gray_b, gray_a) returns all-zero
    for the right camera because minDisparity=0 only yields valid positive
    disparities when the left image is given first.  depth_b.mean()>0 fails
    on the unfixed code.

    W=1280 keeps the valid fraction above 50% for numDisparities=384:
    (1280-384)/1280 = 70%.  640-wide images only give 40% and would fail
    the >0.5 threshold even with correct disparity.
    """
    W, H = 1280, 720
    fx = 1000.0; b = 0.1; Z = 2.0
    d = int(fx * b / Z)          # 50 px

    calib_a = _make_calib(fx, fx, W / 2, H / 2, W, H, [0.0, 0.0, 0.0])
    calib_b = _make_calib(fx, fx, W / 2, H / 2, W, H, [b,   0.0, 0.0])
    computer = StereoDepthComputer(calib_a, calib_b)

    frame_a, frame_b = _shifted_pair(W, H, d)
    depth_a, depth_b = computer.compute_pair(frame_a, frame_b)

    assert (depth_a > 0).mean() > 0.5, "cam A depth mostly empty"
    assert (depth_b > 0).mean() > 0.5, "cam B depth mostly empty — Bug 1 not fixed"


# ── Test 2 — recovered depth must be metrically accurate ────────────────────

def test_metric_depth_accuracy():
    """Median recovered depth must be within 15 % of the true Z for both cams."""
    W, H = 1280, 720
    fx = 1000.0; b = 0.1; Z = 2.0
    d = int(fx * b / Z)

    calib_a = _make_calib(fx, fx, W / 2, H / 2, W, H, [0.0, 0.0, 0.0])
    calib_b = _make_calib(fx, fx, W / 2, H / 2, W, H, [b,   0.0, 0.0])
    computer = StereoDepthComputer(calib_a, calib_b)

    frame_a, frame_b = _shifted_pair(W, H, d)
    depth_a, depth_b = computer.compute_pair(frame_a, frame_b)

    for name, depth in (("A", depth_a), ("B", depth_b)):
        med = _median_depth(depth)
        err = abs(med - Z) / Z
        assert err < 0.15, f"cam {name}: median depth {med:.3f} m, expected {Z} m (err {err:.1%})"


# ── Test 3 — wide-baseline geometry requires numDisparities > 128 (Bug 2) ───

def test_wide_baseline_requires_large_num_disparities():
    """True disparity d=180 px must be recovered within tolerance.

    With numDisparities=128 (the old value) SGBM cannot see d=180 and latches
    onto noise, producing depths of tens of metres.  With numDisparities=256
    the disparity window covers d=180 and the result is metric.

    Geometry: fx=1200, b=0.3 m, Z=2 m → d = 1200·0.3/2 = 180 px.
    (The real live setup is Z≈1 m → d≈360, motivating even larger values;
    this test pins the threshold at 128→256 to guard against regression.)
    """
    W, H = 1280, 720
    fx = 1200.0; b = 0.3; Z = 2.0
    d = int(fx * b / Z)          # 180 px

    calib_a = _make_calib(fx, fx, W / 2, H / 2, W, H, [0.0, 0.0, 0.0])
    calib_b = _make_calib(fx, fx, W / 2, H / 2, W, H, [b,   0.0, 0.0])
    computer = StereoDepthComputer(calib_a, calib_b)

    frame_a, frame_b = _shifted_pair(W, H, d)
    depth_a, depth_b = computer.compute_pair(frame_a, frame_b)

    for name, depth in (("A", depth_a), ("B", depth_b)):
        assert (depth > 0).mean() > 0.5, f"cam {name}: mostly empty depth at d={d} px"
        med = _median_depth(depth)
        err = abs(med - Z) / Z
        assert err < 0.15, f"cam {name}: median depth {med:.3f} m, expected {Z} m (err {err:.1%})"


# ── Test 3b — real live geometry: Z=1 m, d≈360 px, requires numDisparities≥384

def test_real_geometry_z1m():
    """The actual live setup: fx=1200, b=0.3 m, Z=1 m → d=360 px.

    numDisparities=256 cannot see d=360 and would fail this test.
    numDisparities=384 (current value) covers it with margin.

    W=1280 is required: (1280-384)/1280 = 70% valid > 50% threshold.
    With W=640: (640-384)/640 = 40% — fails the coverage check even when
    disparity is correctly found.
    """
    W, H = 1280, 720
    fx = 1200.0; b = 0.3; Z = 1.0
    d = int(fx * b / Z)          # 360 px

    calib_a = _make_calib(fx, fx, W / 2, H / 2, W, H, [0.0, 0.0, 0.0])
    calib_b = _make_calib(fx, fx, W / 2, H / 2, W, H, [b,   0.0, 0.0])
    computer = StereoDepthComputer(calib_a, calib_b)

    frame_a, frame_b = _shifted_pair(W, H, d)
    depth_a, depth_b = computer.compute_pair(frame_a, frame_b)

    for name, depth in (("A", depth_a), ("B", depth_b)):
        assert (depth > 0).mean() > 0.5, f"cam {name}: mostly empty at d={d} px (numDisparities too small?)"
        med = _median_depth(depth)
        err = abs(med - Z) / Z
        assert err < 0.15, f"cam {name}: median {med:.3f} m, expected {Z} m (err {err:.1%})"


# ── Test 4 — out_size intrinsics scaling preserves metric accuracy ───────────

def test_out_size_intrinsics_scaling():
    """When out_size differs from calib resolution, intrinsics are scaled so
    stereoRectify / remap still operate correctly in output-resolution space.

    Calib: 640×480, out_size: 960×720 (1.5×).  Frames fed at 960×720 with
    the proportionally larger pixel shift (75 = 50 × 1.5).
    """
    W, H = 640, 480
    out_W, out_H = 960, 720
    fx = 1000.0; b = 0.1; Z = 2.0
    # Scaled focal length at output resolution: fx_out = fx * (out_W/W) = 1500
    # d_out = fx_out * b / Z = 1500 * 0.1 / 2 = 75
    sx = out_W / W
    d_out = int(fx * sx * b / Z)  # 75 px

    calib_a = _make_calib(fx, fx, W / 2, H / 2, W, H, [0.0, 0.0, 0.0])
    calib_b = _make_calib(fx, fx, W / 2, H / 2, W, H, [b,   0.0, 0.0])
    computer = StereoDepthComputer(calib_a, calib_b, out_size=(out_W, out_H))

    assert computer._size == (out_W, out_H), "out_size not stored"

    frame_a, frame_b = _shifted_pair(out_W, out_H, d_out)
    depth_a, depth_b = computer.compute_pair(frame_a, frame_b)

    for name, depth in (("A", depth_a), ("B", depth_b)):
        med = _median_depth(depth)
        err = abs(med - Z) / Z
        assert err < 0.15, f"cam {name}: median {med:.3f} m, expected {Z} m (err {err:.1%})"
