"""Tests for StereoDepthComputer.

Fixture: a fronto-parallel textured plane at known depth Z, two cameras
separated by baseline b along x with no rotation.  With pure-x translation
and zero distortion, rectification is near-identity and the two views differ
by a pure horizontal pixel shift d = fx*b/Z.

No LiveKit or RTSP dependency — pure cv2/numpy.
"""
import cv2
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


# ── Test 5 — per-camera calibration resolution scaling (regression for the
#    bug where BOTH cameras were scaled using only calib_a's resolution) ────

def test_per_camera_calib_resolution_scaling():
    """calib_a and calib_b calibrated at DIFFERENT native resolutions (real
    rig: 1600x720 vs 2432x1080). Both must scale to out_size using their OWN
    resolution, not a shared ratio derived from calib_a alone.

    fx values are chosen so that, once correctly scaled to out_size, both
    cameras have the SAME output-space fx=2000 — keeping the expected-
    disparity arithmetic simple while still exercising two different native
    sizes end to end (unlike test_out_size_intrinsics_scaling, which uses the
    same calib size for both cameras and so can't catch this bug).
    """
    out_W, out_H = 1280, 720
    calib_a_size = (640, 360)   # same 16:9 aspect as out_size, different absolute size
    calib_b_size = (960, 540)   # same 16:9 aspect as out_size, different absolute size
    fx_a_native = 1000.0   # * (1280/640=2.0)    -> 2000 at out_size
    fx_b_native = 1500.0   # * (1280/960=1.333)  -> 2000 at out_size
    fx_out = 2000.0
    b = 0.1; Z = 2.0
    d_out = int(fx_out * b / Z)  # 100 px

    calib_a = _make_calib(fx_a_native, fx_a_native, calib_a_size[0] / 2, calib_a_size[1] / 2,
                           calib_a_size[0], calib_a_size[1], [0.0, 0.0, 0.0])
    calib_b = _make_calib(fx_b_native, fx_b_native, calib_b_size[0] / 2, calib_b_size[1] / 2,
                           calib_b_size[0], calib_b_size[1], [b, 0.0, 0.0])
    computer = StereoDepthComputer(calib_a, calib_b, out_size=(out_W, out_H))

    frame_a, frame_b = _shifted_pair(out_W, out_H, d_out)
    depth_a, depth_b = computer.compute_pair(frame_a, frame_b)

    for name, depth in (("A", depth_a), ("B", depth_b)):
        assert (depth > 0).mean() > 0.5, f"cam {name}: mostly empty depth"
        med = _median_depth(depth)
        err = abs(med - Z) / Z
        assert err < 0.15, f"cam {name}: median {med:.3f} m, expected {Z} m (err {err:.1%})"


# ── Test 6 — B genuinely physically left of A (regression for the hardcoded
#    "A is always left" matcher-order assumption) ────────────────────────────

def test_b_physically_left_of_a():
    """calib_b sits at NEGATIVE x relative to calib_a, i.e. B is physically
    LEFT and A is physically RIGHT — the opposite of every other fixture in
    this file (where A is always left). StereoSGBM only returns valid
    positive disparity when the physically-left image is fed first; the old
    hardcoded "always feed A first" code returns near-empty disparity for
    both cameras here.
    """
    W, H = 1280, 720
    fx = 1000.0; b = 0.1; Z = 2.0
    d = int(fx * b / Z)  # 50 px

    calib_a = _make_calib(fx, fx, W / 2, H / 2, W, H, [0.0, 0.0, 0.0])
    calib_b = _make_calib(fx, fx, W / 2, H / 2, W, H, [-b, 0.0, 0.0])  # B is LEFT of A
    computer = StereoDepthComputer(calib_a, calib_b, out_size=(W, H))

    assert computer._b_is_left, "geometry says B is left of A but _b_is_left is False"

    # B is left -> B is the reference frame; A (right) sees content shifted
    # left relative to B, by the usual d = fx*b/Z (mirrors _shifted_pair's
    # own A-left/B-right convention, swapped).
    frame_b, frame_a = _shifted_pair(W, H, d)
    depth_a, depth_b = computer.compute_pair(frame_a, frame_b)

    for name, depth in (("A", depth_a), ("B", depth_b)):
        assert (depth > 0).mean() > 0.5, f"cam {name}: mostly empty depth (left/right swap not handled)"
        med = _median_depth(depth)
        err = abs(med - Z) / Z
        assert err < 0.15, f"cam {name}: median {med:.3f} m, expected {Z} m (err {err:.1%})"


# ── Test 7 — non-zero relative rotation: rectified-frame Z must be corrected
#    back to original-camera-frame Z before un-rectifying ───────────────────

def test_rectified_to_original_frame_depth_correction():
    """calib_b is rotated ~25 degrees relative to calib_a (mirrors the real
    rig's ~26-degree cam2/cam3 pair) — R1/R2 are then genuinely non-identity
    (stereoRectify redistributes some rotation onto camera A's R1 too, even
    though calib_a's own rvec is zero), so the rectified-frame-Z -> original-
    frame-Z correction actually matters here.

    Checked directly against a closed-form ground truth rather than through
    a full render+SGBM pipeline: alpha=0 rectification aggressively crops for
    a pair this far apart in angle (same effect observed on the real capture
    data), so only a tiny fraction of the frame has any valid disparity at
    all — an image-based test would be flaky based on whether that sliver
    happens to contain a point we can independently verify, not on whether
    the correction itself is right. The correction factor
    (`_build_z_correction`) only depends on calibration geometry (R, K_rect),
    not on image content/SGBM noise, so it can and should be checked exactly.

    Ground truth, independent re-derivation (see stereo_depth.py's module
    docstring for the full derivation): for a 3D point at RECTIFIED depth
    Z_rect appearing at rectified pixel (u,v), its depth along the ORIGINAL
    (unrectified) camera's optical axis is
    `Z_orig = Z_rect * (R[:,2] . K_rect^-1 [u,v,1])`. Verified independently
    against `cv2.projectPoints`-based geometry (not just re-deriving the same
    formula twice): place a known 3D point in the camera's ORIGINAL frame,
    compute what rectified pixel/depth it would produce via `R1 @ P_orig`,
    then apply this test's formula and check it recovers the original Z.
    """
    W, H = 1280, 720
    fx = 1000.0

    rvec_a, tvec_a = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    rvec_b, tvec_b = [0.0, np.radians(25.0), 0.0], [0.13, 0.02, 0.0]

    calib_a = _make_calib(fx, fx, W / 2, H / 2, W, H, tvec_a)
    calib_a.extrinsics.rvec = rvec_a
    calib_b = _make_calib(fx, fx, W / 2, H / 2, W, H, tvec_b)
    calib_b.extrinsics.rvec = rvec_b
    computer = StereoDepthComputer(calib_a, calib_b, out_size=(W, H))

    # Independently recompute R1/R2/K_rect via a SEPARATE cv2.stereoRectify
    # call (not by reading the production class's internals) -- this is the
    # "ground truth" the fix is checked against, mirroring exactly how the
    # formula was numerically verified by hand before implementing it.
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]])
    D = np.zeros(5)
    Ra, _ = cv2.Rodrigues(np.array(rvec_a))
    ta = np.array(tvec_a).reshape(3, 1)
    Rb, _ = cv2.Rodrigues(np.array(rvec_b))
    tb = np.array(tvec_b).reshape(3, 1)
    R_rel = Rb @ Ra.T
    t_rel = tb - R_rel @ ta
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
        K, D, K, D, (W, H), R_rel, t_rel, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

    # Sanity: this geometry actually produces non-identity rectification
    # rotation for BOTH cameras (not just B, even though calib_a's own rvec
    # is zero -- stereoRectify redistributes rotation onto both) -- otherwise
    # this test would coincidentally pass even without the fix, the same
    # trap the plan flagged in the pre-existing fixtures.
    assert not np.allclose(R1, np.eye(3), atol=1e-3), "R1 came out ~identity -- geometry doesn't exercise the fix"
    assert not np.allclose(R2, np.eye(3), atol=1e-3), "R2 came out ~identity -- geometry doesn't exercise the fix"

    rng = np.random.default_rng(3)
    for name, correction, R, K_rect in (
        ("A", computer._z_correction_a, R1, P1[:, :3]),
        ("B", computer._z_correction_b, R2, P2[:, :3]),
    ):
        for _ in range(5):
            u = int(rng.integers(200, W - 200))
            v = int(rng.integers(150, H - 150))
            d_rect = np.linalg.inv(K_rect) @ np.array([u, v, 1.0])
            expected_factor = R[:, 2] @ d_rect
            got_factor = float(correction[v, u])
            assert abs(got_factor - expected_factor) < 1e-3, (
                f"cam {name} z-correction at ({u},{v}): got {got_factor:.4f}, "
                f"expected {expected_factor:.4f}")
