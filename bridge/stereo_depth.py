"""Live stereo depth for one camera pair — cv2.stereoRectify +
initUndistortRectifyMap + remap, disparity -> metric depth via
depth = fx' * baseline / disparity (rectified focal length and baseline),
per the plan's Phase A spec and action_log_Sep_11th_opendibr.md's stereo TODO.

The actual disparity computation (StereoSGBM today; optionally
Fast-FoundationStereo, see stereo_depth_ffs.py) is pluggable —
BaseStereoDepthComputer owns everything backend-agnostic (rectification,
calibration handling, the un-rectify/Z-correction step below) and subclasses
only implement `_create_matcher`/`_compute_disparity_pair`. This exists so
StereoDepthComputer's hard-won SGBM correctness fixes (see the three points
below) are never at risk from backend experiments — see the "Alternative
stereo backend" section of README.md.

IMPORTANT design point (worked out while writing this, not in the original
plan text): stereo rectification re-warps each image onto a shared epipolar
plane — the resulting "rectified" depth map is NOT pixel-aligned with the
camera's own raw, unrectified frame (the one already being published,
pairing-independently, as that camera's stable color RTSP stream — see
DibrBridgeControlChannel.cs's doc comment on the Unity side and the
`add_camera` contract in the handoff spec). Rather than switch the color
stream to rectified output too (which would make color pairing-dependent and
break the "stable per-camera URL" contract Unity already builds against —
Item 4's `add_camera` sends static per-camera intrinsics once, not
per-pairing), this module un-rectifies the computed depth back onto each
camera's ORIGINAL pixel grid before returning it, so it lines up with the
already-stable raw color stream and the already-stable calibration
intrinsics Unity already sends to OpenDIBR.

Three correctness issues found (2026-09-21, via offline testing against real
capture data + a peer review) and fixed here:

1. Per-camera intrinsics scaling. calib_a and calib_b can have different
   NATIVE calibration resolutions (confirmed real: two cameras calibrated at
   1600x720 vs 2432x1080). Each camera's K must be scaled by ITS OWN
   calib-resolution -> out_size ratio, not a single shared ratio derived from
   only one of the two cameras.
2. Left/right camera identity. stereoRectify doesn't guarantee calib_a ends
   up physically left of calib_b in the rectified pair — that depends on the
   actual relative pose, not argument order. Verified empirically: P1[0,3] is
   always 0 (camera A is the rectified pair's reference/origin) and
   P2[0,3] > 0 iff calib_b is physically to the right of calib_a; when
   P2[0,3] < 0, calib_b is actually LEFT and the matcher's "always feed A
   first" assumption silently produces zero/garbage disparity for one
   camera (StereoSGBM with minDisparity=0 only returns valid positive
   disparity when the physically-left image is given first).
3. Rectified-frame vs. original-frame depth. `depth = fx_rect*baseline/disp`
   is the Z-coordinate along the RECTIFIED camera's optical axis, not the
   original (un-rectified) camera's optical axis -- they differ by the
   rectification rotation (R1/R2) whenever it isn't the identity, which is
   whenever the two cameras have any real relative rotation (confirmed
   real: ~26 degrees between two of the actual capture cameras). Verified
   numerically with a synthetic 25-degree-rotation test case: the
   uncorrected value was off by ~11% (matches "error scales with
   rectification rotation angle"). Fix: for each RECTIFIED pixel (u,v),
   multiply Z_rect by `R1[:,2] . (K_rect^-1 [u,v,1])` (R1's third COLUMN
   dotted with the pixel's ray direction in rectified-camera coordinates,
   evaluated in the rectified pixel grid, BEFORE the un-rectify remap) to
   get the original-frame Z. Derivation: a rectified-frame 3D point
   P_rect = R1 @ P_orig (stereoRectify's R1 maps original camera axes to
   rectified camera axes), so P_orig = R1^T @ P_rect and
   Z_orig = row2(R1^T) . P_rect = column2(R1) . P_rect. Since
   P_rect = Z_rect * K_rect^-1[u,v,1]^T (third row of K_rect^-1 is [0,0,1],
   so the ray's z-component is always 1), this reduces to the stated
   formula. Same derivation applies to camera B via R2/K_rect_b.

UNVERIFIED beyond the above: the underlying CALIBRATION DATA itself (not
this module's math) may still be wrong -- calibration frames may not be
captured through the same orientation-handling path as the live-published
frames they're meant to describe (tracked separately, Master-Thesis-Client
side). Use `check_stereo_calib.py` to validate calibration quality
(epipolar error against real ChArUco correspondences) independently of this
module's own correctness.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger("dibr_bridge.stereo_depth")

from .registration_client import CameraCalibration


def _camera_matrix(intr) -> np.ndarray:
    return np.array([
        [intr.fx, 0, intr.cx],
        [0, intr.fy, intr.cy],
        [0, 0, 1],
    ], dtype=np.float64)


class BaseStereoDepthComputer:
    """Backend-agnostic rectification/calibration/depth-post-processing
    pipeline for one camera pair. Subclasses provide the actual disparity
    computation via `_create_matcher`/`_compute_disparity_pair` — everything
    else here (intrinsics scaling, stereoRectify, the un-rectify/Z-correction
    math) is shared and must not be duplicated per backend."""

    def __init__(self, calib_a: CameraCalibration, calib_b: CameraCalibration,
                 out_size: tuple[int, int] | None = None):
        calib_size_a = (calib_a.intrinsics.image_width, calib_a.intrinsics.image_height)
        self._size = out_size if out_size is not None else calib_size_a
        self._prepare(calib_a, calib_b)

    def _prepare(self, calib_a: CameraCalibration, calib_b: CameraCalibration) -> None:
        Ka = _camera_matrix(calib_a.intrinsics)
        Da = np.array(calib_a.intrinsics.dist_coeffs, dtype=np.float64)
        Kb = _camera_matrix(calib_b.intrinsics)
        Db = np.array(calib_b.intrinsics.dist_coeffs, dtype=np.float64)

        # Scale each camera's intrinsics from ITS OWN calibration resolution
        # to the output resolution so stereoRectify / initUndistortRectifyMap
        # operate in output-resolution space — matching the frames
        # session.py resizes to self._width×self._height. Mirrors Unity's
        # ScaleIntrinsics in OpenDibrSessionManager.cs. calib_a and calib_b
        # may have been calibrated at DIFFERENT native resolutions, so each
        # gets its own scale factor — using one camera's ratio for both
        # (the earlier bug here) silently corrupts whichever camera wasn't
        # used to derive the ratio.
        calib_size_a = (calib_a.intrinsics.image_width, calib_a.intrinsics.image_height)
        calib_size_b = (calib_b.intrinsics.image_width, calib_b.intrinsics.image_height)
        for K, calib_size in ((Ka, calib_size_a), (Kb, calib_size_b)):
            if self._size != calib_size:
                sx = self._size[0] / calib_size[0]
                sy = self._size[1] / calib_size[1]
                K[0, 0] *= sx; K[0, 2] *= sx  # fx, cx
                K[1, 1] *= sy; K[1, 2] *= sy  # fy, cy

        Ra, _ = cv2.Rodrigues(np.array(calib_a.extrinsics.rvec, dtype=np.float64))
        ta = np.array(calib_a.extrinsics.tvec, dtype=np.float64).reshape(3, 1)
        Rb, _ = cv2.Rodrigues(np.array(calib_b.extrinsics.rvec, dtype=np.float64))
        tb = np.array(calib_b.extrinsics.tvec, dtype=np.float64).reshape(3, 1)

        # Relative pose cam A -> cam B, derived from both cameras' board-frame
        # extrinsics (P_cam = R*P_board + t, see CameraCalibrationData.cs):
        #   P_board = Ra^T (P_camA - ta)
        #   P_camB  = Rb * P_board + tb = (Rb*Ra^T) * P_camA + (tb - Rb*Ra^T*ta)
        R_rel = Rb @ Ra.T
        t_rel = tb - R_rel @ ta

        R1, R2, P1, P2, _Q, _roi1, _roi2 = cv2.stereoRectify(
            Ka, Da, Kb, Db, self._size, R_rel, t_rel,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
        )

        self._rect_map_a = cv2.initUndistortRectifyMap(Ka, Da, R1, P1, self._size, cv2.CV_32FC1)
        self._rect_map_b = cv2.initUndistortRectifyMap(Kb, Db, R2, P2, self._size, cv2.CV_32FC1)

        # Inverse mapping (original grid -> rectified sample location) for
        # un-rectifying the computed depth back onto each camera's own raw
        # pixel grid — see module docstring.
        self._unrect_map_a = self._build_original_to_rectified_map(Ka, Da, R1, P1)
        self._unrect_map_b = self._build_original_to_rectified_map(Kb, Db, R2, P2)

        # Per-rectified-pixel factor converting Z along the rectified
        # camera's optical axis to Z along the ORIGINAL camera's optical
        # axis — see module docstring point 3. Applied before un-rectifying.
        self._z_correction_a = self._build_z_correction(R1, P1[:, :3])
        self._z_correction_b = self._build_z_correction(R2, P2[:, :3])

        # P1[0,3] is always 0 (camera A is the rectified pair's reference).
        # P2[0,3] > 0 iff calib_b ends up physically to the RIGHT of calib_a
        # in the rectified geometry (the assumption compute_pair's matcher
        # order used to hardcode); < 0 means calib_b is actually LEFT and the
        # matcher input order must be swapped — see module docstring point 2.
        # Purely a function of calibration geometry, not of any particular
        # matcher, so every backend gets this for free.
        self._b_is_left = P2[0, 3] < 0

        self._fx_rect = float(P1[0, 0])
        self._baseline_m = float(np.linalg.norm(t_rel))

        self._create_matcher()

    def _create_matcher(self) -> None:
        """Backend hook: set up whatever the concrete disparity engine needs
        (an SGBM matcher, a loaded model, ...). Called once, at the end of
        `_prepare`, after all shared geometry (`_size`, `_b_is_left`,
        `_fx_rect`, `_baseline_m`, rectify/unrectify maps) is available."""
        raise NotImplementedError

    def _compute_disparity_pair(self, rect_a_bgr: np.ndarray, rect_b_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Backend hook: given the two RECTIFIED BGR frames, return
        (disp_a, disp_b) — plain float pixel-disparity maps (no fixed-point
        encoding), each aligned to the rectified pixel grid. Must apply
        `self._b_is_left` itself when the matcher needs to know which input
        is physically left (see StereoDepthComputer's flip-trick)."""
        raise NotImplementedError

    def _build_original_to_rectified_map(self, K, D, R, P) -> tuple[np.ndarray, np.ndarray]:
        w, h = self._size
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        pts = np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(-1, 1, 2)
        rectified_pts = cv2.undistortPoints(pts, K, D, R=R, P=P)
        mapx = rectified_pts[:, 0, 0].reshape(h, w)
        mapy = rectified_pts[:, 0, 1].reshape(h, w)
        return mapx, mapy

    def _build_z_correction(self, R: np.ndarray, K_rect: np.ndarray) -> np.ndarray:
        """Per-RECTIFIED-pixel factor `R[:,2] . K_rect^-1[u,v,1]` — see module
        docstring point 3. Multiply the rectified depth map by this (before
        un-rectifying) to convert Z-along-rectified-axis to
        Z-along-original-camera-axis."""
        w, h = self._size
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        ones = np.ones_like(xs)
        rays = np.stack([xs, ys, ones], axis=-1)  # (h, w, 3), K_rect^-1[u,v,1]^T computed below
        K_rect_inv = np.linalg.inv(K_rect)
        rays = rays @ K_rect_inv.T  # (h, w, 3) ray directions in rectified-camera coords
        factor = rays @ R[:, 2]  # dot each ray with R's third column -> (h, w)
        return factor.astype(np.float32)

    def _disparity_to_depth(self, disparity: np.ndarray) -> np.ndarray:
        """disparity: plain float pixel-disparity (already normalized by the
        backend — e.g. SGBM's Q4.4 fixed-point unpack happens in the
        backend's own `_compute_disparity_pair`, not here)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = np.where(disparity > 0.0, (self._fx_rect * self._baseline_m) / disparity, 0.0)
        return depth

    def compute_pair(self, frame_a_bgr: np.ndarray, frame_b_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (depth_a, depth_b), each a float32 metres array aligned to
        that camera's ORIGINAL (unrectified) pixel grid, matching frame_a_bgr
        / frame_b_bgr's own resolution/orientation."""
        map_a_x, map_a_y = self._rect_map_a
        map_b_x, map_b_y = self._rect_map_b
        rect_a = cv2.remap(frame_a_bgr, map_a_x, map_a_y, cv2.INTER_LINEAR)
        rect_b = cv2.remap(frame_b_bgr, map_b_x, map_b_y, cv2.INTER_LINEAR)

        disp_a, disp_b = self._compute_disparity_pair(rect_a, rect_b)

        depth_a_rect = self._disparity_to_depth(disp_a) * self._z_correction_a
        depth_b_rect = self._disparity_to_depth(disp_b) * self._z_correction_b

        unrect_a_x, unrect_a_y = self._unrect_map_a
        unrect_b_x, unrect_b_y = self._unrect_map_b
        depth_a = cv2.remap(depth_a_rect, unrect_a_x, unrect_a_y, cv2.INTER_LINEAR)
        depth_b = cv2.remap(depth_b_rect, unrect_b_x, unrect_b_y, cv2.INTER_LINEAR)

        return depth_a, depth_b


class StereoDepthComputer(BaseStereoDepthComputer):
    """Default backend: cv2.StereoSGBM. See BaseStereoDepthComputer for the
    shared rectification/calibration/depth pipeline this plugs into."""

    def _create_matcher(self) -> None:
        # numDisparities=384: covers d = fx*b/Z for the real live geometry
        # (fx≈1200, b≈0.3 m, Z≈1 m → d≈360 px; 256 was still too small).
        # Must be a multiple of 16.
        self._sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=384, blockSize=5,
            P1=8 * 3 * 5 ** 2, P2=32 * 3 * 5 ** 2,
            disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=32,
        )
        log.info("StereoDepthComputer ready: size=%s fx_rect=%.1f baseline=%.3f m "
                  "numDisparities=%d b_is_left=%s",
                 self._size, self._fx_rect, self._baseline_m,
                 self._sgbm.getNumDisparities(), self._b_is_left)

    def _compute_disparity_pair(self, rect_a_bgr: np.ndarray, rect_b_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gray_a = cv2.cvtColor(rect_a_bgr, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(rect_b_bgr, cv2.COLOR_BGR2GRAY)

        # StereoSGBM with minDisparity=0 only yields valid positive
        # disparities when the physically-LEFT image is given first. Which
        # camera (A or B) is actually left depends on the real geometry, not
        # argument order — decided once in _prepare via P2[0,3]'s sign (see
        # module docstring point 2). The other camera's disparity comes from
        # the horizontal-flip trick: mirror both images so the physically-
        # right camera becomes the effective "left" input, run SGBM, flip
        # the result back.
        if self._b_is_left:
            disp_b_raw = self._sgbm.compute(gray_b, gray_a)
            disp_a_raw = cv2.flip(self._sgbm.compute(cv2.flip(gray_a, 1), cv2.flip(gray_b, 1)), 1)
        else:
            disp_a_raw = self._sgbm.compute(gray_a, gray_b)
            disp_b_raw = cv2.flip(self._sgbm.compute(cv2.flip(gray_b, 1), cv2.flip(gray_a, 1)), 1)

        if log.isEnabledFor(logging.DEBUG):
            for name, raw in (("A", disp_a_raw), ("B", disp_b_raw)):
                valid = raw[raw > 0]
                log.debug("disparity %s: valid_px=%d min=%.1f max=%.1f",
                          name, valid.size,
                          float(valid.min()) / 16 if valid.size else 0.0,
                          float(valid.max()) / 16 if valid.size else 0.0)

        # SGBM returns Q4.4 fixed-point; normalize to plain float disparity
        # before handing it back to the shared compute_pair() shell.
        disp_a = disp_a_raw.astype(np.float32) / 16.0
        disp_b = disp_b_raw.astype(np.float32) / 16.0
        return disp_a, disp_b
