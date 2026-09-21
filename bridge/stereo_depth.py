"""Live stereo depth for one camera pair — cv2.stereoRectify +
initUndistortRectifyMap + remap + StereoSGBM, disparity -> metric depth via
depth = fx' * baseline / disparity (rectified focal length and baseline),
per the plan's Phase A spec and action_log_Sep_11th_opendibr.md's stereo TODO.

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

UNVERIFIED — see dibr-bridge/README.md. The undistortPoints-based inverse
mapping below is the textbook-correct way to go from rectified-space data
back to original-image alignment, but this whole module has not been run
against real frames. Visually check depth-vs-color alignment before trusting
it.
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


class StereoDepthComputer:
    def __init__(self, calib_a: CameraCalibration, calib_b: CameraCalibration,
                 out_size: tuple[int, int] | None = None):
        calib_size = (calib_a.intrinsics.image_width, calib_a.intrinsics.image_height)
        self._size = out_size if out_size is not None else calib_size
        self._prepare(calib_a, calib_b, calib_size)

    def _prepare(self, calib_a: CameraCalibration, calib_b: CameraCalibration,
                 calib_size: tuple[int, int]) -> None:
        Ka = _camera_matrix(calib_a.intrinsics)
        Da = np.array(calib_a.intrinsics.dist_coeffs, dtype=np.float64)
        Kb = _camera_matrix(calib_b.intrinsics)
        Db = np.array(calib_b.intrinsics.dist_coeffs, dtype=np.float64)

        # Scale intrinsics from calibration resolution to output resolution so
        # stereoRectify / initUndistortRectifyMap operate in output-resolution
        # space — matching the frames session.py resizes to self._width×self._height.
        # Mirrors Unity's ScaleIntrinsics in OpenDibrSessionManager.cs.
        if self._size != calib_size:
            sx = self._size[0] / calib_size[0]
            sy = self._size[1] / calib_size[1]
            for K in (Ka, Kb):
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

        self._fx_rect = float(P1[0, 0])
        self._baseline_m = float(np.linalg.norm(t_rel))

        # numDisparities=384: covers d = fx*b/Z for the real live geometry
        # (fx≈1200, b≈0.3 m, Z≈1 m → d≈360 px; 256 was still too small).
        # Must be a multiple of 16.
        self._sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=384, blockSize=5,
            P1=8 * 3 * 5 ** 2, P2=32 * 3 * 5 ** 2,
            disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=32,
        )
        log.info("StereoDepthComputer ready: size=%s fx_rect=%.1f baseline=%.3f m numDisparities=%d",
                 self._size, self._fx_rect, self._baseline_m, self._sgbm.getNumDisparities())

    def _build_original_to_rectified_map(self, K, D, R, P) -> tuple[np.ndarray, np.ndarray]:
        w, h = self._size
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        pts = np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(-1, 1, 2)
        rectified_pts = cv2.undistortPoints(pts, K, D, R=R, P=P)
        mapx = rectified_pts[:, 0, 0].reshape(h, w)
        mapy = rectified_pts[:, 0, 1].reshape(h, w)
        return mapx, mapy

    def _disparity_to_depth(self, disparity_fixedpoint: np.ndarray) -> np.ndarray:
        disparity = disparity_fixedpoint.astype(np.float32) / 16.0  # SGBM returns Q4.4 fixed-point
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

        gray_a = cv2.cvtColor(rect_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(rect_b, cv2.COLOR_BGR2GRAY)

        disp_a_raw = self._sgbm.compute(gray_a, gray_b)
        # Right-camera disparity via horizontal-flip trick: StereoSGBM with
        # minDisparity=0 only yields valid positive disparities when the left
        # image is first; feeding (gray_b, gray_a) directly returns all-zero.
        # Flip both images so the right camera becomes the "left" input, run
        # SGBM, then flip the result back to restore original column order.
        disp_b_raw = cv2.flip(self._sgbm.compute(cv2.flip(gray_b, 1), cv2.flip(gray_a, 1)), 1)

        if log.isEnabledFor(logging.DEBUG):
            for name, raw in (("A", disp_a_raw), ("B", disp_b_raw)):
                valid = raw[raw > 0]
                log.debug("disparity %s: valid_px=%d min=%.1f max=%.1f",
                          name, valid.size,
                          float(valid.min()) / 16 if valid.size else 0.0,
                          float(valid.max()) / 16 if valid.size else 0.0)

        depth_a_rect = self._disparity_to_depth(disp_a_raw)
        depth_b_rect = self._disparity_to_depth(disp_b_raw)

        unrect_a_x, unrect_a_y = self._unrect_map_a
        unrect_b_x, unrect_b_y = self._unrect_map_b
        depth_a = cv2.remap(depth_a_rect, unrect_a_x, unrect_a_y, cv2.INTER_LINEAR)
        depth_b = cv2.remap(depth_b_rect, unrect_b_x, unrect_b_y, cv2.INTER_LINEAR)

        return depth_a, depth_b
