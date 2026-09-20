"""Pushes raw frames into a local RTSP server (mediamtx) via an ffmpeg
subprocess. One instance per published stream (a camera has two: color and
depth). Requires `ffmpeg` on PATH and an RTSP server already listening at
the target URL's host:port — see dibr-bridge/README.md.

UNVERIFIED — see README. The ffmpeg invocation below is a reasonable
first-pass low-latency H.264/RTSP push command, not tuned/tested against
OpenDIBR's actual decode side.
"""
from __future__ import annotations

import subprocess
from typing import Optional

import numpy as np


class RtspPublisher:
    def __init__(self, rtsp_url: str, width: int, height: int, fps: int = 30, pix_fmt_in: str = "bgr24"):
        self.url = rtsp_url
        self._width = width
        self._height = height
        self._fps = fps
        self._pix_fmt_in = pix_fmt_in
        self._proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        if self._proc is not None:
            return
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", self._pix_fmt_in,
            "-s", f"{self._width}x{self._height}", "-r", str(self._fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-f", "rtsp", "-rtsp_transport", "tcp",
            self.url,
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def push(self, frame: np.ndarray) -> None:
        if self._proc is None or self._proc.stdin is None:
            return
        try:
            self._proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, ValueError):
            pass  # ffmpeg died — is_alive() will report it, caller decides whether to restart

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.terminate()
            self._proc.wait(timeout=3)
        except Exception:
            self._proc.kill()
        self._proc = None


def encode_depth_yuv420p(depth_m: np.ndarray, z_near: float, z_far: float) -> np.ndarray:
    """Metric depth (metres, 0 = invalid) -> a flat uint8 YUV420p frame
    buffer (Y = encoded depth, U/V = constant 128 / ignored) ready to push
    straight into RtspPublisher with pix_fmt_in="yuv420p".

    Confirmed against OpenDIBR's actual depth decode shader
    (src/vertex.fs:48-51, per opendibr-c3, 2026-09-20) — it decodes via
    INVERSE depth (disparity-like), not linear min/max:
        metricDepth = 1 / (1/z_far + depth01 * (1/z_near - 1/z_far))
    so encoding must invert exactly that:
        depth01 = (1/metricDepth - 1/z_far) / (1/z_near - 1/z_far)
    depth01=1.0 -> z_near (closest), depth01=0.0 -> z_far (farthest).
    z_near/z_far here are depthMinA/B and depthMaxA/B respectively, as
    already threaded through set_active_pair's contract on both control
    channels — no field renaming needed, just the correct formula.

    8-bit (not the dataset's usual 12-bit) — chosen for RTSP/H.264
    compatibility without depending on an HEVC Main10/12 encoder being
    available; matching bitDepthDepth=8 must be sent in OpenDIBR's
    add_camera call (see OpenDibrSessionManager.cs). Revisit for 10/12-bit
    HEVC if 8-bit banding proves visually insufficient (this is the
    documented tradeoff opendibr-c3 flagged, not an oversight).
    """
    h, w = depth_m.shape
    inv_near = 1.0 / z_near
    inv_far = 1.0 / z_far
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_depth = np.where(depth_m > 0, 1.0 / depth_m, inv_far)  # invalid pixels -> far/background
    depth01 = np.clip((inv_depth - inv_far) / (inv_near - inv_far), 0.0, 1.0)

    y_plane = (depth01 * 255.0).astype(np.uint8)
    uv_h, uv_w = h // 2, w // 2
    u_plane = np.full((uv_h, uv_w), 128, dtype=np.uint8)
    v_plane = np.full((uv_h, uv_w), 128, dtype=np.uint8)

    return np.concatenate([y_plane.ravel(), u_plane.ravel(), v_plane.ravel()])
