"""Top-level bridge state: per-camera color passthrough (pairing-independent,
starts on add_camera) and pairwise live stereo depth (starts fresh on every
set_active_pair — see stereo_depth.py's module docstring for why depth can't
be cached per-camera the way color/URLs are). Matches the control channel
contract documented in the plan file's Phase A section and
DibrBridgeControlChannel.cs on the Unity side.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging

import cv2
import numpy as np

from .livekit_source import LiveKitCameraHub
from .registration_client import RegistrationClient
from .rtsp_publisher import RtspPublisher, encode_depth_yuv420p
from .remote_config import DEFAULTS, ConfigProvider, StereoConfig
from .stereo_depth import BaseStereoDepthComputer, StereoDepthComputer

log = logging.getLogger("dibr_bridge.session")

COLOR_FPS = 30
DEPTH_FPS = 15  # stereo matching is the expensive step — lower rate than color
DEPTH_RANGE_MARGIN = 0.10  # +/-10% safety margin around the pair-start depth range, see session.py notes
PLACEHOLDER_FPS = 5  # nothing ever renders this content, just needs to stay decodable

# OpenDIBR has a hard startup requirement (confirmed by opendibr-c3,
# 2026-09-20, from actually launching it): it needs at least one real,
# already-decodable camera entry in its startup JSON besides the viewport —
# BInitGL() unconditionally demuxes+decodes every startup-JSON camera before
# the app finishes booting, so a dead/unreachable URL there means the whole
# process fails to launch. This bridge runs one small always-on dummy stream
# for exactly that purpose, at a fixed well-known path both this file and
# Unity's startup-JSON generator (OpenDibrSessionManager.cs) agree on by
# convention — never selected by any real add_camera/set_active_pair, exists
# purely to satisfy OpenDIBR's own launch-time check.
PLACEHOLDER_NAME = "dibr_placeholder"


@dataclasses.dataclass
class _CameraState:
    color_url: str
    depth_url: str
    color_publisher: RtspPublisher
    depth_publisher: RtspPublisher
    color_pump_task: asyncio.Task


class BridgeSession:
    def __init__(self, server_url: str, rtsp_base: str, width: int = 1280, height: int = 720, room_code: str = "",
                 config_provider: ConfigProvider | None = None, ffs_checkpoint: str | None = None,
                 ffs_device: str = "cuda"):
        self._reg = RegistrationClient(server_url)
        self._rtsp_base = rtsp_base.rstrip("/")
        self._width = width
        self._height = height
        self._room_code = room_code
        # Stereo tuning (backend, max_disp, valid_iters, inference size, fp16)
        # comes from the config provider -- re-read at startup and on every
        # set_active_pair, see remote_config.py. Paths/devices never do.
        self._config = config_provider or ConfigProvider()
        self._stereo: StereoConfig | None = None  # last successfully applied
        self._ffs_checkpoint = ffs_checkpoint
        self._ffs_device = ffs_device
        self._ffs_broken = False  # set if warm-up failed: sgbm for the rest of the session
        self._ffs_warmup: asyncio.Future | None = None
        self._hub: LiveKitCameraHub | None = None
        self._cameras: dict[str, _CameraState] = {}
        self._active_pair: tuple[str, str] | None = None
        self._depth_task: asyncio.Task | None = None
        self._placeholder_task: asyncio.Task | None = None

    async def start(self) -> None:
        cfg = await asyncio.to_thread(self._config.load)
        if self._resolve_stereo_backend(cfg.backend) == "ffs":
            # Off the event loop: model load + first CUDA inference takes
            # seconds, and must not stall LiveKit/placeholder pumping.
            from . import stereo_depth_ffs
            self._ffs_warmup = asyncio.ensure_future(asyncio.to_thread(
                stereo_depth_ffs.warm_up, self._ffs_checkpoint, self._ffs_device, self._infer_size(cfg),
                cfg.valid_iters, cfg.max_disp, cfg.fp16))

        token = self._reg.fetch_viewer_token(room_code=self._room_code or None)
        self._hub = LiveKitCameraHub(token.livekit_url, token.token)
        await self._hub.connect()
        log.info("Connected to LiveKit at %s", token.livekit_url)

        self._start_placeholder_stream()

    def _infer_size(self, cfg: StereoConfig) -> tuple[int, int]:
        if cfg.infer_width and cfg.infer_height:
            return cfg.infer_width, cfg.infer_height
        return self._width, self._height

    def _resolve_stereo_backend(self, requested: str) -> str:
        """Concrete backend ("sgbm"/"ffs") for a requested one ("auto"/"sgbm"/
        "ffs"): ffs only if it can actually run here, else a loud sgbm
        fallback. Logs only when the outcome changes."""
        if requested == "sgbm":
            return self._log_backend("sgbm", "Stereo backend: sgbm (requested)")
        if requested not in ("auto", "ffs"):
            raise ValueError(f"Unknown stereo backend: {requested!r} (expected 'auto', 'sgbm' or 'ffs')")

        from . import stereo_depth_ffs
        if not self._ffs_checkpoint:
            found = stereo_depth_ffs.find_default_checkpoint()
            self._ffs_checkpoint = str(found) if found else None
        if self._ffs_broken:
            reason = "Fast-FoundationStereo warm-up failed earlier this session"
        else:
            reason = stereo_depth_ffs.unavailable_reason(self._ffs_checkpoint, self._ffs_device)
        if reason is None:
            return self._log_backend("ffs", f"Stereo backend: ffs ({requested}), checkpoint "
                                            f"{self._ffs_checkpoint} on {self._ffs_device}")
        bar = "=" * 72
        return self._log_backend("sgbm", f"{bar}\nStereo backend FALLBACK to sgbm (requested {requested}): "
                                         f"{reason}\n{bar}", level=logging.WARNING)

    def _log_backend(self, backend: str, message: str, level: int = logging.INFO) -> str:
        if message != getattr(self, "_last_backend_message", None):
            log.log(level, "%s", message)
            self._last_backend_message = message
        return backend

    async def _ensure_ffs_ready(self) -> None:
        """Wait for the startup warm-up; if it failed, fall back to sgbm for
        the rest of the session instead of failing every set_active_pair."""
        if self._ffs_warmup is None:
            return
        try:
            await self._ffs_warmup
        except Exception:
            log.exception("=" * 72 + "\nFast-FoundationStereo warm-up failed -- FALLBACK to sgbm")
            self._ffs_broken = True
        self._ffs_warmup = None

    @property
    def placeholder_urls(self) -> tuple[str, str]:
        return f"{self._rtsp_base}/{PLACEHOLDER_NAME}_color", f"{self._rtsp_base}/{PLACEHOLDER_NAME}_depth"

    def _start_placeholder_stream(self) -> None:
        color_url, depth_url = self.placeholder_urls
        color_pub = RtspPublisher(color_url, self._width, self._height, fps=PLACEHOLDER_FPS, pix_fmt_in="bgr24")
        depth_pub = RtspPublisher(depth_url, self._width, self._height, fps=PLACEHOLDER_FPS, pix_fmt_in="yuv420p")
        color_pub.start()
        depth_pub.start()

        gray_frame = np.full((self._height, self._width, 3), 96, dtype=np.uint8)
        # Mid-range constant depth — decoded value is never read by anything
        # (this camera is never part of an active pair), just needs to be a
        # valid encodable frame.
        mid_depth = np.full((self._height, self._width), 5.0, dtype=np.float32)
        depth_frame = encode_depth_yuv420p(mid_depth, 0.1, 30.0)

        async def _pump():
            try:
                while True:
                    color_pub.push(gray_frame)
                    depth_pub.push(depth_frame)
                    await asyncio.sleep(1.0 / PLACEHOLDER_FPS)
            except asyncio.CancelledError:
                pass

        self._placeholder_task = asyncio.ensure_future(_pump())
        log.info("Placeholder stream (%dx%d) live at %s / %s — required for OpenDIBR's startup JSON",
                  self._width, self._height, color_url, depth_url)

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w == self._width and h == self._height:
            return frame
        return cv2.resize(frame, (self._width, self._height))

    # ── Control channel command handlers ────────────────────────────────

    async def add_camera(self, name: str) -> tuple[str, str]:
        if name in self._cameras:
            st = self._cameras[name]
            return st.color_url, st.depth_url

        await self._hub.subscribe(name)
        frame = self._hub.latest_frame(name)
        native_h, native_w = frame.shape[:2]

        color_url = f"{self._rtsp_base}/{name}_color"
        depth_url = f"{self._rtsp_base}/{name}_depth"

        # Always publish at the configured output resolution so OpenDIBR
        # receives streams that match its declared add_camera dimensions and
        # StereoDepthComputer receives frames that match calibration resolution.
        color_pub = RtspPublisher(color_url, self._width, self._height, fps=COLOR_FPS, pix_fmt_in="bgr24")
        depth_pub = RtspPublisher(depth_url, self._width, self._height, fps=DEPTH_FPS, pix_fmt_in="yuv420p")
        color_pub.start()
        depth_pub.start()

        pump_task = asyncio.ensure_future(self._pump_color(name, color_pub))
        self._cameras[name] = _CameraState(color_url, depth_url, color_pub, depth_pub, pump_task)
        log.info("Added camera '%s' (%dx%d → %dx%d) -> %s / %s",
                 name, native_w, native_h, self._width, self._height, color_url, depth_url)
        return color_url, depth_url

    async def remove_camera(self, name: str) -> None:
        st = self._cameras.pop(name, None)
        if st is None:
            return
        st.color_pump_task.cancel()
        st.color_publisher.stop()
        st.depth_publisher.stop()
        self._hub.unsubscribe(name)
        if self._active_pair and name in self._active_pair:
            if self._depth_task:
                self._depth_task.cancel()
            self._active_pair = None
        log.info("Removed camera '%s'", name)

    def _build_depth(self, cfg: StereoConfig, calib_a, calib_b,
                     frame_a: np.ndarray, frame_b: np.ndarray) -> tuple["_DepthPipeline", np.ndarray, np.ndarray]:
        """Blocking (runs in a worker thread): build the computer for `cfg`
        and run its first compute_pair, so a bad setting (e.g. max_disp above
        the checkpoint's range, or GPU OOM at a bigger inference size) fails
        here, before anything is swapped in."""
        backend = self._resolve_stereo_backend(cfg.backend)
        infer_size = self._infer_size(cfg)
        if backend == "sgbm":
            computer = StereoDepthComputer(calib_a, calib_b, out_size=infer_size)
        else:
            # Lazy import: torch/Fast-FoundationStereo are only ever required
            # when ffs is actually in use, so an sgbm-only install never needs
            # them. The model itself is cached process-wide after warm-up.
            from .stereo_depth_ffs import FastFoundationStereoDepthComputer
            computer = FastFoundationStereoDepthComputer(
                calib_a, calib_b, out_size=infer_size,
                checkpoint_path=self._ffs_checkpoint, device=self._ffs_device,
                valid_iters=cfg.valid_iters, max_disp=cfg.max_disp, mixed_precision=cfg.fp16,
            )
        pipeline = _DepthPipeline(computer, infer_size, (self._width, self._height))
        depth_a, depth_b = pipeline.compute(frame_a, frame_b)
        return pipeline, depth_a, depth_b

    async def set_active_pair(self, cam_a: str, cam_b: str) -> tuple[float, float, float, float]:
        if cam_a not in self._cameras or cam_b not in self._cameras:
            raise ValueError(f"Both cameras must be added before set_active_pair (have: {list(self._cameras)})")

        if self._depth_task:
            self._depth_task.cancel()

        calib = self._reg.fetch_calibration_pair(cam_a, cam_b)
        if calib is None:
            raise ValueError(f"No calibration for pair ({cam_a}, {cam_b})")
        calib_a, calib_b = calib

        # Re-read the stereo config on every switch (bounded by the fetch
        # timeout, never raises) so tuning changes apply without a restart.
        cfg = await asyncio.to_thread(self._config.load)
        await self._ensure_ffs_ready()

        frame_a = self._hub.latest_frame(cam_a)
        frame_b = self._hub.latest_frame(cam_b)
        # Worker thread: rectification-map setup and the first compute_pair
        # are slow enough (FFS especially) to stall color pumping otherwise.
        try:
            pipeline, depth_a, depth_b = await asyncio.to_thread(
                self._build_depth, cfg, calib_a, calib_b, frame_a, frame_b)
        except Exception:
            # Keep the last settings that worked (built-in defaults before
            # the first successful switch) rather than failing the switch.
            fallback = self._stereo or DEFAULTS
            if fallback == cfg:
                raise
            log.exception("Stereo config (%s) failed for pair (%s, %s) -- using previous settings (%s)",
                          cfg.describe(), cam_a, cam_b, fallback.describe())
            cfg = fallback
            pipeline, depth_a, depth_b = await asyncio.to_thread(
                self._build_depth, cfg, calib_a, calib_b, frame_a, frame_b)
        self._stereo = cfg

        min_a, max_a = self._range_with_margin(depth_a)
        min_b, max_b = self._range_with_margin(depth_b)

        self._active_pair = (cam_a, cam_b)
        self._depth_task = asyncio.ensure_future(
            self._pump_depth(cam_a, cam_b, pipeline, min_a, max_a, min_b, max_b)
        )
        log.info("Active pair set to (%s, %s); depth range A=[%.2f,%.2f] B=[%.2f,%.2f]",
                  cam_a, cam_b, min_a, max_a, min_b, max_b)
        return min_a, max_a, min_b, max_b

    # ── Background pumps ─────────────────────────────────────────────────

    async def _pump_color(self, name: str, publisher: RtspPublisher) -> None:
        try:
            while True:
                frame = self._hub.latest_frame(name)
                if frame is not None:
                    publisher.push(self._resize(frame))
                await asyncio.sleep(1.0 / COLOR_FPS)
        except asyncio.CancelledError:
            pass

    async def _pump_depth(self, cam_a: str, cam_b: str, pipeline: "_DepthPipeline",
                           min_a: float, max_a: float, min_b: float, max_b: float) -> None:
        st_a = self._cameras[cam_a]
        st_b = self._cameras[cam_b]
        try:
            while True:
                frame_a = self._hub.latest_frame(cam_a)
                frame_b = self._hub.latest_frame(cam_b)
                if frame_a is not None and frame_b is not None:
                    depth_a, depth_b = await asyncio.to_thread(pipeline.compute, frame_a, frame_b)
                    st_a.depth_publisher.push(encode_depth_yuv420p(depth_a, min_a, max_a))
                    st_b.depth_publisher.push(encode_depth_yuv420p(depth_b, min_b, max_b))
                await asyncio.sleep(1.0 / DEPTH_FPS)
        except asyncio.CancelledError:
            pass

    @staticmethod
    def _range_with_margin(depth: np.ndarray) -> tuple[float, float]:
        valid = depth[depth > 0]
        if valid.size == 0:
            return 0.1, 10.0  # fallback — no valid disparity this frame
        d_min, d_max = float(valid.min()), float(valid.max())
        span = max(d_max - d_min, 0.01)
        margin = span * DEPTH_RANGE_MARGIN
        return max(d_min - margin, 0.01), d_max + margin


class _DepthPipeline:
    """A depth computer running at `infer_size`, wrapped so its output always
    comes back at the bridge's output size -- the resolution contract with
    OpenDIBR doesn't change when the config lowers the inference size."""

    def __init__(self, computer: BaseStereoDepthComputer, infer_size: tuple[int, int], out_size: tuple[int, int]):
        self._computer = computer
        self._infer_size = infer_size
        self._out_size = out_size

    def compute(self, frame_a: np.ndarray, frame_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        depth_a, depth_b = self._computer.compute_pair(self._fit(frame_a), self._fit(frame_b))
        return self._upsample(depth_a), self._upsample(depth_b)

    def _fit(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if (w, h) == self._infer_size:
            return frame
        shrinking = w > self._infer_size[0]
        return cv2.resize(frame, self._infer_size, interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)

    def _upsample(self, depth: np.ndarray) -> np.ndarray:
        if self._infer_size == self._out_size:
            return depth
        # Nearest, not bilinear: depth is metric Z (not disparity, so no value
        # rescaling) and 0 marks invalid pixels; interpolating would invent
        # depths across object edges and smear holes into valid neighbours.
        return cv2.resize(depth, self._out_size, interpolation=cv2.INTER_NEAREST)
