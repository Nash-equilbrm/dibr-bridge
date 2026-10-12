"""Session-side config handling: inference size vs the fixed output size,
and keeping the previous settings when a new config fails to build."""
import asyncio

import numpy as np

from bridge.remote_config import DEFAULTS, StereoConfig
from bridge.session import BridgeSession, _DepthPipeline


class _FakeComputer:
    def __init__(self):
        self.seen = None

    def compute_pair(self, a, b):
        self.seen = a.shape
        h, w = a.shape[:2]
        depth = np.zeros((h, w), np.float32)
        depth[:, : w // 2] = 2.0  # left half valid, right half invalid (0)
        return depth, depth


def test_pipeline_infers_at_infer_size_and_returns_output_size():
    comp = _FakeComputer()
    pipe = _DepthPipeline(comp, (640, 360), (1280, 720))
    frame = np.zeros((720, 1280, 3), np.uint8)
    depth_a, _ = pipe.compute(frame, frame)
    assert comp.seen == (360, 640, 3)
    assert depth_a.shape == (720, 1280)
    # Nearest upsampling: only the original values, no blended edge depths.
    assert set(np.unique(depth_a)) == {0.0, 2.0}


def test_pipeline_at_output_size_is_passthrough():
    comp = _FakeComputer()
    frame = np.zeros((720, 1280, 3), np.uint8)
    depth_a, _ = _DepthPipeline(comp, (1280, 720), (1280, 720)).compute(frame, frame)
    assert comp.seen == (720, 1280, 3) and depth_a.shape == (720, 1280)


class _FakeProvider:
    def __init__(self, cfg):
        self.cfg = cfg

    def load(self):
        return self.cfg


class _FakeHub:
    def latest_frame(self, name):
        return np.zeros((720, 1280, 3), np.uint8)


class _FakeReg:
    def fetch_calibration_pair(self, a, b):
        return object(), object()


def _session(cfg, monkeypatch, fail_when):
    s = BridgeSession("http://unused", "rtsp://unused", config_provider=_FakeProvider(cfg))
    s._hub, s._reg = _FakeHub(), _FakeReg()
    s._cameras = {"cam1": object(), "cam2": object()}
    built = []

    def fake_build(cfg, *_):
        built.append(cfg)
        if fail_when(cfg):
            raise RuntimeError("CUDA out of memory")
        depth = np.full((720, 1280), 2.0, np.float32)
        return _DepthPipeline(_FakeComputer(), (1280, 720), (1280, 720)), depth, depth

    monkeypatch.setattr(s, "_build_depth", fake_build)
    monkeypatch.setattr(s, "_pump_depth", lambda *a: asyncio.sleep(0))
    return s, built


def test_failed_config_keeps_previous_settings(monkeypatch):
    big = StereoConfig(infer_width=1920, infer_height=1080)
    s, built = _session(big, monkeypatch, fail_when=lambda c: c == big)
    asyncio.run(s.set_active_pair("cam1", "cam2"))
    assert built == [big, DEFAULTS]  # tried the new config, then fell back
    assert s._stereo == DEFAULTS


def test_good_config_is_applied(monkeypatch):
    cfg = StereoConfig(max_disp=128)
    s, built = _session(cfg, monkeypatch, fail_when=lambda c: False)
    asyncio.run(s.set_active_pair("cam1", "cam2"))
    assert built == [cfg] and s._stereo == cfg
