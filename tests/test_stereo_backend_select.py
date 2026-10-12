"""Backend selection: ffs when it can run, otherwise a clean sgbm fallback --
the bridge must never crash on startup just because torch, a GPU or the
checkpoint is missing. None of these need torch or a GPU to run."""
from bridge import stereo_depth_ffs
from bridge.session import BridgeSession


def _resolve(backend, checkpoint=None):
    s = BridgeSession("http://unused", "rtsp://unused", ffs_checkpoint=checkpoint)
    return s._resolve_stereo_backend(backend)


def test_explicit_sgbm_is_kept():
    assert _resolve("sgbm") == "sgbm"


def test_missing_checkpoint_falls_back_to_sgbm():
    assert _resolve("ffs", "no/such/checkpoint.pth") == "sgbm"
    assert _resolve("auto", "no/such/checkpoint.pth") == "sgbm"


def test_auto_without_any_checkpoint_falls_back_to_sgbm(monkeypatch):
    monkeypatch.setattr(stereo_depth_ffs, "find_default_checkpoint", lambda: None)
    assert _resolve("auto") == "sgbm"


def test_failed_warmup_keeps_sgbm_for_the_session(monkeypatch):
    monkeypatch.setattr(stereo_depth_ffs, "unavailable_reason", lambda *_: None)
    s = BridgeSession("http://unused", "rtsp://unused", ffs_checkpoint="x.pth")
    assert s._resolve_stereo_backend("ffs") == "ffs"
    s._ffs_broken = True
    assert s._resolve_stereo_backend("ffs") == "sgbm"


def test_find_default_checkpoint_prefers_upstream_filename(monkeypatch, tmp_path):
    weights = tmp_path / "weights" / "23-36-37"
    weights.mkdir(parents=True)
    (weights / "other.pth").write_bytes(b"")
    (weights / "model_best_bp2_serialize.pth").write_bytes(b"")
    monkeypatch.setattr(stereo_depth_ffs, "_app_dir", lambda: tmp_path)
    assert stereo_depth_ffs.find_default_checkpoint() == weights / "model_best_bp2_serialize.pth"
