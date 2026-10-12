"""Backend selection: ffs when it can run, otherwise a clean sgbm fallback --
the bridge must never crash on startup just because torch, a GPU or the
checkpoint is missing. None of these need torch or a GPU to run."""
from bridge import stereo_depth_ffs
from bridge.session import BridgeSession


def _resolve(backend, checkpoint=None):
    s = BridgeSession("http://unused", "rtsp://unused", stereo_backend=backend, ffs_checkpoint=checkpoint)
    s._resolve_stereo_backend()
    return s


def test_explicit_sgbm_is_kept():
    assert _resolve("sgbm")._stereo_backend == "sgbm"


def test_missing_checkpoint_falls_back_to_sgbm():
    assert _resolve("ffs", "no/such/checkpoint.pth")._stereo_backend == "sgbm"
    assert _resolve("auto", "no/such/checkpoint.pth")._stereo_backend == "sgbm"


def test_auto_without_any_checkpoint_falls_back_to_sgbm(monkeypatch, tmp_path):
    monkeypatch.setattr(stereo_depth_ffs, "find_default_checkpoint", lambda: None)
    assert _resolve("auto")._stereo_backend == "sgbm"


def test_find_default_checkpoint_prefers_upstream_filename(monkeypatch, tmp_path):
    weights = tmp_path / "weights" / "23-36-37"
    weights.mkdir(parents=True)
    (weights / "other.pth").write_bytes(b"")
    (weights / "model_best_bp2_serialize.pth").write_bytes(b"")
    monkeypatch.setattr(stereo_depth_ffs, "_app_dir", lambda: tmp_path)
    assert stereo_depth_ffs.find_default_checkpoint() == weights / "model_best_bp2_serialize.pth"
