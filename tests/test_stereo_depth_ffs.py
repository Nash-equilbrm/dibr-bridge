"""Guard-rail tests for FastFoundationStereoDepthComputer -- checks the
clear-error paths (missing sibling repo, missing checkpoint) that matter for
safe rollback/experimentation, without requiring an actual GPU or downloaded
checkpoint. Gated on torch being installed (requirements-ffs.txt) so the
default `pytest` run (SGBM-only install, no torch) skips this file cleanly --
see README.md's "Alternative stereo backend" section.
"""
import pytest

torch = pytest.importorskip("torch")

from bridge.stereo_depth_ffs import FastFoundationStereoDepthComputer
from bridge.registration_client import CameraCalibration, Intrinsics, Extrinsics


def _make_calib(tvec):
    return CameraCalibration(
        camera_name="test",
        intrinsics=Intrinsics(
            fx=1000.0, fy=1000.0, cx=640.0, cy=360.0,
            dist_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
            image_width=1280, image_height=720,
            reprojection_error=0.0,
        ),
        extrinsics=Extrinsics(rvec=[0.0, 0.0, 0.0], tvec=tvec),
    )


def test_missing_ffs_repo_raises_clear_error(monkeypatch):
    monkeypatch.setenv("FFS_REPO_PATH", "D:/definitely/not/a/real/path")
    calib_a = _make_calib([0.0, 0.0, 0.0])
    calib_b = _make_calib([0.1, 0.0, 0.0])
    with pytest.raises(RuntimeError, match="Fast-FoundationStereo backend selected but its repo isn't at"):
        FastFoundationStereoDepthComputer(calib_a, calib_b, checkpoint_path="unused.pth")


def test_missing_checkpoint_raises_clear_error(monkeypatch):
    monkeypatch.delenv("DIBR_FFS_CHECKPOINT", raising=False)
    calib_a = _make_calib([0.0, 0.0, 0.0])
    calib_b = _make_calib([0.1, 0.0, 0.0])
    with pytest.raises(RuntimeError, match="no checkpoint given"):
        FastFoundationStereoDepthComputer(calib_a, calib_b, checkpoint_path=None)


def test_nonexistent_checkpoint_path_raises_clear_error():
    calib_a = _make_calib([0.0, 0.0, 0.0])
    calib_b = _make_calib([0.1, 0.0, 0.0])
    with pytest.raises(RuntimeError, match="checkpoint not found"):
        FastFoundationStereoDepthComputer(calib_a, calib_b, checkpoint_path="no/such/checkpoint.pth")
