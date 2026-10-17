"""_ffs_repo_path() resolution, incl. the frozen (PyInstaller) ffs_src branch.
Needs cv2/numpy only (stereo_depth_ffs imports torch lazily), so it runs in
the default SGBM-only install too.
"""
import sys

from bridge import stereo_depth_ffs as ffs


def _frozen(monkeypatch, meipass, exe):
    monkeypatch.delenv("FFS_REPO_PATH", raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(meipass), raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))


def test_not_frozen_uses_sibling_default(monkeypatch):
    monkeypatch.delenv("FFS_REPO_PATH", raising=False)
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert ffs._ffs_repo_path() == ffs._DEFAULT_FFS_REPO_PATH


def test_frozen_prefers_meipass_ffs_src(monkeypatch, tmp_path):
    (tmp_path / "_internal" / "ffs_src").mkdir(parents=True)
    _frozen(monkeypatch, tmp_path / "_internal", tmp_path / "dibr-bridge.exe")
    assert ffs._ffs_repo_path() == tmp_path / "_internal" / "ffs_src"


def test_frozen_falls_back_to_exe_dir_layouts(monkeypatch, tmp_path):
    (tmp_path / "ffs_src").mkdir()
    _frozen(monkeypatch, tmp_path / "missing_meipass", tmp_path / "dibr-bridge.exe")
    assert ffs._ffs_repo_path() == tmp_path / "ffs_src"


def test_frozen_without_bundle_falls_back_to_sibling(monkeypatch, tmp_path):
    _frozen(monkeypatch, tmp_path / "_internal", tmp_path / "dibr-bridge.exe")
    assert ffs._ffs_repo_path() == ffs._DEFAULT_FFS_REPO_PATH


def test_env_override_wins_when_frozen(monkeypatch, tmp_path):
    (tmp_path / "_internal" / "ffs_src").mkdir(parents=True)
    _frozen(monkeypatch, tmp_path / "_internal", tmp_path / "dibr-bridge.exe")
    monkeypatch.setenv("FFS_REPO_PATH", str(tmp_path / "custom"))
    assert ffs._ffs_repo_path() == tmp_path / "custom"
