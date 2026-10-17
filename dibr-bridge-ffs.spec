# -*- mode: python ; coding: utf-8 -*-

# FFS build: bundles torch (cu12x) + Fast-FoundationStereo source. Multi-GB
# onedir -- Windows viewers with an NVIDIA GPU only. The slim SGBM build stays
# dibr-bridge.spec. Built by Tools/BuildStreamingAssets.ps1 -Ffs.
import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

livekit_datas, livekit_binaries, livekit_hiddenimports = collect_all('livekit')

# Spec dir = dibr-bridge repo root; FFS repo is its sibling (override: FFS_REPO_PATH).
_ffs_repo = os.path.abspath(os.environ.get('FFS_REPO_PATH') or os.path.join(SPECPATH, '..', 'Fast-FoundationStereo'))
if not os.path.isdir(os.path.join(_ffs_repo, 'core')):
    raise SystemExit(f'Fast-FoundationStereo repo not found at {_ffs_repo} (set FFS_REPO_PATH)')
sys.path.insert(0, _ffs_repo)

extra_datas, extra_binaries, extra_hidden = [], [], []
for pkg in ('torch', 'torchvision', 'timm'):
    d, b, h = collect_all(pkg)
    extra_datas += d
    extra_binaries += b
    extra_hidden += h

# FFS source: frozen as modules (the checkpoint is a pickled model that imports
# core.*) AND shipped as data under ffs_src/ so stereo_depth_ffs._ffs_repo_path()
# finds a directory when frozen.
ffs_hidden = collect_submodules('core') + ['Utils'] + [
    'omegaconf', 'einops', 'scipy', 'skimage', 'yaml', 'imageio',
]
ffs_datas = [
    (os.path.join(_ffs_repo, 'core'), 'ffs_src/core'),
    (os.path.join(_ffs_repo, 'Utils.py'), 'ffs_src'),
]

a = Analysis(
    ['entrypoint.py'],
    pathex=[_ffs_repo],
    binaries=livekit_binaries + extra_binaries,
    datas=livekit_datas + extra_datas + ffs_datas,
    hiddenimports=livekit_hiddenimports + extra_hidden + ffs_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='dibr-bridge',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='dibr-bridge',
)
