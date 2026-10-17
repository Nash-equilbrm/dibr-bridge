# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all

livekit_datas, livekit_binaries, livekit_hiddenimports = collect_all('livekit')

a = Analysis(
    ['entrypoint.py'],
    pathex=[],
    binaries=livekit_binaries,
    datas=livekit_datas,
    hiddenimports=livekit_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # The ffs backend is never frozen (README "Alternative stereo backend").
    # Without these, a build venv that has requirements-ffs.txt installed
    # makes PyInstaller follow stereo_depth_ffs's lazy `import torch` and
    # bundle several GB of torch+CUDA. Missing torch at runtime already
    # falls back to sgbm (stereo_depth_ffs.unavailable_reason).
    excludes=['torch', 'torchvision', 'timm'],
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
    upx=True,
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
    upx=True,
    upx_exclude=[],
    name='dibr-bridge',
)
