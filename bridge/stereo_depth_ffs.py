"""Fast-FoundationStereo backend for live stereo depth — an opt-in, GPU-only
alternative to StereoDepthComputer's StereoSGBM matcher, selected via
`--stereo-backend ffs` (see main.py/session.py). Plugs into
BaseStereoDepthComputer (stereo_depth.py) for everything backend-agnostic
(rectification, calibration handling, the un-rectify/Z-correction step) —
this module's only job is turning a rectified BGR image pair into a plain
float disparity map via NVIDIA's Fast-FoundationStereo model
(https://github.com/nvlabs/fast-foundationstereo, forked at
https://github.com/Nash-equilbrm/Fast-FoundationStereo for this project).

Never imported at module load time on the default SGBM path — session.py
only imports this module inside its "ffs" backend-dispatch branch, so a
plain `pip install -r requirements.txt` + default SGBM setup never needs
torch/CUDA installed. See README.md's "Alternative stereo backend
(experimental)" section for setup instructions and rationale.

Layout assumption: the Fast-FoundationStereo repo (your fork) is cloned as a
SIBLING repo next to this one, not a submodule — mirroring how
client-sdk-unity is referenced by the two Unity apps via a relative
`file:../../client-sdk-unity` path (see top-level CLAUDE.md). Default
location: ../Fast-FoundationStereo relative to this repo's root; override
with the FFS_REPO_PATH env var if cloned elsewhere.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from .stereo_depth import BaseStereoDepthComputer

log = logging.getLogger("dibr_bridge.stereo_depth_ffs")

_DIBR_BRIDGE_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_FFS_REPO_PATH = _DIBR_BRIDGE_REPO_ROOT.parent / "Fast-FoundationStereo"


def _ffs_repo_path() -> Path:
    override = os.environ.get("FFS_REPO_PATH")
    return Path(override) if override else _DEFAULT_FFS_REPO_PATH


def _ensure_ffs_importable() -> None:
    """Add the Fast-FoundationStereo sibling repo to sys.path. Called lazily,
    only when the ffs backend is actually constructed — never at module
    import time."""
    repo_path = _ffs_repo_path()
    if not repo_path.is_dir():
        raise RuntimeError(
            f"Fast-FoundationStereo backend selected but its repo isn't at "
            f"{repo_path} -- clone your fork there (git clone "
            f"https://github.com/Nash-equilbrm/Fast-FoundationStereo.git), or "
            f"set FFS_REPO_PATH to wherever you cloned it. See README.md's "
            f"'Alternative stereo backend' section."
        )
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


class FastFoundationStereoDepthComputer(BaseStereoDepthComputer):
    """Opt-in alternative to StereoDepthComputer. Requires requirements-ffs.txt
    installed (torch + CUDA) and the Fast-FoundationStereo sibling repo
    cloned next to this one, plus a manually-downloaded checkpoint (research
    checkpoints are Google-Drive-gated under the NVIDIA Open Model Agreement,
    no stable direct-download URL to automate) -- see README.md.
    """

    def __init__(self, calib_a, calib_b, out_size: tuple[int, int] | None = None,
                 checkpoint_path: str | None = None, device: str = "cuda",
                 valid_iters: int = 4, max_disp: int = 192):
        self._checkpoint_path = checkpoint_path or os.environ.get("DIBR_FFS_CHECKPOINT")
        self._device = device
        self._valid_iters = valid_iters
        self._max_disp = max_disp
        super().__init__(calib_a, calib_b, out_size=out_size)

    def _create_matcher(self) -> None:
        _ensure_ffs_importable()
        try:
            import torch
        except ImportError as e:
            raise RuntimeError(
                "Fast-FoundationStereo backend selected but its dependencies "
                "aren't installed -- see requirements-ffs.txt / README.md's "
                "'Alternative stereo backend' section."
            ) from e
        from core.utils.utils import InputPadder  # Fast-FoundationStereo sibling repo

        if not self._checkpoint_path:
            raise RuntimeError(
                "Fast-FoundationStereo backend selected but no checkpoint given "
                "-- pass --ffs-checkpoint <path to a .pth under weights/<name>/> "
                "or set DIBR_FFS_CHECKPOINT. Research checkpoints are on Google "
                "Drive (gated behind the NVIDIA Open Model Agreement, no direct "
                "download URL to automate) -- see the Fast-FoundationStereo "
                "repo's README for the link."
            )
        checkpoint_path = Path(self._checkpoint_path)
        if not checkpoint_path.is_file():
            raise RuntimeError(f"Fast-FoundationStereo checkpoint not found: {checkpoint_path}")

        # Note: unlike the upstream repo's scripts/run_demo.py, we don't read
        # the checkpoint folder's cfg.yaml -- that file only supplies default
        # CLI arg values for their demo script. The checkpoint itself already
        # carries a populated `.args` (saved at training/export time); we just
        # override the two fields we actually want to control, same as
        # run_demo.py does after its own cfg.yaml merge.
        log.info("Loading Fast-FoundationStereo checkpoint %s on %s ...", checkpoint_path, self._device)
        model = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        model.args.valid_iters = self._valid_iters
        model.args.max_disp = self._max_disp
        model.to(self._device).eval()

        self._torch = torch
        self._InputPadder = InputPadder
        self._model = model
        self._amp_dtype = torch.float16
        log.info("FastFoundationStereoDepthComputer ready: size=%s fx_rect=%.1f baseline=%.3f m "
                  "valid_iters=%d max_disp=%d b_is_left=%s",
                  self._size, self._fx_rect, self._baseline_m,
                  self._valid_iters, self._max_disp, self._b_is_left)

    def _infer_disparity(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
        """Run FFS on one (left, right) ordered pair; returns a plain float
        disparity map aligned to `left_bgr`'s pixel grid. Fast-FoundationStereo
        is a rectified, left-image-referenced model exactly like SGBM (its own
        README warns "Do not swap left and right image") -- the caller applies
        the same b_is_left-aware flip-trick StereoDepthComputer's SGBM path
        uses to get the other camera's map, see _compute_disparity_pair."""
        torch = self._torch
        H, W = left_bgr.shape[:2]
        # FFS expects RGB (imageio.imread order in its own demo script);
        # frames here are BGR (OpenCV convention throughout this codebase).
        left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)

        img0 = torch.as_tensor(left_rgb).to(self._device).float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(right_rgb).to(self._device).float()[None].permute(0, 3, 1, 2)
        padder = self._InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)

        with torch.no_grad(), torch.amp.autocast(
            "cuda", enabled=self._device.startswith("cuda"), dtype=self._amp_dtype
        ):
            disp = self._model.forward(img0, img1, iters=self._valid_iters, test_mode=True,
                                        optimize_build_volume="pytorch1")
        disp = padder.unpad(disp.float())
        return disp.data.cpu().numpy().reshape(H, W).clip(0, None)

    def _compute_disparity_pair(self, rect_a_bgr: np.ndarray, rect_b_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Same b_is_left-aware flip-trick pattern as StereoDepthComputer's
        # SGBM path (stereo_depth.py) -- run once with the physically-left
        # image first for the "native" camera's map, once mirrored for the
        # other, then flip that result back.
        if self._b_is_left:
            disp_b = self._infer_disparity(rect_b_bgr, rect_a_bgr)
            disp_a = np.fliplr(self._infer_disparity(
                np.fliplr(rect_a_bgr).copy(), np.fliplr(rect_b_bgr).copy()))
        else:
            disp_a = self._infer_disparity(rect_a_bgr, rect_b_bgr)
            disp_b = np.fliplr(self._infer_disparity(
                np.fliplr(rect_b_bgr).copy(), np.fliplr(rect_a_bgr).copy()))
        return disp_a, disp_b
