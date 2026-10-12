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

# Fast-FoundationStereo's core/submodule.py and core/utils/utils.py decorate a
# few functions with @torch.compile unconditionally (not gated behind the
# optimize_build_volume flag). torch.compile's default "inductor" backend
# needs a matching Triton build; there is no Triton build compatible with
# torch==2.6.0's inductor internals on Windows (confirmed: the community
# triton-windows package installs but raises
# "cannot import name 'AttrsDescriptor'" at model-load time — a version
# mismatch, not something a config flag on our side fixes). Disabling dynamo
# entirely makes @torch.compile a pass-through to eager execution instead of
# crashing; this only costs the fusion speedup on that one cost-volume-build
# op, not correctness. Must be set before `import torch` anywhere in this
# process — this module is the first thing to import torch (lazily, in
# _create_matcher below), so setting it here at module import time is early
# enough.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import dataclasses
import threading
import time

import cv2
import numpy as np

from .stereo_depth import BaseStereoDepthComputer

log = logging.getLogger("dibr_bridge.stereo_depth_ffs")

_DIBR_BRIDGE_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_FFS_REPO_PATH = _DIBR_BRIDGE_REPO_ROOT.parent / "Fast-FoundationStereo"
# Upstream's checkpoint filename inside every weights/<name>/ folder.
_CHECKPOINT_FILENAME = "model_best_bp2_serialize.pth"


def _ffs_repo_path() -> Path:
    override = os.environ.get("FFS_REPO_PATH")
    return Path(override) if override else _DEFAULT_FFS_REPO_PATH


def _app_dir() -> Path:
    """Directory of the frozen exe, or this repo's root when run from source."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return _DIBR_BRIDGE_REPO_ROOT


def find_default_checkpoint() -> Path | None:
    """Look for a checkpoint without any CLI/env config: first
    <app dir>/weights/<name>/<file> (the shipped layout), then the sibling
    Fast-FoundationStereo repo's own weights/ (the dev layout). Sorted so the
    pick is deterministic if more than one checkpoint is present."""
    for root in (_app_dir() / "weights", _ffs_repo_path() / "weights"):
        if root.is_dir():
            hits = sorted(root.glob(f"*/{_CHECKPOINT_FILENAME}")) or sorted(root.glob("*/*.pth"))
            if hits:
                return hits[0]
    return None


def unavailable_reason(checkpoint_path: str | None, device: str) -> str | None:
    """None if the ffs backend can run with these settings, else a short
    human-readable reason -- used by session.py to fall back to sgbm loudly
    instead of crashing the bridge."""
    if not checkpoint_path:
        return "no checkpoint found (pass --ffs-checkpoint, set DIBR_FFS_CHECKPOINT, or place one under weights/<name>/)"
    if not Path(checkpoint_path).is_file():
        return f"checkpoint not found: {checkpoint_path}"
    if not _ffs_repo_path().is_dir():
        return f"Fast-FoundationStereo repo not found at {_ffs_repo_path()} (set FFS_REPO_PATH)"
    try:
        import torch
    except ImportError:
        return "torch is not installed (see requirements-ffs.txt)"
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "CUDA is not available (no NVIDIA GPU/driver, or torch build lacks this GPU's arch)"
    return None


@dataclasses.dataclass
class _LoadedModel:
    model: object
    torch: object
    InputPadder: type
    # Serializes GPU inference: a cancelled depth pump's worker thread can
    # still be mid-inference when the next pair's first compute starts. Also
    # guards the per-call model.args writes in _run_inference.
    infer_lock: threading.Lock
    # The checkpoint's own max_disp, before we override it. Its disparity
    # attention is built for that range (core/foundation_stereo.py,
    # max_len=cfg['max_disp']//16), so asking for more isn't safe.
    trained_max_disp: int | None


# Process-wide model cache, keyed on checkpoint + device only. Loading the
# checkpoint (torch.load + .to(cuda)) is by far the most expensive step of a
# pair switch. max_disp, valid_iters and mixed_precision are only read from
# model.args inside forward(), so _run_inference sets them per call and
# changing them (e.g. via the remote config) never needs a reload.
_MODEL_CACHE: dict[tuple, _LoadedModel] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def load_model(checkpoint_path: str, device: str) -> _LoadedModel:
    key = (str(Path(checkpoint_path).resolve()), device)
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached

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

        # Note: unlike the upstream repo's scripts/run_demo.py, we don't read
        # the checkpoint folder's cfg.yaml -- that file only supplies default
        # CLI arg values for their demo script. The checkpoint itself already
        # carries a populated `.args` (saved at training/export time); the
        # fields we control are overridden per call in _run_inference.
        t0 = time.perf_counter()
        log.info("Loading Fast-FoundationStereo checkpoint %s on %s ...", checkpoint_path, device)
        model = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        trained_max_disp = getattr(model.args, "max_disp", None)
        model.to(device).eval()
        log.info("Fast-FoundationStereo checkpoint loaded in %.2fs (trained max_disp=%s)",
                 time.perf_counter() - t0, trained_max_disp)

        loaded = _LoadedModel(model, torch, InputPadder, threading.Lock(), trained_max_disp)
        _MODEL_CACHE[key] = loaded
        return loaded


def check_max_disp(loaded: _LoadedModel, max_disp: int) -> None:
    if loaded.trained_max_disp is not None and max_disp > loaded.trained_max_disp:
        raise ValueError(f"max_disp={max_disp} exceeds this checkpoint's trained "
                         f"max_disp={loaded.trained_max_disp}")


def warm_up(checkpoint_path: str, device: str, size: tuple[int, int], valid_iters: int = 4,
            max_disp: int = 192, mixed_precision: bool = False) -> None:
    """Load the model into the cache and run one throwaway inference at the
    live inference size, so CUDA context creation, cuDNN algorithm selection
    and allocator growth all happen before the first real set_active_pair.
    Blocking -- session.py runs it in a worker thread at bridge startup."""
    t0 = time.perf_counter()
    loaded = load_model(checkpoint_path, device)
    check_max_disp(loaded, max_disp)
    w, h = size
    dummy = np.zeros((h, w, 3), dtype=np.uint8)
    _run_inference(loaded, device, valid_iters, max_disp, mixed_precision, dummy, dummy)
    log.info("Fast-FoundationStereo warm-up done in %.2fs (%dx%d)", time.perf_counter() - t0, w, h)


def _run_inference(loaded: _LoadedModel, device: str, valid_iters: int, max_disp: int,
                   mixed_precision: bool, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    torch = loaded.torch
    H, W = left_bgr.shape[:2]
    # FFS expects RGB (imageio.imread order in its own demo script);
    # frames here are BGR (OpenCV convention throughout this codebase).
    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)

    with loaded.infer_lock:
        args = loaded.model.args
        args.valid_iters = valid_iters
        args.max_disp = max_disp
        args.mixed_precision = mixed_precision

        img0 = torch.as_tensor(left_rgb).to(device).float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(right_rgb).to(device).float()[None].permute(0, 3, 1, 2)
        padder = loaded.InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)

        # Note: the model gates its OWN internal autocast on
        # model.args.mixed_precision (set just above), independent of
        # this outer context -- this only affects the tensor prep above, kept
        # in sync so nothing here runs fp16 when mixed_precision is disabled.
        with torch.no_grad(), torch.amp.autocast(
            "cuda", enabled=device.startswith("cuda") and mixed_precision, dtype=torch.float16
        ):
            disp = loaded.model.forward(img0, img1, iters=valid_iters, test_mode=True,
                                        optimize_build_volume="pytorch1")
        disp = padder.unpad(disp.float())
        return disp.data.cpu().numpy().reshape(H, W).clip(0, None)


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
    checkpoints are on Google Drive under the NVIDIA Source Code License,
    non-commercial; no stable direct-download URL to automate) -- see
    README.md's "Licensing / redistribution".
    """

    def __init__(self, calib_a, calib_b, out_size: tuple[int, int] | None = None,
                 checkpoint_path: str | None = None, device: str = "cuda",
                 valid_iters: int = 4, max_disp: int = 192, mixed_precision: bool = False):
        self._checkpoint_path = checkpoint_path or os.environ.get("DIBR_FFS_CHECKPOINT")
        self._device = device
        self._valid_iters = valid_iters
        self._max_disp = max_disp
        # Default False: confirmed by direct testing (both on this rig's real
        # captures AND the upstream repo's own bundled demo_data, ruling out a
        # data-specific cause) that this checkpoint's internal fp16 autocast
        # path (model.args.mixed_precision=True, gated on U.AMP_DTYPE=float16
        # in Utils.py, independent of any outer torch.amp.autocast we apply)
        # produces all-NaN disparity on this GPU (GTX 1650, Turing/sm_75) --
        # forcing fp32 fixes it. Costs real throughput (this GPU's fp16 tensor
        # cores go unused) but correctness has to come first; if tested later
        # on a newer GPU (Ampere+) this may be safe to flip back to True.
        self._mixed_precision = mixed_precision
        super().__init__(calib_a, calib_b, out_size=out_size)

    def _create_matcher(self) -> None:
        _ensure_ffs_importable()
        if not self._checkpoint_path:
            raise RuntimeError(
                "Fast-FoundationStereo backend selected but no checkpoint given "
                "-- pass --ffs-checkpoint <path to a .pth under weights/<name>/> "
                "or set DIBR_FFS_CHECKPOINT. Research checkpoints are on Google "
                "Drive (no direct download URL to automate) -- see the "
                "Fast-FoundationStereo repo's README for the link."
            )
        checkpoint_path = Path(self._checkpoint_path)
        if not checkpoint_path.is_file():
            raise RuntimeError(f"Fast-FoundationStereo checkpoint not found: {checkpoint_path}")

        # Cached process-wide: only the first computer pays for the load.
        self._loaded = load_model(str(checkpoint_path), self._device)
        check_max_disp(self._loaded, self._max_disp)
        log.info("FastFoundationStereoDepthComputer ready: size=%s fx_rect=%.1f baseline=%.3f m "
                  "valid_iters=%d max_disp=%d mixed_precision=%s b_is_left=%s",
                  self._size, self._fx_rect, self._baseline_m,
                  self._valid_iters, self._max_disp, self._mixed_precision, self._b_is_left)

    def _infer_disparity(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
        """Run FFS on one (left, right) ordered pair; returns a plain float
        disparity map aligned to `left_bgr`'s pixel grid. Fast-FoundationStereo
        is a rectified, left-image-referenced model exactly like SGBM (its own
        README warns "Do not swap left and right image") -- the caller applies
        the same b_is_left-aware flip-trick StereoDepthComputer's SGBM path
        uses to get the other camera's map, see _compute_disparity_pair."""
        return _run_inference(self._loaded, self._device, self._valid_iters, self._max_disp,
                              self._mixed_precision, left_bgr, right_bgr)

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
