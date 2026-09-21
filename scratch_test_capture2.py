"""Scratch: run StereoDepthComputer against real capture data with alpha and
numDisparities overridden, to check whether alpha=0 (crop-to-valid, current
hardcoded value in stereo_depth.py) is inflating fx_rect and thus the
required disparity search range beyond what's practical. Also dumps a
colorized depth map PNG per camera so the result can be eyeballed, not just
read off summary stats.
"""
import json
import logging
import os
import sys

import cv2
import numpy as np

import bridge.stereo_depth as sd
from bridge.registration_client import CameraCalibration

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

CAPTURE_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    "../DibrDepthTestCaptures/cam3_cam2_20260921_181148"
ALPHA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
NUM_DISP = int(sys.argv[3]) if len(sys.argv) > 3 else 384

WIDTH, HEIGHT = 1280, 720
OUT_DIR = "depth_out"


def resize(frame):
    h, w = frame.shape[:2]
    if w == WIDTH and h == HEIGHT:
        return frame
    return cv2.resize(frame, (WIDTH, HEIGHT))


def stats(name, depth):
    valid = depth[depth > 0]
    frac = valid.size / depth.size
    if valid.size == 0:
        print(f"  {name}: EMPTY (0 valid px)")
        return
    print(f"  {name}: valid={frac:.1%}  min={valid.min():.3f}m  "
          f"median={np.median(valid):.3f}m  max={valid.max():.3f}m")


def save_depth_visualization(name, depth, color_frame, vis_min, vis_max, out_path):
    """Colorize depth (clipped to [vis_min, vis_max], invalid px = black) and
    save it side-by-side with the source color frame for eyeballing."""
    clipped = np.clip(depth, vis_min, vis_max)
    normalized = ((clipped - vis_min) / max(vis_max - vis_min, 1e-6) * 255).astype(np.uint8)
    colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colorized[depth <= 0] = 0  # black out invalid pixels
    composite = np.hstack([color_frame, colorized])
    cv2.imwrite(out_path, composite)
    print(f"  wrote {out_path}")


def main():
    with open(f"{CAPTURE_DIR}/calibration.json") as f:
        raw = json.load(f)
    calib1 = CameraCalibration.from_json(raw["cam1"])
    calib2 = CameraCalibration.from_json(raw["cam2"])

    orig_rectify = cv2.stereoRectify
    def wrapper(*args, **kwargs):
        kwargs["alpha"] = ALPHA
        return orig_rectify(*args, **kwargs)
    sd.cv2.stereoRectify = wrapper

    computer = sd.StereoDepthComputer(calib1, calib2, out_size=(WIDTH, HEIGHT))
    sd.cv2.stereoRectify = orig_rectify
    computer._sgbm.setNumDisparities(NUM_DISP)

    print(f"alpha={ALPHA}  numDisparities={NUM_DISP}  "
          f"fx_rect={computer._fx_rect:.1f}  baseline={computer._baseline_m:.3f}m")
    print(f"  -> expected disparity at Z=1m: {computer._fx_rect*computer._baseline_m/1.0:.0f}px, "
          f"at Z=0.3m: {computer._fx_rect*computer._baseline_m/0.3:.0f}px, "
          f"at Z=5m: {computer._fx_rect*computer._baseline_m/5.0:.0f}px")

    frame1 = resize(cv2.imread(f"{CAPTURE_DIR}/{calib1.camera_name}.png"))
    frame2 = resize(cv2.imread(f"{CAPTURE_DIR}/{calib2.camera_name}.png"))

    # cam2's capture PNG comes out upside-down (DibrDepthTestCapture.cs debug
    # dump, not the same path the live LiveKit-published frames or the
    # calibration itself went through) -- correct the frame content only,
    # leave calib2's intrinsics/extrinsics untouched.
    if calib2.camera_name == "cam2":
        frame2 = cv2.rotate(frame2, cv2.ROTATE_180)
    if calib1.camera_name == "cam2":
        frame1 = cv2.rotate(frame1, cv2.ROTATE_180)

    depth1, depth2 = computer.compute_pair(frame1, frame2)
    print("Result:")
    stats(calib1.camera_name, depth1)
    stats(calib2.camera_name, depth2)

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"a{ALPHA}_nd{NUM_DISP}".replace("-", "neg").replace(".", "p")
    # Visualization range: the server's own declared depth range if sane,
    # else fall back to the actual valid min/max so the colormap isn't blown
    # out by a handful of noise outliers at hundreds of metres.
    vis_min, vis_max = raw.get("depthMin", 0.1), raw.get("depthMax", 10.0)
    save_depth_visualization(calib1.camera_name, depth1, frame1, vis_min, vis_max,
                              f"{OUT_DIR}/{calib1.camera_name}_{tag}.png")
    save_depth_visualization(calib2.camera_name, depth2, frame2, vis_min, vis_max,
                              f"{OUT_DIR}/{calib2.camera_name}_{tag}.png")


if __name__ == "__main__":
    main()
