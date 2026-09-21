"""Scratch: sweep StereoSGBM params (no cv2.ximgproc available — plain
opencv-python only, no contrib) at alpha=-1 on the real capture, looking for
a combo that gives clean, high-coverage depth instead of the noisy/sparse
result from stereo_depth.py's current defaults.
"""
import itertools
import json
import logging
import os
import sys

import cv2
import numpy as np

import bridge.stereo_depth as sd
from bridge.registration_client import CameraCalibration

logging.disable(logging.CRITICAL)

CAPTURE_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    "../DibrDepthTestCaptures/cam3_cam2_20260921_181148"
WIDTH, HEIGHT = 1280, 720
OUT_DIR = "depth_out"
ALPHA = -1.0  # established as necessary in the earlier alpha sweep


def resize(frame):
    h, w = frame.shape[:2]
    if w == WIDTH and h == HEIGHT:
        return frame
    return cv2.resize(frame, (WIDTH, HEIGHT))


def build_computer(calib1, calib2):
    orig_rectify = cv2.stereoRectify
    def wrapper(*args, **kwargs):
        kwargs["alpha"] = ALPHA
        return orig_rectify(*args, **kwargs)
    sd.cv2.stereoRectify = wrapper
    computer = sd.StereoDepthComputer(calib1, calib2, out_size=(WIDTH, HEIGHT))
    sd.cv2.stereoRectify = orig_rectify
    return computer


def compute_with_params(computer, frame1, frame2, params, median_blur=0):
    """Rebuild the SGBM matcher with custom params and rerun compute_pair's
    logic inline (copies stereo_depth.py's compute_pair body so we can swap
    in the new matcher + optional post-filter)."""
    sgbm = cv2.StereoSGBM_create(**params)

    map_a_x, map_a_y = computer._rect_map_a
    map_b_x, map_b_y = computer._rect_map_b
    rect_a = cv2.remap(frame1, map_a_x, map_a_y, cv2.INTER_LINEAR)
    rect_b = cv2.remap(frame2, map_b_x, map_b_y, cv2.INTER_LINEAR)
    gray_a = cv2.cvtColor(rect_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(rect_b, cv2.COLOR_BGR2GRAY)

    disp_a_raw = sgbm.compute(gray_a, gray_b)
    disp_b_raw = cv2.flip(sgbm.compute(cv2.flip(gray_b, 1), cv2.flip(gray_a, 1)), 1)

    if median_blur:
        disp_a_raw = cv2.medianBlur(disp_a_raw.astype(np.int16), median_blur)
        disp_b_raw = cv2.medianBlur(disp_b_raw.astype(np.int16), median_blur)

    depth_a_rect = computer._disparity_to_depth(disp_a_raw)
    depth_b_rect = computer._disparity_to_depth(disp_b_raw)

    unrect_a_x, unrect_a_y = computer._unrect_map_a
    unrect_b_x, unrect_b_y = computer._unrect_map_b
    depth_a = cv2.remap(depth_a_rect, unrect_a_x, unrect_a_y, cv2.INTER_LINEAR)
    depth_b = cv2.remap(depth_b_rect, unrect_b_x, unrect_b_y, cv2.INTER_LINEAR)
    return depth_a, depth_b


def stats(depth):
    valid = depth[depth > 0]
    if valid.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return valid.size / depth.size, float(valid.min()), float(np.median(valid)), float(valid.max())


def main():
    with open(f"{CAPTURE_DIR}/calibration.json") as f:
        raw = json.load(f)
    calib1 = CameraCalibration.from_json(raw["cam1"])
    calib2 = CameraCalibration.from_json(raw["cam2"])
    depth_lo, depth_hi = raw.get("depthMin", 0.1), raw.get("depthMax", 10.0)

    computer = build_computer(calib1, calib2)
    frame1 = resize(cv2.imread(f"{CAPTURE_DIR}/{calib1.camera_name}.png"))
    frame2 = resize(cv2.imread(f"{CAPTURE_DIR}/{calib2.camera_name}.png"))

    base = dict(minDisparity=0, numDisparities=384, disp12MaxDiff=1)
    grid = []
    for blockSize in (3, 5, 9):
        for uniquenessRatio in (10, 20, 30):
            for speckle in ((100, 32), (50, 8), (200, 1)):
                for preFilterCap in (0, 31):
                    grid.append(dict(
                        **base, blockSize=blockSize,
                        P1=8 * 3 * blockSize ** 2, P2=32 * 3 * blockSize ** 2,
                        uniquenessRatio=uniquenessRatio,
                        speckleWindowSize=speckle[0], speckleRange=speckle[1],
                        preFilterCap=preFilterCap, mode=cv2.STEREO_SGBM_MODE_SGBM,
                    ))

    print(f"Sweeping {len(grid)} param combos (alpha={ALPHA}, mode=SGBM) ...")
    results = []
    for i, params in enumerate(grid):
        for mb in (0, 5):
            depth_a, depth_b = compute_with_params(computer, frame1, frame2, params, median_blur=mb)
            fa, mina, meda, maxa = stats(depth_a)
            fb, minb, medb, maxb = stats(depth_b)
            # Score: reward coverage + both medians landing inside the
            # server-declared plausible range + penalize max blowing way out.
            in_range = (depth_lo <= meda <= depth_hi) and (depth_lo <= medb <= depth_hi)
            score = (fa + fb) * (2.0 if in_range else 1.0) - 0.001 * (maxa + maxb)
            results.append((score, params, mb, (fa, mina, meda, maxa), (fb, minb, medb, maxb)))
        if (i + 1) % 20 == 0:
            print(f"  ...{i+1}/{len(grid)}", flush=True)

    results.sort(key=lambda r: -r[0])
    print("\nTop 5 combos:")
    for score, params, mb, sa, sb in results[:5]:
        p = {k: v for k, v in params.items() if k in
             ("blockSize", "uniquenessRatio", "speckleWindowSize", "speckleRange", "preFilterCap", "mode")}
        print(f"score={score:.3f} medianBlur={mb} {p}")
        print(f"   cam1: valid={sa[0]:.1%} min={sa[1]:.3f} median={sa[2]:.3f} max={sa[3]:.3f}")
        print(f"   cam2: valid={sb[0]:.1%} min={sb[1]:.3f} median={sb[2]:.3f} max={sb[3]:.3f}")

    # Save visualization of the best combo
    os.makedirs(OUT_DIR, exist_ok=True)
    best_score, best_params, best_mb, _, _ = results[0]
    depth_a, depth_b = compute_with_params(computer, frame1, frame2, best_params, median_blur=best_mb)

    def save(name, depth, color_frame, path):
        clipped = np.clip(depth, depth_lo, depth_hi)
        norm = ((clipped - depth_lo) / max(depth_hi - depth_lo, 1e-6) * 255).astype(np.uint8)
        colorized = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
        colorized[depth <= 0] = 0
        cv2.imwrite(path, np.hstack([color_frame, colorized]))
        print(f"wrote {path}")

    save(calib1.camera_name, depth_a, frame1, f"{OUT_DIR}/{calib1.camera_name}_best_tuned.png")
    save(calib2.camera_name, depth_b, frame2, f"{OUT_DIR}/{calib2.camera_name}_best_tuned.png")


if __name__ == "__main__":
    main()
