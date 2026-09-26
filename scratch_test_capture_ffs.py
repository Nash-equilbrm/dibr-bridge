"""Scratch: run FastFoundationStereoDepthComputer against the same real
capture data scratch_test_capture2.py uses for StereoDepthComputer (SGBM),
so the two backends' output can be compared side by side on identical
inputs -- same calibration.json, same captured frames, same colorized-depth
visualization format. Also times compute_pair() (after a CUDA warmup call)
and reports peak VRAM, to check the ~66ms/tick (both cameras, DEPTH_FPS=15)
live budget from session.py against this machine's actual GPU -- see the
plan's Stage 4 gate: do not assume the upstream repo's RTX 3090 numbers
transfer to a GTX 1650 at this pipeline's resolution.

Requires requirements-ffs.txt installed and the Fast-FoundationStereo sibling
repo + a downloaded checkpoint -- see README.md's "Alternative stereo
backend" section. Usage mirrors scratch_test_capture2.py:

    python scratch_test_capture_ffs.py [capture_dir] --checkpoint weights/23-36-37/model_best_bp2_serialize.pth
"""
import argparse
import json
import logging
import os
import time

import cv2
import numpy as np

from bridge.registration_client import CameraCalibration
from bridge.stereo_depth_ffs import FastFoundationStereoDepthComputer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

OUT_DIR = "depth_out_ffs"


def resize(frame, width, height):
    h, w = frame.shape[:2]
    if w == width and h == height:
        return frame
    return cv2.resize(frame, (width, height))


def stats(name, depth):
    valid = depth[depth > 0]
    frac = valid.size / depth.size
    if valid.size == 0:
        print(f"  {name}: EMPTY (0 valid px)")
        return
    print(f"  {name}: valid={frac:.1%}  min={valid.min():.3f}m  "
          f"median={np.median(valid):.3f}m  max={valid.max():.3f}m")


def save_depth_visualization(name, depth, color_frame, vis_min, vis_max, out_path):
    clipped = np.clip(depth, vis_min, vis_max)
    normalized = ((clipped - vis_min) / max(vis_max - vis_min, 1e-6) * 255).astype(np.uint8)
    colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colorized[depth <= 0] = 0  # black out invalid pixels
    composite = np.hstack([color_frame, colorized])
    cv2.imwrite(out_path, composite)
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir", nargs="?", default="../DibrDepthTestCaptures/cam3_cam2_20260921_181148")
    parser.add_argument("--checkpoint", default=os.environ.get("DIBR_FFS_CHECKPOINT"),
                         help="Path to a Fast-FoundationStereo .pth checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--valid-iters", type=int, default=4)
    parser.add_argument("--max-disp", type=int, default=192)
    parser.add_argument("--width", type=int, default=1280, help="Output resolution (default matches session.py's live pipeline)")
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    if not args.checkpoint:
        raise SystemExit("Pass --checkpoint <path to .pth> or set DIBR_FFS_CHECKPOINT")

    with open(f"{args.capture_dir}/calibration.json") as f:
        raw = json.load(f)
    calib1 = CameraCalibration.from_json(raw["cam1"])
    calib2 = CameraCalibration.from_json(raw["cam2"])

    computer = FastFoundationStereoDepthComputer(
        calib1, calib2, out_size=(args.width, args.height),
        checkpoint_path=args.checkpoint, device=args.device,
        valid_iters=args.valid_iters, max_disp=args.max_disp,
    )

    frame1 = resize(cv2.imread(f"{args.capture_dir}/{calib1.camera_name}.png"), args.width, args.height)
    frame2 = resize(cv2.imread(f"{args.capture_dir}/{calib2.camera_name}.png"), args.width, args.height)

    # Same debug-capture upside-down quirk scratch_test_capture2.py corrects for.
    if calib2.camera_name == "cam2":
        frame2 = cv2.rotate(frame2, cv2.ROTATE_180)
    if calib1.camera_name == "cam2":
        frame1 = cv2.rotate(frame1, cv2.ROTATE_180)

    # Warmup call -- excluded from timing (1st run is slower due to CUDA/cudnn
    # compilation, per the upstream repo's own README tip).
    computer.compute_pair(frame1, frame2)

    import torch
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    depth1, depth2 = computer.compute_pair(frame1, frame2)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\ncompute_pair() latency: {elapsed_ms:.1f} ms  "
          f"(budget for BOTH cameras at DEPTH_FPS=15 is ~66.7 ms)")
    if args.device.startswith("cuda") and torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak GPU memory: {peak_mb:.0f} MB")

    print("Result:")
    stats(calib1.camera_name, depth1)
    stats(calib2.camera_name, depth2)

    os.makedirs(OUT_DIR, exist_ok=True)
    vis_min, vis_max = raw.get("depthMin", 0.1), raw.get("depthMax", 10.0)
    save_depth_visualization(calib1.camera_name, depth1, frame1, vis_min, vis_max,
                              f"{OUT_DIR}/{calib1.camera_name}_ffs.png")
    save_depth_visualization(calib2.camera_name, depth2, frame2, vis_min, vis_max,
                              f"{OUT_DIR}/{calib2.camera_name}_ffs.png")
    print(f"\nCompare against depth_out/*.png (SGBM, from scratch_test_capture2.py) for the same capture.")


if __name__ == "__main__":
    main()
