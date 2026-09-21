"""Ad-hoc offline test: run StereoDepthComputer against real captured frames +
real calibration.json (from DibrDepthTestCaptures/), mirroring exactly what
session.py's set_active_pair does (resize to 1280x720, out_size=(1280,720)).
Not part of the test suite — scratch/diagnostic script.
"""
import json
import logging
import sys

import cv2
import numpy as np

from bridge.registration_client import CameraCalibration
from bridge.stereo_depth import StereoDepthComputer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

CAPTURE_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    "../DibrDepthTestCaptures/cam3_cam2_20260921_181148"
NUM_DISPARITIES = int(sys.argv[2]) if len(sys.argv) > 2 else None

WIDTH, HEIGHT = 1280, 720


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


def main():
    with open(f"{CAPTURE_DIR}/calibration.json") as f:
        raw = json.load(f)

    calib1 = CameraCalibration.from_json(raw["cam1"])
    calib2 = CameraCalibration.from_json(raw["cam2"])
    print(f"Pair: {calib1.camera_name} (calib {calib1.intrinsics.image_width}x"
          f"{calib1.intrinsics.image_height}, fx={calib1.intrinsics.fx:.1f})  <->  "
          f"{calib2.camera_name} (calib {calib2.intrinsics.image_width}x"
          f"{calib2.intrinsics.image_height}, fx={calib2.intrinsics.fx:.1f})")
    print(f"Server-declared depth range: [{raw.get('depthMin')}, {raw.get('depthMax')}]")

    frame1 = resize(cv2.imread(f"{CAPTURE_DIR}/{calib1.camera_name}.png"))
    frame2 = resize(cv2.imread(f"{CAPTURE_DIR}/{calib2.camera_name}.png"))
    print(f"Frames resized to {WIDTH}x{HEIGHT} to match session.py's pipeline output")

    computer = StereoDepthComputer(calib1, calib2, out_size=(WIDTH, HEIGHT))
    if NUM_DISPARITIES is not None:
        computer._sgbm.setNumDisparities(NUM_DISPARITIES)
        print(f"Overriding numDisparities -> {NUM_DISPARITIES}")

    depth1, depth2 = computer.compute_pair(frame1, frame2)
    print("Result:")
    stats(calib1.camera_name, depth1)
    stats(calib2.camera_name, depth2)


if __name__ == "__main__":
    main()
