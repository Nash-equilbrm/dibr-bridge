"""Scratch: test whether depth quality is still compressed/poor on the real
rig's cam2/cam3 images even WITHOUT any of our calibration.json data in the
loop -- i.e. isolate "is FFS bad" / "is our calibration bad" by replacing
BaseStereoDepthComputer's calibration-driven cv2.stereoRectify with a fully
uncalibrated rectification estimated straight from feature matches between
the two images (cv2.findFundamentalMat + cv2.stereoRectifyUncalibrated).

No metric depth is possible this way (no known baseline/intrinsics), so this
only evaluates DISPARITY quality/coverage/coherence, not metres. Compare the
resulting disparity map's coverage and structure against the
calibration-driven runs (depth_out/, depth_out_ffs/) for the same capture.

Usage:
    python scratch_test_uncalibrated_ffs.py [capture_dir] --checkpoint <path to .pth>
"""
import argparse
import os

import cv2
import numpy as np

OUT_DIR = "depth_out_uncalibrated"


def load_pair(capture_dir):
    import json
    with open(f"{capture_dir}/calibration.json") as f:
        raw = json.load(f)
    name1 = raw["cam1"].get("camera_name") or "cam1"
    name2 = raw["cam2"].get("camera_name") or "cam2"
    # Fall back to scanning the directory for the two capture PNGs if the
    # calibration.json camera_name fields are empty (as seen on this rig).
    import glob
    pngs = sorted(glob.glob(f"{capture_dir}/*.png"))
    if len(pngs) < 2:
        raise SystemExit(f"Expected 2 capture PNGs in {capture_dir}, found {pngs}")
    print(f"Using images: {pngs[0]}, {pngs[1]}")
    frame1 = cv2.imread(pngs[0])
    frame2 = cv2.imread(pngs[1])
    # Same known upside-down capture-tool quirk scratch_test_capture2.py
    # corrects for -- a property of the capture PNG itself, not of the
    # calibration data, so it still applies here.
    if "cam2" in pngs[0]:
        frame1 = cv2.rotate(frame1, cv2.ROTATE_180)
    if "cam2" in pngs[1]:
        frame2 = cv2.rotate(frame2, cv2.ROTATE_180)
    return frame1, frame2


def estimate_uncalibrated_rectification(frame1, frame2):
    """Feature-match the two raw images and estimate rectifying homographies
    with zero calibration data -- cv2.findFundamentalMat +
    cv2.stereoRectifyUncalibrated, the classical uncalibrated-stereo recipe."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    print(f"SIFT keypoints: {len(kp1)} / {len(kp2)}")

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    print(f"Good matches (Lowe ratio test): {len(good)}")
    if len(good) < 8:
        raise SystemExit("Too few matches to estimate a fundamental matrix -- "
                          "images may not share enough overlap/texture.")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
    inliers = int(mask.sum())
    print(f"Fundamental matrix inliers: {inliers}/{len(good)}")
    pts1_in = pts1[mask.ravel() == 1]
    pts2_in = pts2[mask.ravel() == 1]

    h1, w1 = gray1.shape
    h2, w2 = gray2.shape
    ok, H1, H2 = cv2.stereoRectifyUncalibrated(pts1_in, pts2_in, F, (w1, h1))
    if not ok:
        raise SystemExit("cv2.stereoRectifyUncalibrated failed to converge.")
    return H1, H2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir", nargs="?", default="../DibrDepthTestCaptures/cam3_cam2_20260921_181148")
    parser.add_argument("--left", help="Explicit path to the left/first image (bypasses capture_dir + calibration.json entirely)")
    parser.add_argument("--right", help="Explicit path to the right/second image")
    parser.add_argument("--checkpoint", default=os.environ.get("DIBR_FFS_CHECKPOINT"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--valid-iters", type=int, default=4)
    parser.add_argument("--max-disp", type=int, default=192)
    parser.add_argument("--width", type=int, default=640, help="Resize rectified pair to this width before running FFS (GPU memory / FFS's own <1000px guidance)")
    args = parser.parse_args()
    if not args.checkpoint:
        raise SystemExit("Pass --checkpoint <path to .pth> or set DIBR_FFS_CHECKPOINT")

    if args.left and args.right:
        print(f"Using images: {args.left}, {args.right}")
        frame1 = cv2.imread(args.left)
        frame2 = cv2.imread(args.right)
        if frame1 is None or frame2 is None:
            raise SystemExit(f"Failed to read one of: {args.left}, {args.right}")
    else:
        frame1, frame2 = load_pair(args.capture_dir)
    H1, H2 = estimate_uncalibrated_rectification(frame1, frame2)

    h, w = frame1.shape[:2]
    rect1 = cv2.warpPerspective(frame1, H1, (w, h))
    rect2 = cv2.warpPerspective(frame2, H2, (w2 := frame2.shape[1], h2 := frame2.shape[0]))
    # Resize both to a common size for the network (uncalibrated rectification
    # doesn't guarantee identical output sizes across the two images).
    rect2 = cv2.resize(rect2, (w, h))

    # Downscale to a GPU-memory-friendly size (also matches FFS's own
    # guidance: "the model performs better for image width size <1000").
    if w > args.width:
        scale = args.width / w
        new_size = (args.width, int(round(h * scale)))
        rect1 = cv2.resize(rect1, new_size)
        rect2 = cv2.resize(rect2, new_size)
        print(f"Resized rectified pair to {new_size} for inference")

    os.makedirs(OUT_DIR, exist_ok=True)
    cv2.imwrite(f"{OUT_DIR}/rect1.png", rect1)
    cv2.imwrite(f"{OUT_DIR}/rect2.png", rect2)
    print(f"Wrote rectified pair to {OUT_DIR}/rect1.png, {OUT_DIR}/rect2.png -- "
          f"inspect these for horizontal epipolar alignment before trusting FFS output.")

    # Run FFS directly (no BaseStereoDepthComputer -- no calibration data
    # exists to build one). No metric depth possible without a known
    # baseline; report raw disparity coverage/coherence only.
    import sys
    sys.path.insert(0, r"D:\WorkSpace\Master\Thesis\Repo\dibr-bridge")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    ffs_repo = os.environ.get("FFS_REPO_PATH", r"D:\WorkSpace\Master\Thesis\Repo\Fast-FoundationStereo")
    sys.path.insert(0, ffs_repo)
    import torch
    from core.utils.utils import InputPadder

    model = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.args.valid_iters = args.valid_iters
    model.args.max_disp = args.max_disp
    model.args.mixed_precision = False
    model.to(args.device).eval()

    def infer(left_bgr, right_bgr):
        left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
        img0 = torch.as_tensor(left_rgb).to(args.device).float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(right_rgb).to(args.device).float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)
        with torch.no_grad():
            disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True,
                                  optimize_build_volume="pytorch1")
        disp = padder.unpad(disp.float())
        H, W = left_bgr.shape[:2]
        return disp.data.cpu().numpy().reshape(H, W).clip(0, None)

    disp = infer(rect1, rect2)
    valid = disp[disp > 0]
    frac = valid.size / disp.size
    print(f"\nRaw disparity (no metric conversion -- no baseline known):")
    if valid.size:
        print(f"  valid={frac:.1%}  min={valid.min():.1f}px  median={np.median(valid):.1f}px  max={valid.max():.1f}px")
    else:
        print("  EMPTY (0 valid px)")

    vis_max = np.percentile(valid, 99) if valid.size else 1.0
    normalized = (np.clip(disp, 0, vis_max) / max(vis_max, 1e-6) * 255).astype(np.uint8)
    colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colorized[disp <= 0] = 0
    cv2.imwrite(f"{OUT_DIR}/disparity.png", np.hstack([rect1, colorized]))
    print(f"Wrote {OUT_DIR}/disparity.png")


if __name__ == "__main__":
    main()
