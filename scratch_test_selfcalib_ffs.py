"""Scratch: get real stereo depth from an arbitrary pair of photos with NO
known calibration, using a more robust recipe than
scratch_test_uncalibrated_ffs.py's pure cv2.stereoRectifyUncalibrated (which
degenerates on general camera motion -- confirmed on the plant-pot photos in
test_in/, even with 3000+ good feature matches).

Recipe: assume a generic/approximate camera intrinsic matrix K (no real
calibration -- just a plausible focal length for a typical phone camera),
then use the CALIBRATED pipeline's own math (essential matrix ->
recoverPose -> cv2.stereoRectify, the exact same stereoRectify call
BaseStereoDepthComputer uses) instead of the fragile uncalibrated projective
solver. This is far more numerically stable to general (rotating/
forward-moving) camera motion, at the cost of: (a) K being a guess, not
measured, and (b) the recovered translation is unit-length (scale is
inherently unobservable from two uncalibrated views), so the output is a
real, correctly-shaped disparity map, but NOT metric depth -- there is no
known baseline to convert px -> metres.

Usage:
    python scratch_test_selfcalib_ffs.py --left <img1> --right <img2> --checkpoint <path>
"""
import argparse
import os

import cv2
import numpy as np


def estimate_pose_and_rectify(frame1, frame2, K, alpha=-1.0):
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
        raise SystemExit("Too few matches to estimate an essential matrix.")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    inliers = int(mask.sum())
    print(f"Essential matrix inliers: {inliers}/{len(good)}")
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    print(f"Recovered relative rotation (deg): {np.degrees(cv2.Rodrigues(R)[0]).ravel()}")
    print(f"Recovered translation direction (unit, scale unknown): {t.ravel()}")

    D = np.zeros(5)
    h, w = gray1.shape
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K, D, K, D, (w, h), R, t, flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha)
    print(f"roi1={roi1} roi2={roi2}")
    map1x, map1y = cv2.initUndistortRectifyMap(K, D, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K, D, R2, P2, (w, h), cv2.CV_32FC1)
    rect1 = cv2.remap(frame1, map1x, map1y, cv2.INTER_LINEAR)
    rect2 = cv2.remap(frame2, map2x, map2y, cv2.INTER_LINEAR)
    return rect1, rect2, roi1, roi2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--checkpoint", default=os.environ.get("DIBR_FFS_CHECKPOINT"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--valid-iters", type=int, default=4)
    parser.add_argument("--max-disp", type=int, default=192)
    parser.add_argument("--width", type=int, default=640, help="Resize rectified pair to this width before running FFS")
    parser.add_argument("--hfov-deg", type=float, default=65.0, help="Assumed horizontal field of view (typical phone main camera)")
    parser.add_argument("--alpha", type=float, default=-1.0, help="cv2.stereoRectify crop: -1 auto, 0 tight crop, 1 keep full frame")
    args = parser.parse_args()
    if not args.checkpoint:
        raise SystemExit("Pass --checkpoint <path to .pth> or set DIBR_FFS_CHECKPOINT")

    frame1 = cv2.imread(args.left)
    frame2 = cv2.imread(args.right)
    if frame1 is None or frame2 is None:
        raise SystemExit(f"Failed to read one of: {args.left}, {args.right}")
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

    h, w = frame1.shape[:2]
    # No EXIF/known focal length -- assume a plausible phone-camera FOV.
    fx = fy = (w / 2) / np.tan(np.radians(args.hfov_deg) / 2)
    K = np.array([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], dtype=np.float64)
    print(f"Assumed K (NOT measured -- {args.hfov_deg} deg HFOV guess): fx=fy={fx:.1f}, cx={w/2:.1f}, cy={h/2:.1f}")

    rect1, rect2, roi1, roi2 = estimate_pose_and_rectify(frame1, frame2, K, alpha=args.alpha)

    os.makedirs("depth_out_selfcalib", exist_ok=True)
    cv2.imwrite("depth_out_selfcalib/rect1.png", rect1)
    cv2.imwrite("depth_out_selfcalib/rect2.png", rect2)
    print("Wrote depth_out_selfcalib/rect1.png, rect2.png -- check these look like normal photos, not a warped mess.")

    if args.width < w:
        scale = args.width / w
        new_size = (args.width, int(round(h * scale)))
        rect1 = cv2.resize(rect1, new_size)
        rect2 = cv2.resize(rect2, new_size)
        print(f"Resized to {new_size} for inference")

    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    import sys
    ffs_repo = os.environ.get("FFS_REPO_PATH", r"D:\WorkSpace\Master\Thesis\Repo\Fast-FoundationStereo")
    sys.path.insert(0, ffs_repo)
    import torch
    from core.utils.utils import InputPadder

    model = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.args.valid_iters = args.valid_iters
    model.args.max_disp = args.max_disp
    model.args.mixed_precision = False
    model.to(args.device).eval()

    left_rgb = cv2.cvtColor(rect1, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(rect2, cv2.COLOR_BGR2RGB)
    img0 = torch.as_tensor(left_rgb).to(args.device).float()[None].permute(0, 3, 1, 2)
    img1 = torch.as_tensor(right_rgb).to(args.device).float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0p, img1p = padder.pad(img0, img1)
    with torch.no_grad():
        disp_t = model.forward(img0p, img1p, iters=args.valid_iters, test_mode=True, optimize_build_volume="pytorch1")
    disp_t = padder.unpad(disp_t.float())
    Hc, Wc = rect1.shape[:2]
    disp = disp_t.data.cpu().numpy().reshape(Hc, Wc).clip(0, None)

    valid = disp[disp > 0]
    print(f"\nDisparity (pixels, NOT metric -- baseline scale unknown):")
    if valid.size:
        print(f"  valid={valid.size/disp.size:.1%}  min={valid.min():.1f}px  median={np.median(valid):.1f}px  max={valid.max():.1f}px")
    else:
        print("  EMPTY")

    vis_max = np.percentile(valid, 99) if valid.size else 1.0
    norm = (np.clip(disp, 0, vis_max) / max(vis_max, 1e-6) * 255).astype(np.uint8)
    colorized = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    colorized[disp <= 0] = 0
    cv2.imwrite("depth_out_selfcalib/disparity.png", np.hstack([rect1, colorized]))
    print("Wrote depth_out_selfcalib/disparity.png")


if __name__ == "__main__":
    main()
