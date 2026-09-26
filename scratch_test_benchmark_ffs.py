"""Scratch: run FFS on a public, already-rectified stereo benchmark pair with
known ground-truth disparity (classic Middlebury "tsukuba" pair, mirrored in
opencv_extra), and score predicted disparity directly against ground truth.

This sidesteps BOTH open questions from the rig/casual-photo tests: no
calibration data needed (comparing disparity, not metric depth) and no
rectification-estimation step needed (the benchmark images are already
properly rectified) -- so any error here is genuinely about FFS's matching
accuracy, not about our calibration or an ad-hoc rectification guess.

Usage:
    python scratch_test_benchmark_ffs.py --left im2.png --right im6.png --gt disp2.png --checkpoint <path>
"""
import argparse
import os

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--gt", required=True, help="Ground-truth disparity PNG (tsukuba convention: pixel_value / 16 = disparity px)")
    parser.add_argument("--gt-scale", type=float, default=16.0)
    parser.add_argument("--checkpoint", default=os.environ.get("DIBR_FFS_CHECKPOINT"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--valid-iters", type=int, default=4)
    parser.add_argument("--max-disp", type=int, default=192)
    args = parser.parse_args()
    if not args.checkpoint:
        raise SystemExit("Pass --checkpoint <path to .pth> or set DIBR_FFS_CHECKPOINT")

    left_bgr = cv2.imread(args.left)
    right_bgr = cv2.imread(args.right)
    gt_raw = cv2.imread(args.gt, cv2.IMREAD_UNCHANGED)
    if gt_raw.ndim == 3:
        gt_raw = gt_raw[:, :, 0]
    gt = gt_raw.astype(np.float32) / args.gt_scale
    H, W = left_bgr.shape[:2]

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

    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
    img0 = torch.as_tensor(left_rgb).to(args.device).float()[None].permute(0, 3, 1, 2)
    img1 = torch.as_tensor(right_rgb).to(args.device).float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0p, img1p = padder.pad(img0, img1)
    with torch.no_grad():
        disp_t = model.forward(img0p, img1p, iters=args.valid_iters, test_mode=True, optimize_build_volume="pytorch1")
    disp_t = padder.unpad(disp_t.float())
    pred = disp_t.data.cpu().numpy().reshape(H, W).clip(0, None)

    # Score against ground truth. Exclude gt==0 -- in this dataset that marks
    # the untextured background/unknown region, not a real zero-disparity
    # surface (only 8 discrete real disparity planes exist: 5,6,7,8,10,11,14).
    valid_gt = gt > 0
    err = np.abs(pred - gt)
    print(f"Predicted disparity: min={pred.min():.2f} max={pred.max():.2f} median={np.median(pred):.2f}")
    print(f"Ground truth (valid px only): min={gt[valid_gt].min():.1f} max={gt[valid_gt].max():.1f}  "
          f"planes={sorted(np.unique(gt[valid_gt]).tolist())}")
    print(f"\nError over {valid_gt.sum()}/{valid_gt.size} valid ground-truth pixels ({valid_gt.mean():.1%}):")
    print(f"  MAE = {err[valid_gt].mean():.2f} px   median = {np.median(err[valid_gt]):.2f} px")
    for thresh in (1, 2, 4):
        bad = (err[valid_gt] > thresh).mean()
        print(f"  bad-{thresh}px (error > {thresh}px) = {bad:.1%}")

    vis_max = gt[valid_gt].max()
    def colorize(d, mask=None):
        norm = (np.clip(d, 0, vis_max) / vis_max * 255).astype(np.uint8)
        c = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
        if mask is not None:
            c[~mask] = 0
        return c

    os.makedirs("depth_out_benchmark", exist_ok=True)
    composite = np.hstack([left_bgr, colorize(pred), colorize(gt, valid_gt)])
    cv2.imwrite("depth_out_benchmark/comparison.png", composite)
    print("\nWrote depth_out_benchmark/comparison.png (left image | FFS prediction | ground truth)")


if __name__ == "__main__":
    main()
