from __future__ import annotations

import argparse
import asyncio
import logging
import os

from .control_server import serve
from .session import BridgeSession


def main() -> None:
    parser = argparse.ArgumentParser(description="DIBR bridge — live stereo depth + RTSP serve for OpenDIBR")
    parser.add_argument("--server-url", required=True, help="registration-service base URL, e.g. http://localhost:3000")
    # Default matches OpenDibrSessionManager.cs's assumed convention — keep
    # these in sync if either side's default changes.
    parser.add_argument("--rtsp-base", default="rtsp://localhost:8554", help="RTSP server base URL to push into")
    # Must match the real cameras' resolution AND OpenDibrSessionManager.cs's
    # _outputWidth/_outputHeight (OpenDIBR's startup-JSON viewport locks the
    # render resolution; the placeholder camera here locks the comparison
    # add_camera checks new cameras against — see opendibr-c3's findings).
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--room-code", required=True, help="LiveKit room code to join, e.g. FADW88")
    parser.add_argument("--log-level", default="INFO")
    # Depth backend — see stereo_depth_ffs.py and README.md's "Stereo
    # backends" section. Env var fallback matters because Unity launches the
    # bridge with fixed args; this lets the backend be swapped locally
    # without touching Unity C#.
    parser.add_argument("--stereo-backend", choices=["auto", "sgbm", "ffs"],
                         default=os.environ.get("DIBR_STEREO_BACKEND", "auto"),
                         help="Depth backend: 'auto' (default: ffs if a checkpoint, torch and CUDA are "
                              "available, else sgbm with a warning), 'sgbm' (StereoSGBM) or "
                              "'ffs' (Fast-FoundationStereo; still falls back to sgbm if it can't run)")
    parser.add_argument("--ffs-checkpoint", default=os.environ.get("DIBR_FFS_CHECKPOINT"),
                         help="Path to a Fast-FoundationStereo .pth checkpoint (default: first "
                              "weights/<name>/model_best_bp2_serialize.pth next to the exe/repo, "
                              "then under the Fast-FoundationStereo repo)")
    parser.add_argument("--ffs-device", default=os.environ.get("DIBR_FFS_DEVICE", "cuda"),
                         help="torch device for the ffs backend (only used with --stereo-backend ffs)")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    asyncio.run(_run(args.server_url, args.rtsp_base, args.width, args.height, args.room_code,
                      args.stereo_backend, args.ffs_checkpoint, args.ffs_device))


async def _run(server_url: str, rtsp_base: str, width: int, height: int, room_code: str,
                stereo_backend: str = "sgbm", ffs_checkpoint: str | None = None, ffs_device: str = "cuda") -> None:
    session = BridgeSession(server_url, rtsp_base, width, height, room_code,
                             stereo_backend=stereo_backend, ffs_checkpoint=ffs_checkpoint, ffs_device=ffs_device)
    await session.start()
    await serve(session)


if __name__ == "__main__":
    main()
