from __future__ import annotations

import argparse
import asyncio
import logging

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
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    asyncio.run(_run(args.server_url, args.rtsp_base, args.width, args.height))


async def _run(server_url: str, rtsp_base: str, width: int, height: int) -> None:
    session = BridgeSession(server_url, rtsp_base, width, height)
    await session.start()
    await serve(session)


if __name__ == "__main__":
    main()
