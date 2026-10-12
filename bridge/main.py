from __future__ import annotations

import argparse
import asyncio
import logging
import os

from .control_server import serve
from .remote_config import ConfigError, ConfigProvider, check_url, parse_overrides
from .session import BridgeSession

log = logging.getLogger("dibr_bridge.main")


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
    # Stereo tuning — see remote_config.py and README.md's "Remote stereo
    # config". Env var fallbacks matter because Unity launches the bridge
    # with fixed args; they let settings be swapped locally without touching
    # Unity C#. Precedence: CLI > env > config file/URL > built-in defaults.
    parser.add_argument("--config-url", default=None,
                         help="https:// URL of a stereo config JSON (env DIBR_CONFIG_URL); re-fetched on "
                              "every set_active_pair, last-good copy cached locally")
    parser.add_argument("--config-path", default=None,
                         help="Local stereo config JSON for offline dev (env DIBR_CONFIG_PATH); beats --config-url")
    parser.add_argument("--stereo-backend", choices=["auto", "sgbm", "ffs"], default=None,
                         help="Override the config's backend (env DIBR_STEREO_BACKEND). 'auto' (built-in "
                              "default): ffs if a checkpoint, torch and CUDA are available, else sgbm with a "
                              "warning; 'ffs' still falls back to sgbm if it can't run")
    parser.add_argument("--stereo-override", action="append", default=[], metavar="KEY=VALUE",
                         help="Override one stereo config key, e.g. max_disp=128 or fp16=true; repeatable "
                              "(env DIBR_STEREO_OVERRIDES, comma-separated)")
    parser.add_argument("--ffs-checkpoint", default=os.environ.get("DIBR_FFS_CHECKPOINT"),
                         help="Path to a Fast-FoundationStereo .pth checkpoint (default: first "
                              "weights/<name>/model_best_bp2_serialize.pth next to the exe/repo, "
                              "then under the Fast-FoundationStereo repo)")
    parser.add_argument("--ffs-device", default=os.environ.get("DIBR_FFS_DEVICE", "cuda"),
                         help="torch device for the ffs backend")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    try:
        provider = build_config_provider(args, os.environ)
    except ConfigError as e:
        parser.error(str(e))

    asyncio.run(_run(args.server_url, args.rtsp_base, args.width, args.height, args.room_code,
                      provider, args.ffs_checkpoint, args.ffs_device))


def build_config_provider(args: argparse.Namespace, environ) -> ConfigProvider:
    """CLI values beat env values. A bad CLI value raises ConfigError (the
    caller exits with a usage error); a bad env value is logged and ignored,
    since the bridge must still start on an end-user machine."""
    overrides = {}
    env_text = ",".join(filter(None, [
        f"backend={environ['DIBR_STEREO_BACKEND']}" if environ.get("DIBR_STEREO_BACKEND") else None,
        environ.get("DIBR_STEREO_OVERRIDES"),
    ]))
    try:
        overrides.update(parse_overrides(env_text, "env DIBR_STEREO_BACKEND/DIBR_STEREO_OVERRIDES"))
    except ConfigError as e:
        log.warning("Ignoring env stereo overrides: %s", e)
    cli_items = list(args.stereo_override) + ([f"backend={args.stereo_backend}"] if args.stereo_backend else [])
    overrides.update(parse_overrides(",".join(cli_items), "--stereo-override/--stereo-backend"))

    path = args.config_path or environ.get("DIBR_CONFIG_PATH") or None
    url = args.config_url or environ.get("DIBR_CONFIG_URL") or None
    if url and not path:
        try:
            check_url(url)
        except ConfigError:
            if args.config_url:
                raise
            log.warning("Ignoring env DIBR_CONFIG_URL: %s", url)
            url = None
    return ConfigProvider(url=url, path=path, overrides=overrides)


async def _run(server_url: str, rtsp_base: str, width: int, height: int, room_code: str,
                config_provider: ConfigProvider | None = None, ffs_checkpoint: str | None = None,
                ffs_device: str = "cuda") -> None:
    session = BridgeSession(server_url, rtsp_base, width, height, room_code, config_provider=config_provider,
                             ffs_checkpoint=ffs_checkpoint, ffs_device=ffs_device)
    await session.start()
    await serve(session)


if __name__ == "__main__":
    main()
