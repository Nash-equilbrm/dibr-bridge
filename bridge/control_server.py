"""TCP, newline-delimited JSON control server — port 40125. Contract matches
Assets/Scripts/Dibr/DibrBridgeControlChannel.cs on the Unity side exactly
(see the plan file's Phase A section for the authoritative spec)."""
from __future__ import annotations

import asyncio
import json
import logging

from .session import BridgeSession

log = logging.getLogger("dibr_bridge.control_server")

PORT = 40125


async def _handle_client(session: BridgeSession, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    peer = writer.get_extra_info("peername")
    log.info("Control connection from %s", peer)
    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            try:
                reply = await _dispatch(session, json.loads(line))
            except Exception as e:
                log.warning("Command failed: %s", e)
                reply = {"ok": False, "error": str(e)}

            writer.write((json.dumps(reply) + "\n").encode("utf-8"))
            await writer.drain()
    except (ConnectionResetError, asyncio.IncompleteReadError):
        pass
    finally:
        writer.close()
        log.info("Control connection closed: %s", peer)


async def _dispatch(session: BridgeSession, msg: dict) -> dict:
    cmd = msg.get("cmd")

    if cmd == "add_camera":
        color_url, depth_url = await session.add_camera(msg["name"])
        return {"ok": True, "colorUrl": color_url, "depthUrl": depth_url}

    if cmd == "remove_camera":
        await session.remove_camera(msg["name"])
        return {"ok": True}

    if cmd == "set_active_pair":
        min_a, max_a, min_b, max_b = await session.set_active_pair(msg["camA"], msg["camB"])
        return {"ok": True, "depthMinA": min_a, "depthMaxA": max_a, "depthMinB": min_b, "depthMaxB": max_b}

    return {"ok": False, "error": f"Unknown command '{cmd}'"}


async def serve(session: BridgeSession) -> None:
    server = await asyncio.start_server(
        lambda r, w: _handle_client(session, r, w), host="127.0.0.1", port=PORT
    )
    log.info("Control server listening on 127.0.0.1:%d", PORT)
    async with server:
        await server.serve_forever()
