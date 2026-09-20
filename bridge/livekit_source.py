"""Wraps a single shared LiveKit room connection and exposes each
subscribed camera identity's latest decoded frame as a BGR numpy array
(OpenCV's expected channel order).

VERIFIED 2026-09-20 against the actual installed `livekit` package
(v1.1.19, read directly from .venv/Lib/site-packages/livekit/rtc/
video_stream.py and video_frame.py — not docs/examples this time). Confirms:
`VideoStream.__anext__` yields a `VideoFrameEvent` dataclass with a `.frame`
field (a `VideoFrame` with `.data`/`.width`/`.height` properties); `Room` has
`.remote_participants`; `RemoteParticipant` has `.track_publications`;
`RemoteTrackPublication` has `.kind`/`.set_subscribed()`. All match what this
file already assumed — no code changes needed, just removing the "unverified"
hedge and the now-unnecessary defensive `getattr` fallback in `_consume`
below (VideoFrameEvent.frame has no default, so it doesn't show via a bare
`dir()` on the class — a normal dataclass quirk, not evidence it might be
absent at the instance level).
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

import cv2
import numpy as np
from livekit import rtc


class LiveKitCameraHub:
    def __init__(self, livekit_url: str, token: str):
        self._room = rtc.Room()
        self._url = livekit_url
        self._token = token
        self._latest_frames: dict[str, np.ndarray] = {}
        self._wanted_identities: set[str] = set()
        self._consume_tasks: dict[str, asyncio.Task] = {}

    async def connect(self) -> None:
        self._room.on("track_subscribed", self._on_track_subscribed)
        await self._room.connect(
            self._url, self._token,
            options=rtc.RoomOptions(auto_subscribe=False),
        )

    async def disconnect(self) -> None:
        for task in self._consume_tasks.values():
            task.cancel()
        self._consume_tasks.clear()
        await self._room.disconnect()

    def _on_track_subscribed(self, track, publication, participant) -> None:
        if track.kind != rtc.TrackKind.KIND_VIDEO:
            return
        identity = participant.identity
        if identity not in self._wanted_identities:
            return  # a track we didn't ask for — ignore

        stream = rtc.VideoStream(track, format=rtc.VideoBufferType.RGBA)
        self._consume_tasks[identity] = asyncio.ensure_future(self._consume(identity, stream))

    async def _consume(self, identity: str, stream) -> None:
        try:
            async for event in stream:  # yields VideoFrameEvent
                frame = event.frame
                arr = np.frombuffer(frame.data, dtype=np.uint8).reshape((frame.height, frame.width, 4))
                self._latest_frames[identity] = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        except asyncio.CancelledError:
            pass

    async def subscribe(self, identity: str, timeout_s: float = 8.0) -> None:
        self._wanted_identities.add(identity)

        pub = self._find_video_publication(identity)
        if pub is None:
            # Participant/publication may not have arrived yet if this is
            # called right after connect() — give it a moment.
            deadline = time.time() + timeout_s
            while pub is None and time.time() < deadline:
                await asyncio.sleep(0.1)
                pub = self._find_video_publication(identity)
            if pub is None:
                raise RuntimeError(f"No video publication found for identity '{identity}'")

        pub.set_subscribed(True)

        deadline = time.time() + timeout_s
        while identity not in self._latest_frames:
            if time.time() > deadline:
                raise TimeoutError(f"No frame received from '{identity}' within {timeout_s}s")
            await asyncio.sleep(0.05)

    def unsubscribe(self, identity: str) -> None:
        self._wanted_identities.discard(identity)
        pub = self._find_video_publication(identity)
        if pub is not None:
            pub.set_subscribed(False)
        self._latest_frames.pop(identity, None)
        task = self._consume_tasks.pop(identity, None)
        if task is not None:
            task.cancel()

    def latest_frame(self, identity: str) -> Optional[np.ndarray]:
        return self._latest_frames.get(identity)

    def _find_video_publication(self, identity: str):
        participant = self._room.remote_participants.get(identity)
        if participant is None:
            return None
        for pub in participant.track_publications.values():
            if pub.kind == rtc.TrackKind.KIND_VIDEO:
                return pub
        return None
