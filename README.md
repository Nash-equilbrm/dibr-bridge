# dibr-bridge

Phase A of the live DIBR camera-switch feature (see
`Master-Thesis-Reports/handoff_spec_Sep_20th_opendibr_live_control_and_export.md`
for the OpenDIBR side, and `C:\Users\Admin\.claude\plans\tranquil-seeking-clarke.md`
— or wherever that plan lands once committed — for the full picture).

**Status (2026-09-20): fully packaged and bundled, not yet run against a
live room.** Dependencies install clean, all modules import successfully,
the `livekit` SDK usage in `livekit_source.py` is verified directly against
the actual installed package source (v1.1.19, not docs/examples — see that
file's docstring), and the bridge is now frozen into a standalone
`dibr-bridge.exe` (PyInstaller `--onedir`, via `entrypoint.py` — `main.py`
can't be the PyInstaller target directly, its relative imports need `bridge`
loaded as a real package) with no system Python required. Smoke-tested: the
frozen exe's `--help` runs clean (confirms all native deps — livekit, cv2,
numpy — bundled correctly) and a real launch against an unreachable server
correctly reaches its actual HTTP call before failing (crashes with a raw
traceback rather than a clean retry — pre-existing behavior, not a
packaging artifact; worth a `try/except` around `start()` at some point but
not blocking).

Everything is now bundled into `Master-Thesis-Client/Assets/StreamingAssets/`
for a self-contained Windows build: `DibrBridge/` (this, frozen, ~158MB),
`FFmpeg/` (~212MB, full_build for libx264 support), `MediaMTX/` (~54MB),
`OpenDIBR/` (~43MB, existing Debug build). ~467MB total — fine for a thesis
build, no app-store constraints; the FFmpeg full_build could be trimmed to
an "essentials" build later if size matters.

Still NOT run against a live LiveKit room/real camera devices — that's the
remaining gap, not a design or packaging uncertainty. To actually test:
registration-service stack up, real camera devices
registered+calibrated+streaming, then let `OpenDibrSessionManager` launch
mediamtx → bridge → OpenDIBR for real.

Local dev setup (already done once in this environment, only needed again
if re-freezing after a code change): a `.venv` exists here with
`requirements.txt` + `pyinstaller` installed;
`python -m PyInstaller --onedir --name dibr-bridge --console --noconfirm entrypoint.py`
rebuilds `dist/dibr-bridge/`, which then needs re-copying into
`Assets/StreamingAssets/DibrBridge/`.

## What this does

Subscribes to camera tracks in the same LiveKit room the Unity apps use,
computes live stereo depth between whichever two cameras are the current
"active pair," and republishes color + depth as RTSP streams that a patched
OpenDIBR process reads directly.

```
Unity viewer  --TCP control (port 40125)-->  dibr-bridge  --RTSP-->  OpenDIBR
                                                  |
                                                  +--subscribes to LiveKit tracks directly
                                                  +--fetches calibration from registration-service
```

Full control-channel contract (this is the authoritative spec — build to it):
see the "Control channel contract" block in the plan file's Phase A section.
Summary: TCP port 40125, newline-delimited JSON, three commands —
`add_camera` (per-identity, starts color passthrough, returns stable RTSP
URLs), `remove_camera`, `set_active_pair` (pairwise — (re)computes stereo
depth for exactly these two identities, must be called on every pair change).

## Requires

- Python 3.11+
- An RTSP server already running locally for ffmpeg to push into and for
  OpenDIBR to pull from — this bridge does not implement its own RTSP server.
  [mediamtx](https://github.com/bluenviron/mediamtx) is the same one
  referenced in the OpenDIBR handoff spec's Item 1 test instructions; run it
  with its default config (RTSP on `:8554`) alongside this bridge.
- `ffmpeg` on PATH (for encoding raw frames into RTSP streams).
- The registration-service (`Master-Thesis-Server/registration-service`)
  reachable — used for `/viewer-token` (LiveKit room join) and
  `/calibration-data/pair` (stereo calibration).

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```
python -m bridge.main --server-url http://localhost:3000 --rtsp-base rtsp://localhost:8554
```

(`--server-url` = the registration-service base URL; `--rtsp-base` = the
mediamtx RTSP base URL streams get pushed to and read back from.)

## Known gaps / next steps for whoever picks this up

- **Resolved 2026-09-20**: `bridge/livekit_source.py`'s LiveKit SDK usage —
  verified directly against installed package source, not docs. See its
  module docstring; the earlier defensive `getattr` fallback was removed.
- A permanent placeholder color+depth stream is started at bridge startup
  (`PLACEHOLDER_NAME` in `session.py`) — required because OpenDIBR refuses
  to launch with zero cameras in its startup JSON (confirmed by opendibr-c3
  actually launching it). Never selected into a real pairing; exists purely
  to satisfy that check. Its resolution must match `--width`/`--height`.
- `bridge/stereo_depth.py`: `cv2.reprojectImageTo3D` / the `Q` matrix path
  isn't used — depth is computed directly via `depth = fx * baseline /
  disparity` per the plan, which is simpler but doesn't account for
  post-rectification principal point shifts the way `Q` does; revisit if
  depth accuracy is off in a way that looks like a systematic bias rather
  than noise.
- No retry/reconnect logic yet for a LiveKit track that drops mid-session.
- `rtsp_publisher.py`'s ffmpeg invocation is a first-pass encode command
  (`libx264`, `ultrafast`, low-latency flags) — untuned; revisit bitrate/GOP
  settings once OpenDIBR's actual decode-side tolerance is known.
- **Resolved 2026-09-20** (was previously a gap): depth encoding — confirmed
  with opendibr-c3 against the actual decode shader (`src/vertex.fs`).
  `encode_depth_yuv420p` now does inverse-depth (disparity-like) encoding
  into an 8-bit YUV420p frame (depth in the Y plane, chroma ignored), not
  the earlier linear-16-bit-grayscale placeholder — that placeholder would
  not have decoded correctly (OpenDIBR's H.264/HEVC decoder only recognizes
  YUV420P/10LE/12LE; a raw 16-bit grayscale stream would hit its
  "ChromaFormat not recognized" fallback and silently corrupt the data). The
  chosen 8-bit depth over 10/12-bit HEVC Main10/12 is a documented
  compatibility tradeoff (avoids depending on an HEVC Main10/12 encoder
  being available) — revisit if 8-bit banding proves visually insufficient.
  `bitDepthColor`/`bitDepthDepth` (both 8, matching this) are now sent in
  OpenDIBR's `add_camera` call from the Unity side.
