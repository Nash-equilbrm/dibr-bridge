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
`python -m PyInstaller --noconfirm --clean dibr-bridge.spec`
rebuilds `dist/dibr-bridge/`, which then needs re-copying into
`Assets/StreamingAssets/DibrBridge/`.

**Always build from `dibr-bridge.spec`, never `python -m PyInstaller
entrypoint.py --name dibr-bridge ...` directly** — passing a script (not a
`.spec` path) makes PyInstaller silently regenerate and overwrite
`dibr-bridge.spec` with bare defaults, which **deletes the
`collect_all('livekit')` fix** (needed to bundle `livekit_ffi.dll` — without
it the frozen exe raises `ImportError` on launch, before ever reaching
`--help` or any real code). This is exactly what happened here: an earlier
session's `collect_all` fix never actually made it into any build because
every rebuild since then followed the (wrong) instruction that used to be
written here, each time reverting its own fix. Confirmed 2026-09-21: after
rebuilding from the `.spec` file, `livekit_ffi.dll` is present and a real
launch reaches its actual HTTP call (`/viewer-token`) rather than an
`ImportError`.

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

## Stereo backends (FFS default when available)

NVIDIA's [Fast-FoundationStereo](https://github.com/nvlabs/fast-foundationstereo)
(zero-shot, GPU-based, no training needed) is the intended depth backend;
StereoSGBM stays in the code as the fallback. `--stereo-backend` (env
`DIBR_STEREO_BACKEND`) defaults to `auto`: ffs if a checkpoint is found and
torch + CUDA work, otherwise sgbm with a loud `FALLBACK to sgbm` warning
explaining why. Explicit `ffs` that can't run, or a failed model warm-up,
falls back the same way rather than crashing the bridge. Checkpoint lookup
when `--ffs-checkpoint`/`DIBR_FFS_CHECKPOINT` is unset: first
`<exe or repo dir>/weights/<name>/model_best_bp2_serialize.pth`, then the
sibling `Fast-FoundationStereo/weights/<name>/`. The model is loaded and
warmed up once at bridge startup in a worker thread, then reused for every
`set_active_pair`. This lives on the `feature/fast-foundationstereo-backend`
branch; `main` never depends on it.

Measured on an RTX 5070 (driver 591.86, torch 2.7.1+cu128, checkpoint
`23-36-37`, `valid_iters=4`, `max_disp=192`, 1280×720): torch import ~1.2s;
startup warm-up (load + first inference) ~2.9s; one inference ~280ms fp32 /
~240ms fp16; `compute_pair` (two inferences, one per camera) ~590ms fp32 /
~490ms fp16; building the per-pair computer with the cached model ~0.2s;
peak VRAM ~7.6GB. fp16 vs fp32 disparity on the upstream demo pair: MAE
0.008px, max 1.9px. fp32 stays the default (see stereo_depth_ffs.py: fp16
gave all-NaN output on a Turing GTX 1650).

**Rollback, cheapest to most complete:**
1. Same process/branch: pass `--stereo-backend sgbm` (or set
   `DIBR_STEREO_BACKEND=sgbm`) — restores the exact SGBM behavior instantly.
2. Same branch: the `ffs` code (`bridge/stereo_depth_ffs.py`) is only ever
   imported lazily inside `BridgeSession._build_depth`'s `"ffs"`
   branch — `StereoDepthComputer`/SGBM is never modified and needs no
   `requirements-ffs.txt` install to keep working.
3. Full rollback: `git checkout main` — this branch and everything on it
   disappears.

**Setup:**
1. Clone the fork **as a sibling repo**, not a submodule (same pattern as
   `client-sdk-unity` — see top-level `CLAUDE.md`), next to this repo:
   ```
   git clone https://github.com/Nash-equilbrm/Fast-FoundationStereo.git ../Fast-FoundationStereo
   ```
   (Override the expected location with the `FFS_REPO_PATH` env var if cloned
   elsewhere.) No commit is pinned automatically (same as `client-sdk-unity`'s
   `file:` reference) — this integration was built and tested against commit
   `476f4249561f7c79ca707326954f9255643412a6`.
2. In the same `.venv` as this repo:
   ```
   pip uninstall opencv-python -y
   pip install -r requirements-ffs.txt
   ```
   (`opencv-contrib-python` — required by the FFS repo — conflicts with plain
   `opencv-python` in the same venv; contrib is a superset so this doesn't
   affect the SGBM path's behavior. See comments in `requirements-ffs.txt`.)
3. Requires an NVIDIA GPU + a driver supporting CUDA 12.8 (the cu128 torch
   build is the first with RTX 50xx / sm_120 kernels). Confirmed working:
   RTX 5070 (numbers above) and, on the earlier cu124 build, a
   GTX 1650 Max-Q, 4GB VRAM, driver supports CUDA 12.7 — a much
   weaker mobile GPU than the RTX 3090 used for upstream's benchmark numbers
   (14–49ms/frame @ 640×480), so **real-time at this pipeline's 15fps/1280×720
   defaults is not guaranteed and must be measured**, not assumed — see
   `scratch_test_capture_ffs.py`.
4. Download a research checkpoint (Google Drive, no stable direct-download
   URL to automate; licensed under the NVIDIA Source Code License — see
   "Licensing / redistribution" below) per the Fast-FoundationStereo
   repo's own README, and place it under its `weights/<checkpoint-name>/`
   folder (e.g. `weights/23-36-37/model_best_bp2_serialize.pth`, plus its
   `cfg.yaml`).

**Licensing / redistribution** (our reading of the Fast-FoundationStereo
repo's `LICENSE.txt`, `model_card.md` and `readme.md` — not legal advice):
- The research checkpoints on Google Drive (e.g. `23-36-37`) fall under the
  same NVIDIA Source Code License as the code (`LICENSE.txt`; the model
  card applies it to the model). It allows reproduction and redistribution
  (§2.1), but only non-commercially, "for research purposes only" (§3.3).
- Any redistribution, e.g. bundling the checkpoint into the app's
  `StreamingAssets/DibrBridge/weights/`, must ship a complete copy of the
  FFS `LICENSE.txt` alongside it and keep its notices intact (§3.1).
- Only the separate C-Fast-FoundationStereo model on HuggingFace is under
  the [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/)
  (the commercial-use path). A commercial release would need that model
  and its terms, not the research checkpoints.
- The licence text names the "Software"/"Work" and does not mention weights
  explicitly; ask NVIDIA (contact in the FFS readme) if certainty matters.

**Try it offline first** (before touching the live pipeline):
```
python scratch_test_capture_ffs.py <capture_dir> --checkpoint ../Fast-FoundationStereo/weights/<name>/model_best_bp2_serialize.pth
```
Compares against `depth_out/*.png` (SGBM, from `scratch_test_capture2.py`)
and reports measured `compute_pair()` latency + peak VRAM against the
~66ms/tick budget (`DEPTH_FPS=15`, both cameras, in `session.py`).

**Then live** (once offline latency/quality look acceptable):
```
python -m bridge.main --server-url http://localhost:3000 --rtsp-base rtsp://localhost:8554 --room-code <CODE> ^
  --stereo-backend ffs --ffs-checkpoint ../Fast-FoundationStereo/weights/<name>/model_best_bp2_serialize.pth
```

PyInstaller packaging for this backend is intentionally not done — it's not
needed just to evaluate it, and would need its own `collect_all('torch')`
treatment plus a real size-impact assessment (torch+CUDA wheels are
multi-GB) if the backend is ever adopted as more than an experiment.
`dibr-bridge.spec` explicitly excludes `torch`/`torchvision`/`timm`, so
freezing from a venv that also has `requirements-ffs.txt` installed still
produces the ~180 MB SGBM-only exe (the ffs backend falls back to sgbm when
torch is missing).

## Remote stereo config

Stereo tuning can be changed without rebuilding or restarting, through a
small versioned JSON file (`bridge/remote_config.py`; the defaults are in
`config.example.json`):

```json
{"version": 1, "stereo": {"backend": "auto", "max_disp": 192, "valid_iters": 4,
                          "infer_width": 1280, "infer_height": 720, "fp16": false}}
```

| Key | Allowed values | Notes |
|---|---|---|
| `backend` | `auto`, `sgbm`, `ffs` | `auto`/`ffs` still fall back to sgbm if ffs can't run |
| `max_disp` | 32–512, multiple of 32 | ffs only; must also be ≤ the checkpoint's trained value (416 for `23-36-37`) |
| `valid_iters` | 1–32 | ffs only |
| `infer_width`, `infer_height` | 320–1920 / 180–1080, multiples of 8, set together | omit both to infer at `--width`/`--height` |
| `fp16` | `true`, `false` | ffs only |

All keys are optional; missing ones keep the built-in defaults. A known key
with a bad value rejects the whole file. Unknown keys are ignored with a
warning. **The file can only set these numbers/enums.** Checkpoint paths,
devices, URLs and commands stay CLI/env-only, because the bridge runs on
end-user machines.

**Sources and precedence** (highest first):
1. CLI: `--stereo-backend`, `--stereo-override KEY=VALUE` (repeatable).
2. Env: `DIBR_STEREO_BACKEND`, `DIBR_STEREO_OVERRIDES="max_disp=128,fp16=true"`.
3. The config file: `--config-path` / `DIBR_CONFIG_PATH` (local, for offline
   dev; beats any URL), else `--config-url`, else `DIBR_CONFIG_URL`, else the
   built-in default URL (the project gist below). If the file or URL is
   unusable, the last-good copy cached at
   `%LOCALAPPDATA%\dibr-bridge\config_cache.json` is used instead.
4. Built-in defaults.

A bad CLI value is a usage error. A bad env value is logged and ignored (a
bad `DIBR_CONFIG_URL` falls back to the default URL). The startup log line
names the source, e.g. `Stereo config from URL https://… (default URL): …`,
`… (env DIBR_CONFIG_URL)`, `last-good cache … (default URL)` or
`built-in defaults (remote config disabled by --config-url)`.

**When it's read:** at startup (which picks the backend and warm-up
settings) and again on every `set_active_pair`. Each read runs in a worker
thread with a 2 s timeout and `If-None-Match`/ETag, and never raises. One log
line shows the source and resolved values whenever they change. Every
setting applies on the next pair switch. None of them needs a model reload,
because the FFS model is cached per checkpoint and the tuning values are set
per inference. The new pair's depth computer is built and run once *before*
anything is swapped in. If that fails (e.g. GPU out of memory, `max_disp`
above the checkpoint's range), the switch uses the previous working settings
instead, or the built-in defaults on the first switch. When the inference
size differs from the output size, frames are resized down for matching and
depth is upsampled back with nearest-neighbour. The 1280×720 contract with
OpenDIBR doesn't change.

**The project's current config is the built-in default.** With no
`--config-path`/`--config-url`/`DIBR_CONFIG_PATH`/`DIBR_CONFIG_URL`, the
bridge reads `DEFAULT_CONFIG_URL` in `bridge/remote_config.py`, a public
gist whose always-latest Raw URL serves its newest revision (GitHub's raw
CDN can lag a few minutes after an edit):

```
https://gist.githubusercontent.com/Nash-equilbrm/e5d57617062f0dd1222c8bea30ed7163/raw/dibr-config.json
```

Editing that gist retunes every installed bridge on its next pair switch, so
the Unity launcher doesn't need to pass anything.

**Privacy:** unless disabled, every bridge start and every pair switch sends
one HTTPS request to `gist.githubusercontent.com` (GitHub sees the machine's
IP and a `dibr-bridge` User-Agent; nothing else is sent). **To disable:**
`--config-url off` (or an empty value) or env `DIBR_CONFIG_URL=off` (`none`
also works). Remote config is then skipped entirely, the cache is ignored
too, and only the built-in defaults plus CLI/env overrides apply.

**Reproducible experiments:** pin a specific gist revision instead of the
always-latest URL. On the gist page, open "Revisions", click "Raw" on the
revision you want, and pass that URL. It contains the revision hash
(`…/raw/<revision-sha>/dibr-config.json`) and never changes:

```
python -m bridge.main --server-url http://localhost:3000 --room-code <CODE> ^
  --config-url https://gist.githubusercontent.com/Nash-equilbrm/e5d57617062f0dd1222c8bea30ed7163/raw/<revision-sha>/dibr-config.json
```

Or use `--config-path` with a local copy, which needs no network at all.

**URLs and hosting:** only `https://` (plain `http://` only for
`localhost`/`127.0.0.1`, for dev); redirects must stay on an allowed scheme;
max 64 KB. Host the file somewhere that serves the raw bytes, e.g. a public
GitHub repo's `raw.githubusercontent.com/...` URL or a gist's "Raw" URL.
Google Drive share links don't work: they return an HTML page (preview or
virus-scan confirmation), which is rejected as invalid JSON, and they hit
download quotas. A Drive file only works as a direct-download link
(`https://drive.google.com/uc?export=download&id=<FILE_ID>`), and even that
is less reliable than GitHub raw.

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
