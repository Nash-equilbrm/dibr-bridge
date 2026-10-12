"""Remote/local stereo tuning config, so max_disp, valid_iters, inference
resolution, fp16 and the backend can be changed without rebuilding the
bridge. See config.example.json and README.md's "Remote stereo config".

Security: the bridge runs on end-user machines, so a config file may only
set the whitelisted numbers/enums in STEREO_FIELDS below — never paths,
checkpoint locations, URLs, commands or env vars. Anything else in the file
is ignored with a warning. Remote URLs must be https (plain http only for
localhost, for dev).

Precedence, highest first: CLI overrides > env overrides > config source >
built-in defaults. Config source: local path (if given) beats URL; if the
chosen source fails, the last-good copy cached on disk is used, then the
built-in defaults. ConfigProvider.load() never raises.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

log = logging.getLogger("dibr_bridge.remote_config")

SCHEMA_VERSION = 1
FETCH_TIMEOUT_S = 2.0
MAX_CONFIG_BYTES = 64 * 1024
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}

# The project's tuning file (a public gist; this Raw URL always serves its
# newest revision). Used when no --config-path/--config-url or
# DIBR_CONFIG_PATH/DIBR_CONFIG_URL is given, so end users need no setup.
# Opt out with --config-url off or DIBR_CONFIG_URL=off (see is_off).
DEFAULT_CONFIG_URL = ("https://gist.githubusercontent.com/Nash-equilbrm/"
                      "e5d57617062f0dd1222c8bea30ed7163/raw/dibr-config.json")
_OFF_VALUES = {"", "off", "none"}


def is_off(value: str) -> bool:
    """True for a --config-url/DIBR_CONFIG_URL value that disables remote config."""
    return value.strip().lower() in _OFF_VALUES


@dataclasses.dataclass(frozen=True)
class StereoConfig:
    backend: str = "auto"
    max_disp: int = 192
    valid_iters: int = 4
    # None = infer at the bridge's output size (--width/--height).
    infer_width: int | None = None
    infer_height: int | None = None
    fp16: bool = False

    def describe(self) -> str:
        size = f"{self.infer_width}x{self.infer_height}" if self.infer_width else "output size"
        return (f"backend={self.backend} max_disp={self.max_disp} valid_iters={self.valid_iters} "
                f"infer={size} fp16={self.fp16}")


DEFAULTS = StereoConfig()


def _int_field(lo: int, hi: int, multiple: int):
    def check(v):
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"must be an integer, got {v!r}")
        if not lo <= v <= hi or v % multiple:
            raise ValueError(f"must be {lo}..{hi} and a multiple of {multiple}, got {v}")
        return v
    return check


def _enum_field(*allowed):
    def check(v):
        if v not in allowed:
            raise ValueError(f"must be one of {list(allowed)}, got {v!r}")
        return v
    return check


def _bool_field(v):
    if not isinstance(v, bool):
        raise ValueError(f"must be true/false, got {v!r}")
    return v


# The complete whitelist of what a config may set. Multiples of 8 for the
# inference size (not 32) so 1280x720 and 640x360 are expressible; the FFS
# backend pads to a multiple of 32 internally anyway.
STEREO_FIELDS = {
    "backend": _enum_field("auto", "sgbm", "ffs"),
    "max_disp": _int_field(32, 512, 32),
    "valid_iters": _int_field(1, 32, 1),
    "infer_width": _int_field(320, 1920, 8),
    "infer_height": _int_field(180, 1080, 8),
    "fp16": _bool_field,
}


class ConfigError(ValueError):
    pass


def parse_stereo_section(stereo: dict, where: str) -> dict:
    """Validate a {"key": value} dict against STEREO_FIELDS. Unknown keys are
    warned about and dropped; any invalid known key rejects the whole input,
    so a half-valid file never gets half-applied."""
    if not isinstance(stereo, dict):
        raise ConfigError(f"{where}: 'stereo' must be an object")
    out = {}
    for key, value in stereo.items():
        check = STEREO_FIELDS.get(key)
        if check is None:
            log.warning("%s: ignoring unknown stereo key %r", where, key)
            continue
        try:
            out[key] = check(value)
        except ValueError as e:
            raise ConfigError(f"{where}: stereo.{key} {e}") from None
    if ("infer_width" in out) != ("infer_height" in out):
        raise ConfigError(f"{where}: infer_width and infer_height must be set together")
    return out


def parse_config_document(doc, where: str) -> dict:
    if not isinstance(doc, dict):
        raise ConfigError(f"{where}: top level must be a JSON object")
    if doc.get("version") != SCHEMA_VERSION:
        raise ConfigError(f"{where}: unsupported version {doc.get('version')!r} (expected {SCHEMA_VERSION})")
    for key in doc:
        if key not in ("version", "stereo"):
            log.warning("%s: ignoring unknown top-level key %r", where, key)
    return parse_stereo_section(doc.get("stereo", {}), where)


def parse_overrides(text: str | None, where: str) -> dict:
    """'key=value,key=value' (env DIBR_STEREO_OVERRIDES / repeated
    --stereo-override). Values are parsed as JSON scalars, falling back to a
    plain string, so max_disp=128, fp16=true and backend=sgbm all work."""
    if not text:
        return {}
    raw = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        key, sep, value = item.partition("=")
        if not sep:
            raise ConfigError(f"{where}: expected key=value, got {item!r}")
        try:
            raw[key.strip()] = json.loads(value.strip())
        except json.JSONDecodeError:
            raw[key.strip()] = value.strip()
    return parse_stereo_section(raw, where)


def check_url(url: str) -> None:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme == "https" and parts.hostname:
        return
    if parts.scheme == "http" and parts.hostname in _LOCAL_HOSTS:
        return
    raise ConfigError(f"config URL must be https:// (http:// only for localhost): {url}")


def default_cache_path() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "dibr-bridge" / "config_cache.json"


class ConfigProvider:
    def __init__(self, url: str | None = None, path: str | None = None,
                 overrides: dict | None = None, cache_path: Path | None = None,
                 origin: str | None = None):
        self._url = url
        # Where the url/path came from, for the startup log line (e.g.
        # "default URL", "env DIBR_CONFIG_URL", "remote config disabled by
        # --config-url off").
        self._origin = origin
        self._path = path
        self._overrides = overrides or {}
        self._cache_path = cache_path if cache_path is not None else default_cache_path()
        if url and not path:
            check_url(url)  # fail fast on a bad CLI/env value, before any fetch
        self._last_summary: str | None = None

    def load(self) -> StereoConfig:
        """Blocking (network/disk) -- call via asyncio.to_thread. Never raises."""
        values, source = self._load_source()
        values = {**values, **self._overrides}
        if self._overrides:
            source += f" + overrides {sorted(self._overrides)}"
        cfg = dataclasses.replace(DEFAULTS, **values)
        summary = f"Stereo config from {source}: {cfg.describe()}"
        if summary != self._last_summary:
            log.info("%s", summary)
            self._last_summary = summary
        return cfg

    def _load_source(self) -> tuple[dict, str]:
        if self._path:
            try:
                text = Path(self._path).read_text(encoding="utf-8")
                return parse_config_document(json.loads(text), self._path), self._label(f"local file {self._path}")
            except (OSError, ValueError) as e:
                log.warning("Config file unusable, falling back: %s", e)
        elif self._url:
            try:
                return self._fetch_url(), self._label(f"URL {self._url}")
            except (OSError, ValueError) as e:
                log.warning("Config URL unusable, falling back: %s", e)

        if self._url or self._path:
            cached = self._read_cache()
            if cached is not None:
                return cached, self._label(f"last-good cache {self._cache_path}")
        return {}, self._label("built-in defaults")

    def _label(self, source: str) -> str:
        return f"{source} ({self._origin})" if self._origin else source

    def _fetch_url(self) -> dict:
        cache = self._read_cache_raw()
        req = urllib.request.Request(self._url, headers={"User-Agent": "dibr-bridge"})
        if cache and cache.get("url") == self._url and cache.get("etag"):
            req.add_header("If-None-Match", cache["etag"])
        try:
            with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as resp:
                # Redirects (e.g. GitHub raw) must stay on an allowed scheme too.
                check_url(resp.geturl())
                body = resp.read(MAX_CONFIG_BYTES + 1)
                etag = resp.headers.get("ETag")
        except urllib.error.HTTPError as e:
            if e.code == 304 and cache:
                return parse_config_document(cache["document"], f"{self._url} (304, cached)")
            raise
        if len(body) > MAX_CONFIG_BYTES:
            raise ConfigError(f"{self._url}: config larger than {MAX_CONFIG_BYTES} bytes")
        try:
            doc = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            # The usual symptom of a Google Drive share link: an HTML page.
            raise ConfigError(f"{self._url}: not valid JSON ({e}) -- is it a direct/raw link?") from None
        values = parse_config_document(doc, self._url)
        self._write_cache({"url": self._url, "etag": etag, "document": doc})
        return values

    def _read_cache_raw(self) -> dict | None:
        try:
            return json.loads(self._cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _read_cache(self) -> dict | None:
        cache = self._read_cache_raw()
        if not cache:
            return None
        try:
            return parse_config_document(cache.get("document"), f"cache {self._cache_path}")
        except ConfigError as e:
            log.warning("Ignoring unusable config cache: %s", e)
            return None

    def _write_cache(self, data: dict) -> None:
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            os.replace(tmp, self._cache_path)
        except OSError as e:
            log.warning("Could not write config cache %s: %s", self._cache_path, e)
