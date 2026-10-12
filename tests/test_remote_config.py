"""Remote stereo config: validation, source fallback, caching and precedence.
No network needed -- remote fetches go to a throwaway local HTTP server."""
import argparse
import http.server
import json
import threading

import urllib.parse
import urllib.request

import pytest

from bridge import main as bridge_main
from bridge.main import build_config_provider
from bridge.remote_config import (DEFAULT_CONFIG_URL, DEFAULTS, ConfigError, ConfigProvider, check_url,
                                  parse_config_document, parse_overrides)

_DEAD_URL = "http://127.0.0.1:1/dibr-config.json"  # closed port: fails fast, offline


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch, tmp_path):
    """Tests never contact GitHub: the built-in default URL points at a dead
    local port, any non-localhost request fails loudly, and the last-good
    cache lives in a per-test temp dir."""
    monkeypatch.setattr(bridge_main, "DEFAULT_CONFIG_URL", _DEAD_URL)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    real_urlopen = urllib.request.urlopen

    def guarded(req, *a, **kw):
        url = req.full_url if isinstance(req, urllib.request.Request) else req
        assert urllib.parse.urlsplit(url).hostname in ("127.0.0.1", "localhost"), f"real network: {url}"
        return real_urlopen(req, *a, **kw)

    monkeypatch.setattr(urllib.request, "urlopen", guarded)

GOOD = {"version": 1, "stereo": {"backend": "ffs", "max_disp": 128, "valid_iters": 2,
                                  "infer_width": 640, "infer_height": 360, "fp16": True}}


# ── Validation ─────────────────────────────────────────────────────────────

def test_valid_document_parses():
    assert parse_config_document(GOOD, "t") == GOOD["stereo"]


@pytest.mark.parametrize("key,value", [
    ("max_disp", 100),        # not a multiple of 32
    ("max_disp", 1024),       # out of range
    ("max_disp", "192"),      # wrong type
    ("valid_iters", 0),
    ("valid_iters", True),    # bool is not an int here
    ("infer_width", 641),
    ("fp16", "yes"),
    ("backend", "cuda"),
])
def test_invalid_values_reject_the_whole_document(key, value):
    with pytest.raises(ConfigError):
        parse_config_document({"version": 1, "stereo": {key: value}}, "t")


def test_wrong_version_is_rejected():
    with pytest.raises(ConfigError):
        parse_config_document({"version": 2, "stereo": {}}, "t")


def test_infer_size_must_be_set_as_a_pair():
    with pytest.raises(ConfigError):
        parse_config_document({"version": 1, "stereo": {"infer_width": 640}}, "t")


def test_unknown_and_non_whitelisted_keys_are_ignored(caplog):
    doc = {"version": 1, "checkpoint": "C:/evil.pth",
           "stereo": {"max_disp": 96, "ffs_checkpoint": "C:/evil.pth", "command": "calc.exe"}}
    assert parse_config_document(doc, "t") == {"max_disp": 96}
    assert "checkpoint" in caplog.text and "command" in caplog.text


@pytest.mark.parametrize("url,ok", [
    ("https://raw.githubusercontent.com/u/r/main/c.json", True),
    ("http://localhost:8000/c.json", True),
    ("http://127.0.0.1:8000/c.json", True),
    ("http://example.com/c.json", False),
    ("file:///C:/c.json", False),
    ("ftp://example.com/c.json", False),
])
def test_url_scheme_whitelist(url, ok):
    if ok:
        check_url(url)
    else:
        with pytest.raises(ConfigError):
            check_url(url)


def test_overrides_parse_scalars():
    assert parse_overrides("max_disp=128, fp16=true,backend=sgbm", "t") == \
        {"max_disp": 128, "fp16": True, "backend": "sgbm"}
    with pytest.raises(ConfigError):
        parse_overrides("max_disp", "t")


# ── Sources and fallback ───────────────────────────────────────────────────

class _Server:
    """Serves self.body at /c.json with an ETag; counts 304s."""

    def __init__(self):
        self.body = json.dumps(GOOD).encode()
        self.etag = '"v1"'
        self.status = 200
        self.not_modified = 0
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if outer.status != 200:
                    self.send_response(outer.status)
                    self.end_headers()
                    return
                if self.headers.get("If-None-Match") == outer.etag:
                    outer.not_modified += 1
                    self.send_response(304)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("ETag", outer.etag)
                self.send_header("Content-Length", str(len(outer.body)))
                self.end_headers()
                self.wfile.write(outer.body)

            def log_message(self, *a):
                pass

        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/c.json"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


@pytest.fixture
def server():
    s = _Server()
    yield s
    s.httpd.shutdown()


def test_no_source_gives_defaults(tmp_path):
    assert ConfigProvider(cache_path=tmp_path / "c.json").load() == DEFAULTS


def test_url_fetch_then_etag_304_uses_cache(server, tmp_path):
    p = ConfigProvider(url=server.url, cache_path=tmp_path / "c.json")
    assert p.load().max_disp == 128
    assert p.load().max_disp == 128
    assert server.not_modified == 1


def test_url_failure_falls_back_to_last_good_cache(server, tmp_path):
    cache = tmp_path / "c.json"
    assert ConfigProvider(url=server.url, cache_path=cache).load().max_disp == 128
    server.status = 500
    assert ConfigProvider(url=server.url, cache_path=cache).load().max_disp == 128


def test_invalid_remote_document_falls_back_to_cache_then_defaults(server, tmp_path):
    cache = tmp_path / "c.json"
    server.body = b"<html>Google Drive - virus scan warning</html>"
    assert ConfigProvider(url=server.url, cache_path=cache).load() == DEFAULTS
    server.body, server.etag = json.dumps(GOOD).encode(), '"v2"'
    ConfigProvider(url=server.url, cache_path=cache).load()
    server.body, server.etag = json.dumps({"version": 1, "stereo": {"max_disp": 7}}).encode(), '"v3"'
    assert ConfigProvider(url=server.url, cache_path=cache).load().max_disp == 128  # bad v3 -> cached v2


def test_unreachable_url_never_raises(tmp_path):
    p = ConfigProvider(url="http://127.0.0.1:1/c.json", cache_path=tmp_path / "c.json")
    assert p.load() == DEFAULTS


def test_local_path_beats_url(server, tmp_path):
    f = tmp_path / "local.json"
    f.write_text(json.dumps({"version": 1, "stereo": {"max_disp": 64}}))
    assert ConfigProvider(url=server.url, path=str(f), cache_path=tmp_path / "c.json").load().max_disp == 64


def test_overrides_beat_the_config_source(tmp_path):
    f = tmp_path / "local.json"
    f.write_text(json.dumps(GOOD))
    cfg = ConfigProvider(path=str(f), overrides={"max_disp": 96}, cache_path=tmp_path / "c.json").load()
    assert cfg.max_disp == 96 and cfg.valid_iters == 2


def test_example_config_is_valid_and_matches_defaults():
    with open("config.example.json", encoding="utf-8") as f:
        values = parse_config_document(json.load(f), "config.example.json")
    assert {k: v for k, v in values.items() if not k.startswith("infer_")} == \
        {k: getattr(DEFAULTS, k) for k in ("backend", "max_disp", "valid_iters", "fp16")}


# ── CLI/env precedence (main.build_config_provider) ────────────────────────

def _args(**kw):
    base = dict(config_url=None, config_path=None, stereo_backend=None, stereo_override=[])
    return argparse.Namespace(**{**base, **kw})


def test_cli_beats_env_beats_file(tmp_path):
    f = tmp_path / "local.json"
    f.write_text(json.dumps(GOOD))
    env = {"DIBR_CONFIG_PATH": str(f), "DIBR_STEREO_OVERRIDES": "max_disp=96,valid_iters=8",
           "DIBR_STEREO_BACKEND": "sgbm"}
    cfg = build_config_provider(_args(stereo_override=["max_disp=64"]), env).load()
    assert (cfg.max_disp, cfg.valid_iters, cfg.backend, cfg.infer_width) == (64, 8, "sgbm", 640)
    cfg = build_config_provider(_args(stereo_backend="ffs"), env).load()
    assert cfg.backend == "ffs"


def test_bad_env_values_are_ignored_but_bad_cli_values_fail():
    env = {"DIBR_STEREO_OVERRIDES": "max_disp=7", "DIBR_CONFIG_URL": "http://example.com/c.json"}
    p = build_config_provider(_args(), env)
    assert p._url == _DEAD_URL  # bad env URL -> the default URL, not example.com
    assert p.load() == DEFAULTS
    with pytest.raises(ConfigError):
        build_config_provider(_args(stereo_override=["max_disp=7"]), {})
    with pytest.raises(ConfigError):
        build_config_provider(_args(config_url="http://example.com/c.json"), {})


# ── Built-in default URL and opt-out ───────────────────────────────────────

def test_real_default_url_is_the_project_gist_and_allowed():
    assert DEFAULT_CONFIG_URL.startswith("https://gist.githubusercontent.com/Nash-equilbrm/")
    check_url(DEFAULT_CONFIG_URL)


def test_default_url_is_used_when_nothing_is_configured(server, monkeypatch):
    monkeypatch.setattr(bridge_main, "DEFAULT_CONFIG_URL", server.url)
    cfg = build_config_provider(_args(), {}).load()
    assert cfg.max_disp == 128


def test_cli_and_env_url_and_path_beat_the_default(server, tmp_path):
    assert build_config_provider(_args(config_url=server.url), {})._url == server.url
    assert build_config_provider(_args(), {"DIBR_CONFIG_URL": server.url})._url == server.url
    f = tmp_path / "local.json"
    f.write_text(json.dumps({"version": 1, "stereo": {"max_disp": 64}}))
    p = build_config_provider(_args(), {"DIBR_CONFIG_PATH": str(f)})
    assert p._url is None and p.load().max_disp == 64


@pytest.mark.parametrize("cli,env", [
    ("off", None), ("", None), ("OFF", None),
    (None, "off"), (None, "none"), (None, ""),
])
def test_opt_out_disables_remote_and_ignores_cache(server, monkeypatch, cli, env, caplog):
    monkeypatch.setattr(bridge_main, "DEFAULT_CONFIG_URL", server.url)
    build_config_provider(_args(), {}).load()  # populate the last-good cache
    environ = {} if env is None else {"DIBR_CONFIG_URL": env}
    p = build_config_provider(_args(config_url=cli), environ)
    assert p._url is None
    caplog.set_level("INFO")
    assert p.load() == DEFAULTS  # built-in defaults only, cache ignored
    assert "remote config disabled" in caplog.text


def test_default_url_offline_falls_back_to_cache_then_defaults(server, monkeypatch, caplog):
    caplog.set_level("INFO")
    assert build_config_provider(_args(), {}).load() == DEFAULTS  # dead URL, no cache yet
    assert "built-in defaults (default URL)" in caplog.text
    monkeypatch.setattr(bridge_main, "DEFAULT_CONFIG_URL", server.url)
    build_config_provider(_args(), {}).load()  # online once -> cache
    monkeypatch.setattr(bridge_main, "DEFAULT_CONFIG_URL", _DEAD_URL)
    assert build_config_provider(_args(), {}).load().max_disp == 128
    assert "last-good cache" in caplog.text
