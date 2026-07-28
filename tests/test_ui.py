"""The interface, as far as it can be tested without a browser.

The §9 anti-goals are acceptance criteria, so some of them are asserted here:
no CDN dependencies (the stack runs with no outbound access), no rounded
corners or shadows in the stylesheet, and gold reserved for live state.
"""

from __future__ import annotations

import re
from dataclasses import replace as dataclass_replace
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="install the api extras to test the interface")

from fastapi.testclient import TestClient  # noqa: E402

from nawat.api import create_app, storage_warnings  # noqa: E402

from .conftest import FakeBackend  # noqa: E402

CACHE_CEILING = 50 * 10**6

UI_DIR = Path(__file__).resolve().parent.parent / "nawat" / "ui"


@pytest.fixture
def client(cache, free_port):
    cache.config = dataclass_replace(cache.config, serve_port=free_port)
    app = create_app(cache=cache, backend=FakeBackend(), start_queue=False, sweep_idle=False)
    with TestClient(app) as test_client:
        yield test_client


def test_the_root_serves_the_instrument(client):
    landing = client.get("/", follow_redirects=True)
    assert landing.status_code == 200
    assert "NAWĀT" in landing.text
    assert 'id="strip"' in landing.text, "the status strip is part of the frame"


def test_the_assets_are_served(client):
    for asset in ("style.css", "app.js", "fonts/fonts.css"):
        assert client.get(f"/ui/{asset}").status_code == 200, asset


def test_the_interface_reaches_no_external_host():
    """NFR-3.4: the stack operates with no outbound internet access."""
    external = re.compile(r"https?://(?!localhost|127\.0\.0\.1)[\w.-]+\.[a-z]{2,}", re.IGNORECASE)
    for path in UI_DIR.rglob("*"):
        if path.suffix not in (".html", ".css", ".js"):
            continue
        for match in external.finditer(path.read_text()):
            # SVG/XML namespaces are identifiers, not requests.
            if "www.w3.org" in match.group(0):
                continue
            pytest.fail(f"{path.name} references {match.group(0)}")


def test_the_fonts_are_vendored_not_fetched():
    css = (UI_DIR / "fonts" / "fonts.css").read_text()
    assert "gstatic" not in css and "googleapis" not in css
    woff2 = list((UI_DIR / "fonts").glob("*.woff2"))
    assert len(woff2) >= 6, "IBM Plex Sans, Mono, Condensed and Arabic faces must ship with the UI"
    for name in re.findall(r"url\(([^)]+\.woff2)\)", css):
        assert (UI_DIR / "fonts" / name).exists(), name


def test_the_stylesheet_honours_the_anti_goals():
    css = (UI_DIR / "style.css").read_text()
    assert "box-shadow" not in css, "no drop shadows"
    assert "linear-gradient" not in css and "radial-gradient" not in css, "no gradients"
    for radius in re.findall(r"border-radius:\s*([^;]+);", css):
        assert radius.strip().startswith("0"), f"no corner radii, found {radius}"
    for token, value in (("--ground", "#0F1620"), ("--gold", "#C9A227"),
                         ("--rubric", "#C2492E"), ("--verdigris", "#4E8C86")):
        assert f"{token}:" in css and value in css, f"palette token {token} missing"
    assert "prefers-reduced-motion" in css


def test_gold_is_reserved_for_live_state():
    """The reservation rule: gold appears only where something is live."""
    css = (UI_DIR / "style.css").read_text()
    gold_lines = [line.strip() for line in css.splitlines() if "--gold)" in line]
    for line in gold_lines:
        assert "live" in line or "running" in line or "staging" in line or "publishing" in line \
            or "starting" in line, f"gold spent on something not live: {line}"
    js = (UI_DIR / "app.js").read_text()
    assert "#C9A227" in js, "the live trace draws in gold"


def test_health_carries_the_strip_readings(client):
    health = client.get("/health").json()
    assert "gpu" in health
    assert "warnings" in health and isinstance(health["warnings"], list)
    assert "cache" in health and "fraction" in health["cache"]


def test_storage_pressure_warnings_name_the_condition(cache, cached):
    cached("runs/x/adapter", 45 * 10**6, replicated=False)
    warnings = storage_warnings(cache.status())
    assert any("% of its ceiling" in w for w in warnings)
    assert any("only on this disk" in w for w in warnings)


def test_the_ui_needs_no_token_but_the_api_it_calls_does(cache, free_port):
    cache.config = dataclass_replace(cache.config, serve_port=free_port, api_token="secret-token")
    app = create_app(cache=cache, backend=FakeBackend(), start_queue=False, sweep_idle=False)
    with TestClient(app) as client:
        assert client.get("/", follow_redirects=True).status_code == 200
        assert client.get("/cache").status_code == 401
        assert "gate" in client.get("/ui/index.html").text, "the UI carries its own token gate"
