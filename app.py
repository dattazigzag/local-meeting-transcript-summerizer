#!/usr/bin/env python3
"""
Gradio front-end for the Local Meeting Transcript Summarizer.

Thin skin over ``main.py``'s pipeline. See ``contexts/gradio_app.md`` for the
full spec, decisions, and milestone breakdown.

Run:
    uv run app.py

Then open http://localhost:7860 in a browser.

Status: M3 — Ollama connection + model validation. Sidebar settings are live:
host reachability check on change, reactive model validation (✓ / ✗), explicit
Test-connection button, and a startup banner when Ollama is unreachable. Per-
session state isolated via an ``init_session_state`` factory. No upload and no
pipeline yet — those land in M4 and M5.
"""

from __future__ import annotations

import os
from typing import Any

import gradio as gr
import httpx
from dotenv import load_dotenv


# ─── Defaults ─────────────────────────────────────────────────────────────

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_EDITOR_MODEL = "gemma4:26b"
DEFAULT_EXTRACTOR_MODEL = "qwen3.5:27b"

SERVER_PORT = 7860
SERVER_HOST = "0.0.0.0"  # reachable on LAN / inside Docker; revisit at M9

# HTTP timeout for Ollama probes. Short because these are reachability checks,
# not model-generation calls (which happen in M5 with their own timeouts).
OLLAMA_PROBE_TIMEOUT = 3.0


# ─── Dark-mode forcing ────────────────────────────────────────────────────
# Gradio 6 respects the `?__theme=dark` URL param, but sending the user to
# the raw URL is awkward. Injecting a one-line JS snippet on page load adds
# the `dark` class to <body>, which the Monochrome theme then honors.
FORCE_DARK_MODE_JS = """
() => {
    if (!document.body.classList.contains('dark')) {
        document.body.classList.add('dark');
    }
}
"""


# ─── Ollama helpers (pure I/O; no Gradio imports) ────────────────────────

def test_ollama_connection(host: str) -> tuple[bool, str]:
    """Ping ``GET {host}/api/tags`` with a short timeout.

    Returns (success, user-facing message). Best-effort; never raises.
    """
    if not host or not host.strip():
        return False, "Host is empty."
    url = f"{host.rstrip('/')}/api/tags"
    try:
        r = httpx.get(url, timeout=OLLAMA_PROBE_TIMEOUT)
        r.raise_for_status()
        return True, f"✓ Connected to {host}"
    except httpx.ConnectError:
        return False, f"✗ Cannot reach {host} (connection refused)"
    except httpx.TimeoutException:
        return False, f"✗ Cannot reach {host} (timed out after {OLLAMA_PROBE_TIMEOUT}s)"
    except httpx.HTTPStatusError as e:
        return False, f"✗ {host} responded with HTTP {e.response.status_code}"
    except Exception as e:  # pragma: no cover — catch-all for exotic failures
        return False, f"✗ {host}: {type(e).__name__}: {e}"


def list_available_models(host: str) -> list[str]:
    """Return list of model names currently pulled on Ollama.

    Returns [] on any connection failure. Never raises.
    """
    if not host or not host.strip():
        return []
    url = f"{host.rstrip('/')}/api/tags"
    try:
        r = httpx.get(url, timeout=OLLAMA_PROBE_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception:
        return []


def validate_model_available(host: str, model: str) -> bool:
    """True iff ``model`` is in Ollama's pulled list. Does NOT attempt pull."""
    if not model or not host:
        return False
    return model in list_available_models(host)


def unload_model(host: str, model: str) -> None:
    """Send ``keep_alive=0`` to Ollama to evict a model.

    Best-effort: swallows all exceptions. Called from M6 cancellation paths;
    defined here in M3 so the full Ollama surface lives in one place.
    """
    if not host or not model:
        return
    url = f"{host.rstrip('/')}/api/generate"
    try:
        httpx.post(
            url,
            json={"model": model, "keep_alive": 0, "prompt": ""},
            timeout=OLLAMA_PROBE_TIMEOUT,
        )
    except Exception:
        pass


def preflight_check(
    host: str, editor_model: str, extractor_model: str
) -> tuple[bool, str]:
    """Gate function for Run (wired up in M5). Defined here as a stub.

    Verifies Ollama is reachable AND both required models are pulled.
    Returns (ready, user-facing message).
    """
    ok, msg = test_ollama_connection(host)
    if not ok:
        return False, msg
    models = set(list_available_models(host))
    missing = [m for m in (editor_model, extractor_model) if m and m not in models]
    if missing:
        return False, f"✗ Models not pulled on {host}: {', '.join(missing)}"
    return True, "✓ Ready"


# ─── Session state factory ───────────────────────────────────────────────

def init_session_state() -> dict[str, Any]:
    """Factory for ``gr.State``. Returns a fresh dict per session.

    Shape::

        {'tempdir': TemporaryDirectory | None,
         'models_used': set[str],
         'ollama_host': str,
         'run_in_progress': bool}

    ``tempdir`` stays ``None`` until M4 when uploads start.
    ``models_used`` collects model names touched during a session so M6 can
    eject them on cleanup.

    Passed to ``gr.State(init_session_state)`` — Gradio calls the factory
    per session, which prevents the known cross-session leak where a mutable
    default would be shared across browser tabs.
    """
    return {
        "tempdir": None,
        "models_used": set(),
        "ollama_host": DEFAULT_OLLAMA_HOST,
        "run_in_progress": False,
    }


# ─── UI event handlers ───────────────────────────────────────────────────

def _banner_update_for_host(host: str) -> dict:
    """Build a ``gr.update`` for the unreachable-Ollama banner."""
    ok, _ = test_ollama_connection(host)
    if ok:
        return gr.update(value="", visible=False)
    return gr.update(
        value=(
            f"⚠ Cannot reach Ollama at `{host}`. "
            "Update the host in the sidebar and click **Test connection**."
        ),
        visible=True,
    )


def _model_indicator(host: str, model: str) -> str:
    """Return small markdown indicating model status. Empty string when no
    model typed; '—' when we can't tell (host unreachable or no models pulled)
    so the user doesn't see a false ✗.
    """
    if not model:
        return ""
    if not host:
        return "—"
    models = list_available_models(host)
    if not models:
        return "— _(host unreachable or no models pulled)_"
    if model in models:
        return "✓ available"
    return "✗ not pulled"


def on_startup(state_val: dict[str, Any]) -> tuple[dict, str, str]:
    """``demo.load`` handler. Checks reachability + validates default models."""
    host = state_val.get("ollama_host", DEFAULT_OLLAMA_HOST)
    banner = _banner_update_for_host(host)
    editor_ind = _model_indicator(host, DEFAULT_EDITOR_MODEL)
    extractor_ind = _model_indicator(host, DEFAULT_EXTRACTOR_MODEL)
    return banner, editor_ind, extractor_ind


def on_host_change(
    host: str, editor: str, extractor: str, state_val: dict[str, Any]
) -> tuple[dict, dict, str, str]:
    """When host textbox changes: update session state, refresh banner,
    re-validate both model fields (they depend on host reachability)."""
    state_val["ollama_host"] = host
    banner = _banner_update_for_host(host)
    editor_ind = _model_indicator(host, editor)
    extractor_ind = _model_indicator(host, extractor)
    return state_val, banner, editor_ind, extractor_ind


def on_test_connection(host: str) -> tuple[str, dict]:
    """Explicit 'Test connection' button click.

    Returns (status-line markdown, banner update). Success hides the banner,
    failure shows it so the user sees the same message in both places.
    """
    ok, msg = test_ollama_connection(host)
    if ok:
        return msg, gr.update(value="", visible=False)
    return msg, gr.update(
        value=(
            f"⚠ Cannot reach Ollama at `{host}`. "
            "Update the host in the sidebar and click **Test connection**."
        ),
        visible=True,
    )


# ─── UI construction ──────────────────────────────────────────────────────

def build_demo() -> gr.Blocks:
    """Build the Gradio Blocks app."""
    theme = gr.themes.Monochrome()  # squarish by default (radius_none)

    with gr.Blocks(title="Local Meeting Summarizer", theme=theme) as demo:
        # Per-session state — factory callable, NOT a mutable default. This
        # avoids the known cross-session leak where two tabs would share the
        # same dict object.
        state = gr.State(init_session_state)

        with gr.Sidebar():
            gr.Markdown("### Settings")

            ollama_host = gr.Textbox(
                label="Ollama host",
                value=DEFAULT_OLLAMA_HOST,
                placeholder="http://<host>:<port>",
                info="Where your Ollama server is reachable.",
            )

            editor_model = gr.Textbox(
                label="Editor model (steps 2 & 5)",
                value=DEFAULT_EDITOR_MODEL,
                info="Used for cleanup + final formatting.",
            )
            editor_status = gr.Markdown("", elem_id="editor-status")

            extractor_model = gr.Textbox(
                label="Extractor model (step 4)",
                value=DEFAULT_EXTRACTOR_MODEL,
                info="Used for information extraction.",
            )
            extractor_status = gr.Markdown("", elem_id="extractor-status")

            test_btn = gr.Button("Test connection", variant="secondary")
            test_status = gr.Markdown("", elem_id="test-status")

        with gr.Column():
            # Startup banner: shown only when Ollama is unreachable.
            banner = gr.Markdown("", visible=False, elem_id="ollama-banner")

            gr.Markdown("# Local Meeting Summarizer")
            gr.Markdown(
                "Upload a transcript to begin. Supports `.rtf` from Moonshine "
                "and `.md` from the local transcriber."
            )
            gr.Markdown(
                "_(Upload component comes online in milestone M4.)_"
            )

        gr.Markdown(
            "<sub>Each tab runs independently — multiple tabs = multiple "
            "queue slots. Tab close ejects any models this session loaded.</sub>"
        )

        # ── Event wiring ──────────────────────────────────────────────────

        # Page load: check reachability + validate default models + force dark.
        demo.load(
            on_startup,
            inputs=[state],
            outputs=[banner, editor_status, extractor_status],
        )
        demo.load(fn=None, inputs=None, outputs=None, js=FORCE_DARK_MODE_JS)

        # Host change: update session state, refresh banner, re-validate models.
        ollama_host.change(
            on_host_change,
            inputs=[ollama_host, editor_model, extractor_model, state],
            outputs=[state, banner, editor_status, extractor_status],
        )

        # Model textbox changes: validate only that one.
        editor_model.change(
            _model_indicator,
            inputs=[ollama_host, editor_model],
            outputs=[editor_status],
        )
        extractor_model.change(
            _model_indicator,
            inputs=[ollama_host, extractor_model],
            outputs=[extractor_status],
        )

        # Explicit Test connection button — mirrors banner state.
        test_btn.click(
            on_test_connection,
            inputs=[ollama_host],
            outputs=[test_status, banner],
        )

    return demo


# ─── Entry point ──────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()
    demo = build_demo()
    demo.launch(
        server_name=SERVER_HOST,
        server_port=SERVER_PORT,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
