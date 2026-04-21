#!/usr/bin/env python3
"""
Gradio front-end for the Local Meeting Transcript Summarizer.

Thin skin over ``main.py``'s pipeline. See ``contexts/gradio_app.md`` for the
full spec, decisions, and milestone breakdown.

Run:
    uv run app.py

Then open http://localhost:7860 in a browser.

Status: M5 — Full pipeline run with progress + final output.
Sidebar settings (M3), upload + ingest + preview + speaker form (M4) all
unchanged. Run button now wired to ``run_pipeline_generator`` which streams
steps 2→5 end-to-end. On completion a rendered summary + Copy button +
Download button appear. On error or exception the button re-enables and an
inline error surfaces. Model unload happens in a finally block; full
cancellation handling comes in M6.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

import gradio as gr
import httpx
from dotenv import load_dotenv

from pipeline.step1_convert import convert
from pipeline.step2_cleanup import clean_transcript
from pipeline.step3_mapping import apply_speaker_mapping, detect_generic_speakers
from pipeline.step4_extraction import extract_information
from pipeline.step5_formatter import format_summary


# Load .env at module import so OLLAMA_HOST (and any other env-driven
# defaults below) are populated BEFORE the constants block evaluates.
# main() hard-fails if OLLAMA_HOST is still missing, matching main.py.
load_dotenv()


# ─── Defaults ─────────────────────────────────────────────────────────────

# No silent fallback to localhost — that hides misconfiguration. If
# OLLAMA_HOST isn't set in .env or the shell environment, this is "" and
# main() will refuse to launch.
DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "")
DEFAULT_EDITOR_MODEL = "gemma4:26b"
DEFAULT_EXTRACTOR_MODEL = "gemma4:26b"

SERVER_PORT = 7860
SERVER_HOST = "0.0.0.0"  # reachable on LAN / inside Docker; revisit at M9

# HTTP timeout for Ollama probes. Short because these are reachability checks,
# not model-generation calls (which happen inside the step functions with
# their own implicit timeouts via ollama.Client).
OLLAMA_PROBE_TIMEOUT = 3.0

# Upload cap. Enforced at demo.launch(max_file_size=...); gr.File itself has
# no per-component size kwarg (issue #7825).
UPLOAD_MAX_SIZE = "10mb"


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

# JS fired by the Copy button. Reads the rendered summary's innerText and
# puts it on the clipboard. Uses innerText (not innerHTML) so the user gets
# clean, human-readable text rather than HTML tags. The elem_id here MUST
# match the gr.Markdown elem_id below (``final-summary``).
COPY_SUMMARY_JS = """
() => {
    const el = document.getElementById('final-summary');
    if (!el) {
        console.warn('final-summary element not found');
        return;
    }
    const text = el.innerText || '';
    navigator.clipboard.writeText(text);
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

    Best-effort: swallows all exceptions. Called from the pipeline's
    finally block and (in M6) from session-cleanup paths.
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
    """Gate function for Run.

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

        {'tempdir_path':        str | None,
         'canonical_md':        str | None,  # path to step1 output
         'uploaded_stem':       str | None,  # original filename stem
         'final_summary_path':  str | None,  # path to the step5 output
         'models_used':         set[str],    # eject these on finally / M6 cleanup
         'ollama_host':         str,
         'run_in_progress':     bool}

    ``tempdir_path`` uses ``tempfile.mkdtemp()`` (string path) rather than a
    ``TemporaryDirectory`` object — the object carries a finalizer that
    doesn't play well with gr.State deepcopy semantics. Cleanup is explicit
    in M6 via ``shutil.rmtree(ignore_errors=True)``.

    Passed as ``gr.State(init_session_state())`` — the call produces a
    dict which Gradio then deepcopies per session. (Passing the callable
    itself, ``gr.State(init_session_state)``, does NOT work in Gradio 6.x:
    the function object is handed through to handlers verbatim instead of
    being invoked.)
    """
    return {
        "tempdir_path": None,
        "canonical_md": None,
        "uploaded_stem": None,
        "final_summary_path": None,
        "models_used": set(),
        "ollama_host": DEFAULT_OLLAMA_HOST,
        "run_in_progress": False,
    }


def _ensure_tempdir(state_val: dict[str, Any]) -> Path:
    """Lazily create the session tempdir on first upload. Returns the Path."""
    if not state_val.get("tempdir_path"):
        state_val["tempdir_path"] = tempfile.mkdtemp(prefix="meeting_summarizer_")
    return Path(state_val["tempdir_path"])


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


# Empty updates for the four run-output components, used by on_file_upload
# to clear stale results when a new file is uploaded or the upload is cleared.
_RUN_OUTPUT_RESET = (
    gr.update(value="", visible=False),   # run_status
    gr.update(value="", visible=False),   # final_summary_md
    gr.update(visible=False),             # copy_btn
    gr.update(value=None, visible=False), # download_btn
)


def on_file_upload(
    uploaded_file: str | None,
    state_val: dict[str, Any],
) -> tuple:
    """Handle file upload / clear.

    Runs step1_convert into the session tempdir, renders preview, detects
    generic speakers, enables Run if ingest succeeded.

    Returns an 11-tuple in this order to match the event listener's outputs:
        preview_md update,
        error_md update,
        run_btn update,
        detected_speakers_state,
        speaker_map_state (always reset to {} on new upload),
        session_state,
        meta_md update,
        run_status update,          (reset — stale after new upload)
        final_summary_md update,    (reset)
        copy_btn update,            (reset)
        download_btn update         (reset)

    Any exception from step1 — including ``SystemExit`` raised on format-
    detection failures — is caught explicitly. A bare ``except Exception``
    would miss ``SystemExit`` (it's a ``BaseException`` subclass) and kill
    the worker.
    """
    # User cleared the file — reset everything to the pre-upload state.
    if uploaded_file is None:
        state_val["canonical_md"] = None
        state_val["uploaded_stem"] = None
        state_val["final_summary_path"] = None
        return (
            gr.update(value="", visible=False),   # preview
            gr.update(value="", visible=False),   # error
            gr.update(interactive=False),         # run button
            [],                                   # detected_speakers_state
            {},                                   # speaker_map_state
            state_val,                            # session_state
            gr.update(value="", visible=False),   # meta line
            *_RUN_OUTPUT_RESET,                   # clear stale run results
        )

    src = Path(uploaded_file)

    try:
        tempdir = _ensure_tempdir(state_val)
        raw_dir = tempdir / "raw_files"
        raw_dir.mkdir(parents=True, exist_ok=True)
        # Copy upload into session-owned tempdir so step1 writes its outputs
        # alongside and everything for this session lives under one root.
        dst = raw_dir / src.name
        shutil.copy(src, dst)
        _, md_path = convert(dst, raw_dir)
    except BaseException as e:
        # SystemExit is raised by step1 on unrecognized format / empty parse;
        # catch it here so Gradio's worker keeps running.
        err_type = type(e).__name__
        err_msg = str(e) or err_type
        state_val["canonical_md"] = None
        state_val["uploaded_stem"] = None
        state_val["final_summary_path"] = None
        return (
            gr.update(value="", visible=False),
            gr.update(
                value=f"❌ Could not ingest `{src.name}`: {err_msg}",
                visible=True,
            ),
            gr.update(interactive=False),
            [],
            {},
            state_val,
            gr.update(value="", visible=False),
            *_RUN_OUTPUT_RESET,
        )

    canonical_md = md_path.read_text(encoding="utf-8")
    state_val["canonical_md"] = str(md_path)
    state_val["uploaded_stem"] = src.stem
    state_val["final_summary_path"] = None  # prior runs' output is now stale

    speakers = detect_generic_speakers(canonical_md)
    # Line count excluding headings/blanks gives a rough turn count for the
    # meta display; exact turn count lives in the .json sidecar emitted by
    # step1, but this is cheaper than re-reading it.
    n_turns = sum(
        1 for line in canonical_md.splitlines() if line.startswith("**")
    )
    meta_bits = [f"{n_turns} turns ingested"]
    if speakers:
        meta_bits.append(f"{len(speakers)} generic speakers detected")
    else:
        meta_bits.append("no generic speakers — Run enabled")
    meta_line = " · ".join(meta_bits)

    return (
        gr.update(value=canonical_md, visible=True),
        gr.update(value="", visible=False),
        gr.update(interactive=True),
        speakers,
        {},  # reset speaker_map for the new upload
        state_val,
        gr.update(value=f"<sub>{meta_line}</sub>", visible=True),
        *_RUN_OUTPUT_RESET,
    )


# ─── Pipeline orchestration ──────────────────────────────────────────────

def run_pipeline_generator(
    state_val: dict[str, Any],
    editor_model: str,
    extractor_model: str,
    ollama_host: str,
    speaker_map: dict[str, str],
    progress: gr.Progress = gr.Progress(),
) -> Iterator[tuple]:
    """Run steps 2 → 5 on the session's ingested transcript.

    Generator: yields a 6-tuple at each phase boundary matching the Run
    button's outputs list (run_status, final_summary_md, copy_btn,
    download_btn, run_btn, session_state).

    On pre-flight failure, yields once with an error and stops — no
    try/finally entered, so no models are touched.

    On any step raising (``SystemExit`` from the step modules' ``sys.exit(1)``
    paths, or any ``Exception``), yields an error and cleans up models in
    ``finally``. ``KeyboardInterrupt`` and ``GeneratorExit`` are left to
    propagate so the user / Gradio can cancel cleanly.

    Non-streaming Ollama calls mean cancellation during a step is blocked
    until the step returns — accepted trade-off per spec F3 / Risks.
    """
    # ── Pre-flight ────────────────────────────────────────────────────────
    if not state_val.get("canonical_md"):
        yield (
            gr.update(
                value="❌ No transcript ingested. Upload a file first.",
                visible=True,
            ),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(interactive=True),
            state_val,
        )
        return

    ok, msg = preflight_check(ollama_host, editor_model, extractor_model)
    if not ok:
        yield (
            gr.update(value=f"❌ Pre-flight failed: {msg}", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(interactive=True),
            state_val,
        )
        return

    tempdir = Path(state_val["tempdir_path"])
    canonical_md_path = Path(state_val["canonical_md"])

    # Track both models eagerly so the finally block can eject them even if
    # a step raises before touching Ollama. ``unload_model`` is best-effort,
    # so extra calls on non-loaded models are harmless.
    state_val["models_used"].update({editor_model, extractor_model})
    state_val["run_in_progress"] = True

    # Stable yields for "no change on this output" — keeps the 6-tuple shape
    # consistent without re-clobbering fields we don't want to touch.
    NOOP_FINAL = gr.update()
    NOOP_COPY = gr.update()
    NOOP_DOWNLOAD = gr.update()

    try:
        # Run button disabled during the run; re-enabled in finally via the
        # terminal yields (success or error).
        progress(0.0, desc="Cleaning transcript…")
        yield (
            gr.update(value="**Step 1/4** · Cleaning transcript…", visible=True),
            NOOP_FINAL, NOOP_COPY, NOOP_DOWNLOAD,
            gr.update(interactive=False),
            state_val,
        )
        cleaned_path = clean_transcript(
            canonical_md_path,
            tempdir / "cleaned",
            editor_model,
            ollama_host,
        )

        # Step 3: pure, in-memory speaker mapping. No Ollama call.
        progress(0.25, desc="Applying speaker names…")
        yield (
            gr.update(value="**Step 2/4** · Applying speaker names…", visible=True),
            NOOP_FINAL, NOOP_COPY, NOOP_DOWNLOAD,
            gr.update(interactive=False),
            state_val,
        )
        cleaned_text = cleaned_path.read_text(encoding="utf-8")
        named_text = apply_speaker_mapping(cleaned_text, speaker_map or {})
        named_dir = tempdir / "named"
        named_dir.mkdir(parents=True, exist_ok=True)
        # Mirror the CLI's stem convention so step4's strip of "_named" works
        # out to the original uploaded_stem.
        named_stem = cleaned_path.stem.replace("_cleaned", "_named")
        named_path = named_dir / f"{named_stem}.md"
        named_path.write_text(named_text, encoding="utf-8")

        progress(0.50, desc="Extracting information…")
        yield (
            gr.update(value="**Step 3/4** · Extracting information…", visible=True),
            NOOP_FINAL, NOOP_COPY, NOOP_DOWNLOAD,
            gr.update(interactive=False),
            state_val,
        )
        extracted_path = extract_information(
            named_path,
            tempdir / "extracted",
            extractor_model,
            ollama_host,
        )

        progress(0.75, desc="Formatting summary…")
        yield (
            gr.update(value="**Step 4/4** · Formatting final summary…", visible=True),
            NOOP_FINAL, NOOP_COPY, NOOP_DOWNLOAD,
            gr.update(interactive=False),
            state_val,
        )
        final_path = format_summary(
            extracted_path,
            tempdir / "final",
            editor_model,
            ollama_host,
        )

        # Rename the summary file to something user-friendly for download,
        # based on the original upload's stem. The internal file in /final
        # lives alongside; we copy rather than rename so any intermediate
        # debugging artifacts stay intact.
        stem = state_val.get("uploaded_stem") or final_path.stem
        download_dir = tempdir / "download"
        download_dir.mkdir(parents=True, exist_ok=True)
        download_path = download_dir / f"{stem}_summary.md"
        shutil.copy(final_path, download_path)
        state_val["final_summary_path"] = str(download_path)

        final_content = download_path.read_text(encoding="utf-8")

        progress(1.0, desc="Done")
        yield (
            gr.update(value="✅ Summary ready.", visible=True),
            gr.update(value=final_content, visible=True),
            gr.update(visible=True),
            gr.update(value=str(download_path), visible=True),
            gr.update(interactive=True),
            state_val,
        )

    except (Exception, SystemExit) as e:
        # Excludes KeyboardInterrupt and GeneratorExit — those must propagate
        # so cancellation works cleanly (M6 wires it up in detail).
        err_type = type(e).__name__
        err_msg = str(e) or err_type
        yield (
            gr.update(
                value=f"❌ Pipeline failed at runtime: {err_type}: {err_msg}",
                visible=True,
            ),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(interactive=True),
            state_val,
        )

    finally:
        # Eject models regardless of outcome. Important on success too —
        # without this, ``keep_alive=-1`` from inside the step functions
        # leaves both models hot on ziggie until another run or timeout.
        state_val["run_in_progress"] = False
        for m in list(state_val.get("models_used", ())):
            unload_model(ollama_host, m)


# ─── UI construction ──────────────────────────────────────────────────────

def build_demo() -> gr.Blocks:
    """Build the Gradio Blocks app."""
    with gr.Blocks(title="Local Meeting Summarizer") as demo:
        # Per-session state. ``init_session_state()`` is called here to
        # produce the initial dict; Gradio deepcopies it per session so each
        # browser tab gets its own independent copy. Do NOT pass the bare
        # callable — see ``init_session_state`` docstring for why.
        session_state = gr.State(init_session_state())

        # Drives the @gr.render-decorated speaker form. List of 'Speaker N'
        # tags detected in the latest upload; empty list = no form shown.
        detected_speakers_state = gr.State([])

        # Written by each speaker-name textbox on change. Read by the Run
        # handler and passed to apply_speaker_mapping().
        speaker_map_state = gr.State({})

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
                "Upload a `.rtf` from Moonshine or `.md` from the local "
                "transcriber. Files above 10 MB are rejected; ~2.5h of speech "
                "is the practical ceiling (LLM context window)."
            )

            # ── Upload ────────────────────────────────────────────────────
            upload = gr.File(
                label="Transcript",
                file_types=[".rtf", ".md"],
                file_count="single",
                elem_id="transcript-upload",
            )

            # Inline error message (e.g. "Could not detect transcript format").
            # Hidden until ingest fails.
            error_md = gr.Markdown("", visible=False, elem_id="ingest-error")

            # One-line status under the upload — turn count + speaker summary.
            meta_md = gr.Markdown("", visible=False, elem_id="ingest-meta")

            # Canonical markdown preview. Scrollable. (No built-in copy
            # button on gr.Markdown in this Gradio version; users can still
            # select-and-copy. The final summary has a dedicated Copy button.)
            preview_md = gr.Markdown(
                "",
                label="Preview",
                max_height=400,
                visible=False,
                elem_id="transcript-preview",
            )

            # ── Dynamic speaker form ─────────────────────────────────────
            # @gr.render rebuilds its children each time its inputs change.
            # When detected_speakers_state is an empty list, the function
            # returns early and nothing renders — that's the "no form" case
            # from spec scenario #3 (fully pre-named transcripts).
            @gr.render(inputs=[detected_speakers_state])
            def render_speaker_form(speakers: list[str]):
                if not speakers:
                    return
                gr.Markdown(
                    "### Speaker names\n"
                    "Enter a real name for each detected speaker, or leave a "
                    "field blank to keep the original `Speaker N` label."
                )
                for tag in speakers:
                    tb = gr.Textbox(
                        label=tag,
                        placeholder="Leave blank to keep original label",
                    )

                    # Closure captures the tag; without this, every textbox
                    # would write to the same key (late-binding gotcha).
                    def _make_updater(captured_tag: str):
                        def _update(new_val: str, current_map: dict[str, str]):
                            new_map = dict(current_map or {})
                            if new_val and new_val.strip():
                                new_map[captured_tag] = new_val.strip()
                            else:
                                new_map.pop(captured_tag, None)
                            return new_map
                        return _update

                    tb.change(
                        _make_updater(tag),
                        inputs=[tb, speaker_map_state],
                        outputs=[speaker_map_state],
                    )

            run_btn = gr.Button(
                "Run",
                variant="primary",
                interactive=False,
                elem_id="run-btn",
            )

            # ── Run-time output section ─────────────────────────────────
            # All hidden until the pipeline makes progress. Status ticks
            # through the 4 phases; on success the rendered summary, Copy,
            # and Download reveal together.
            run_status = gr.Markdown("", visible=False, elem_id="run-status")

            final_summary_md = gr.Markdown(
                "",
                label="Meeting summary",
                visible=False,
                elem_id="final-summary",  # matched by COPY_SUMMARY_JS
            )

            with gr.Row():
                copy_btn = gr.Button(
                    "Copy summary",
                    variant="secondary",
                    visible=False,
                    elem_id="copy-btn",
                )
                download_btn = gr.DownloadButton(
                    "Download .md",
                    visible=False,
                    elem_id="download-btn",
                )

        gr.Markdown(
            "<sub>Each tab runs independently — multiple tabs = multiple "
            "queue slots. Tab close ejects any models this session loaded.</sub>"
        )

        # ── Event wiring ──────────────────────────────────────────────────

        # Page load: check reachability + validate default models + force dark.
        demo.load(
            on_startup,
            inputs=[session_state],
            outputs=[banner, editor_status, extractor_status],
        )
        demo.load(fn=None, inputs=None, outputs=None, js=FORCE_DARK_MODE_JS)

        # Host change: update session state, refresh banner, re-validate models.
        ollama_host.change(
            on_host_change,
            inputs=[ollama_host, editor_model, extractor_model, session_state],
            outputs=[session_state, banner, editor_status, extractor_status],
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

        # File upload: ingest, render preview, detect speakers, enable Run.
        # Also clears any stale run-result components from a previous run.
        upload.change(
            on_file_upload,
            inputs=[upload, session_state],
            outputs=[
                preview_md,
                error_md,
                run_btn,
                detected_speakers_state,
                speaker_map_state,
                session_state,
                meta_md,
                run_status,
                final_summary_md,
                copy_btn,
                download_btn,
            ],
        )

        # Run button → streams through steps 2–5.
        run_btn.click(
            run_pipeline_generator,
            inputs=[
                session_state,
                editor_model,
                extractor_model,
                ollama_host,
                speaker_map_state,
            ],
            outputs=[
                run_status,
                final_summary_md,
                copy_btn,
                download_btn,
                run_btn,
                session_state,
            ],
        )

        # Copy button is pure JS — reads the rendered summary's innerText
        # from the DOM and writes it to the clipboard. No Python round-trip.
        copy_btn.click(
            None,
            inputs=None,
            outputs=None,
            js=COPY_SUMMARY_JS,
        )

    return demo


# ─── Entry point ──────────────────────────────────────────────────────────

def main() -> None:
    # load_dotenv() already ran at module import, so DEFAULT_OLLAMA_HOST
    # reflects .env by now. Refuse to launch if it's still unset — matches
    # main.py's stance that running without an explicit host config is
    # user error, not something to paper over with a localhost default.
    if not DEFAULT_OLLAMA_HOST:
        print(
            "Error: OLLAMA_HOST is missing. Please define it in a .env file.",
            file=sys.stderr,
        )
        sys.exit(1)
    demo = build_demo()
    demo.launch(
        server_name=SERVER_HOST,
        server_port=SERVER_PORT,
        inbrowser=True,
        theme=gr.Theme.from_hub("Nymbo/Nymbo_Theme"),
        max_file_size=UPLOAD_MAX_SIZE,
    )


if __name__ == "__main__":
    main()
