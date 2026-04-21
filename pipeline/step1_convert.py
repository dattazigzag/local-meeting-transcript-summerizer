#!/usr/bin/env python3
"""
Step 1: Ingest + Normalize a meeting transcript.

Dispatches by file suffix:
  - .rtf  ->  Moonshine.ai transcript (via striprtf)
  - .md   ->  Our local transcriber's export (M2, not yet implemented)

Both paths emit the same canonical markdown shape:

    # Transcript: <source_filename>

    **Speaker N:** merged text for this turn...

    **Speaker M:** merged text for this turn...

Plus a JSON sidecar with turn stats.

Usage:
    uv run python -m pipeline.step1_convert input.rtf [--out-dir .]
    uv run python -m pipeline.step1_convert input.md  [--out-dir .]   # M2+

Dependencies:
    striprtf  (pure-python, no native deps)

Output:
    <stem>.json  — structured turns, suitable for programmatic use
    <stem>.md    — human-readable, optimized for LLM summarization input
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from striprtf.striprtf import rtf_to_text

SPEAKER_RE = re.compile(r"^\s*(Speaker\s+\d+)\s*:\s*$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Shared parsing/emission helpers (format-agnostic)
# ---------------------------------------------------------------------------


def parse_turns(plain_text: str) -> list[dict]:
    """Group consecutive lines under each `Speaker N:` header into one turn."""
    turns: list[dict] = []
    current_speaker: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current_speaker and buffer:
            text = " ".join(s.strip() for s in buffer if s.strip())
            if text:
                turns.append(
                    {
                        "index": len(turns),
                        "speaker": current_speaker,
                        "text": text,
                    }
                )

    for raw in plain_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = SPEAKER_RE.match(line)
        if m:
            flush()
            current_speaker = m.group(1).title()  # normalize "speaker 1" -> "Speaker 1"
            buffer = []
        else:
            buffer.append(line)
    flush()
    return turns


def build_markdown(turns: list[dict], source_name: str) -> str:
    lines = [f"# Transcript: {source_name}", ""]
    for t in turns:
        lines.append(f"**{t['speaker']}:** {t['text']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_json(turns: list[dict], source_name: str) -> dict:
    speaker_counts = Counter(t["speaker"] for t in turns)
    word_count = sum(len(t["text"].split()) for t in turns)
    return {
        "source": source_name,
        "n_turns": len(turns),
        "speakers": sorted(
            speaker_counts.keys(),
            key=lambda s: int(s.split()[-1]) if s.split()[-1].isdigit() else 0,
        ),
        "turns_per_speaker": dict(speaker_counts),
        "word_count": word_count,
        "turns": turns,
    }


def _write_outputs(
    turns: list[dict], source_name: str, out_dir: Path, stem: str
) -> tuple[Path, Path]:
    """Shared writer: emit <stem>.json + <stem>.md into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"

    json_path.write_text(
        json.dumps(build_json(turns, source_name), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(build_markdown(turns, source_name), encoding="utf-8")
    return json_path, md_path


# ---------------------------------------------------------------------------
# Format-specific ingesters
# ---------------------------------------------------------------------------


def _ingest_moonshine_rtf(rtf_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """Parse a Moonshine.ai RTF transcript into canonical markdown + JSON."""
    rtf_text = rtf_path.read_text(encoding="utf-8", errors="replace")
    plain = rtf_to_text(rtf_text, errors="ignore")
    turns = parse_turns(plain)
    if not turns:
        raise SystemExit(
            f"No speaker turns found in {rtf_path}. Is this really a moonshine RTF?"
        )
    return _write_outputs(turns, rtf_path.name, out_dir, rtf_path.stem)


def _ingest_markdown(md_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """Parse a .md transcript into canonical markdown + JSON.

    Handles two shapes:
      1. Our transcriber's H3-heading style (primary M2 target).
      2. Already-canonical '**Speaker N:** text' style (re-ingest passthrough).
    """
    content = md_path.read_text(encoding="utf-8")

    if _looks_like_transcriber(content):
        turns = _parse_transcriber_turns(content)
    elif _looks_like_canonical(content):
        turns = _parse_canonical_md_turns(content)
    else:
        raise SystemExit(
            f"Could not detect transcript format in {md_path}. "
            "Expected either transcriber-style '### SPEAKER_XX [...]' headings "
            "or canonical '**Speaker N:**' inline tags."
        )

    if not turns:
        raise SystemExit(
            f"No speaker turns found in {md_path}. "
            "File was detected as markdown but no parseable content was extracted."
        )

    return _write_outputs(turns, md_path.name, out_dir, md_path.stem)


# ---------------------------------------------------------------------------
# Markdown parsing helpers (transcriber format + canonical passthrough)
# ---------------------------------------------------------------------------

# Transcriber H3 heading shape:
#   ### SPEAKER_02 [00:00 → 00:51] EN     (speaker ID form)
#   ### Amanda [00:00 → 00:51] EN         (if transcriber already renamed)
#   ### SPEAKER_01                        (bare, no timing/lang)
TRANSCRIBER_H3_RE = re.compile(r"^###\s+(.+?)\s*$")

# Canonical inline form (our step1 output; re-ingest case):
#   **Speaker 1:** body text
#   **Amanda:**   body text
CANONICAL_INLINE_RE = re.compile(r"^\*\*([^*]+?):\*\*\s*(.*)$")

# Speaker-ID shape: SPEAKER_00, SPEAKER_1, speaker 02, etc.
SPEAKER_ID_RE = re.compile(r"^SPEAKER[_\s]?(\d+)$", re.IGNORECASE)


def _extract_speaker_from_heading(body: str) -> str:
    """Pull the speaker identifier from an H3 heading body.

    Strips an optional [timestamp] block and an optional trailing 2-3 letter
    uppercase language token.

        'SPEAKER_02 [00:00 → 00:51] EN' -> 'SPEAKER_02'
        'Mary Jane [01:00 → 02:00] EN'  -> 'Mary Jane'
        'SPEAKER_01'                    -> 'SPEAKER_01'
    """
    without_bracket = re.sub(r"\s*\[[^\]]*\]\s*", " ", body).strip()
    without_lang = re.sub(r"\s+[A-Z]{2,3}\s*$", "", without_bracket).strip()
    return without_lang or without_bracket


def _normalize_speaker_tag(raw: str) -> str:
    """Apply D1 policy: SPEAKER_02 -> 'Speaker 2'; real names pass through.

    Zero-padding is stripped (via int()), zero-indexing is preserved:
        SPEAKER_00 -> 'Speaker 0'
        SPEAKER_02 -> 'Speaker 2'
        SPEAKER_12 -> 'Speaker 12'
        Amanda     -> 'Amanda'
    """
    m = SPEAKER_ID_RE.match(raw)
    if m:
        return f"Speaker {int(m.group(1))}"
    return raw


def _looks_like_transcriber(text: str) -> bool:
    """Cheap sniff: does this file contain H3 speaker headings?"""
    return bool(re.search(r"^###\s+\S", text, re.MULTILINE))


def _looks_like_canonical(text: str) -> bool:
    """Cheap sniff: does this file contain inline **Name:** tags?"""
    return bool(re.search(r"^\*\*[^*\n]+:\*\*", text, re.MULTILINE))


def _parse_transcriber_turns(text: str) -> list[dict]:
    """Parse our transcriber's .md format into canonical turn dicts.

    Skips metadata per D2: bare '---', lines starting with '**Duration:**',
    and any pre-existing '# Transcript:' title line. Merges consecutive
    same-speaker H3 blocks into one turn.
    """
    turns: list[dict] = []
    current_speaker: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current_speaker and buffer:
            joined = " ".join(s for s in (b.strip() for b in buffer) if s)
            if joined:
                turns.append(
                    {
                        "index": len(turns),
                        "speaker": current_speaker,
                        "text": joined,
                    }
                )

    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        # D2: drop metadata/structural lines
        if stripped == "---":
            continue
        if stripped.startswith("**Duration:**"):
            continue
        if stripped.startswith("# Transcript:"):
            continue

        m = TRANSCRIBER_H3_RE.match(stripped)
        if m:
            raw_speaker = _extract_speaker_from_heading(m.group(1))
            new_speaker = _normalize_speaker_tag(raw_speaker)
            if new_speaker == current_speaker:
                # Consecutive block from same speaker — keep accumulating.
                continue
            flush()
            current_speaker = new_speaker
            buffer = []
        else:
            buffer.append(stripped)

    flush()
    return turns


def _parse_canonical_md_turns(text: str) -> list[dict]:
    """Parse already-canonical '**Speaker N:** text' markdown into turn dicts.

    Handles the re-ingest case where a prior step1 output is fed back in.
    """
    turns: list[dict] = []
    current_speaker: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current_speaker and buffer:
            joined = " ".join(s for s in (b.strip() for b in buffer) if s)
            if joined:
                turns.append(
                    {
                        "index": len(turns),
                        "speaker": current_speaker,
                        "text": joined,
                    }
                )

    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue  # '# Transcript: ...' title line

        m = CANONICAL_INLINE_RE.match(stripped)
        if m:
            flush()
            current_speaker = m.group(1).strip()
            first_text = m.group(2).strip()
            buffer = [first_text] if first_text else []
        else:
            # Continuation line (rare — our emitter writes single-line turns)
            buffer.append(stripped)

    flush()
    return turns


# ---------------------------------------------------------------------------
# Public entry point (dispatch by suffix)
# ---------------------------------------------------------------------------


def convert(input_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """Ingest a transcript file and emit canonical markdown + JSON sidecar.

    Dispatches by file suffix:
      .rtf  ->  Moonshine RTF ingester
      .md   ->  Local transcriber markdown ingester (M2)

    Returns (json_path, md_path).
    """
    suffix = input_path.suffix.lower()
    if suffix == ".rtf":
        return _ingest_moonshine_rtf(input_path, out_dir)
    if suffix == ".md":
        return _ingest_markdown(input_path, out_dir)
    raise SystemExit(
        f"Unsupported input format: '{suffix}'. Expected .rtf or .md."
    )


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------


def main() -> None:
    doc_lines = (__doc__ or "").splitlines()
    ap = argparse.ArgumentParser(
        description=doc_lines[1] if len(doc_lines) > 1 else ""
    )
    ap.add_argument(
        "input_file",
        type=Path,
        help="Input transcript (.rtf from Moonshine, or .md from our transcriber)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Output directory (default: cwd)",
    )
    args = ap.parse_args()

    json_path, md_path = convert(args.input_file, args.out_dir)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
