#!/usr/bin/env python3
"""
Agent 3: Meeting Summary Formatter
Reads the exhaustive extraction from Agent 2 and formats it into a final,
polished meeting summary with a Participants list and Action Item tables.
Automatically selects the optimal system prompt based on the chosen model.
"""

import argparse
import sys
from pathlib import Path
from ollama import Client, ResponseError

# --- Global Configurations ---
DEFAULT_MODEL = "gemma4:26b"
DEFAULT_HOST = "http://192.168.178.160:11434"

# --- Prompts ---

PROMPT_GEMMA = """
You are an expert technical writer and executive assistant. Your task is to format a raw meeting extraction into a final, polished Markdown document.

Strict Formatting Rules:
1. DOCUMENT STRUCTURE: The final document must follow this exact order:
   # Meeting Minutes: [Infer a concise title based on the context]
   
   ## Participants
   * [Extract a list of all unique names mentioned as participants or task owners in the text]
   
   ## 1. Executive Summary
   [Polish the provided summary for a professional corporate tone]
   
   ## 2. Key Discussion Points
   [Keep all the detailed H3 sub-categories (e.g., ### A. Technical Requirements) and bullet points provided in the text. Polish the grammar but DO NOT remove any data, metrics, or specific details.]
   
   ## 3. Action Items
   [Convert the provided action items into a Markdown table exactly like this]
   | Task | Owner | Status |
   | :--- | :--- | :--- |
   | [Task Description] | [Owner Name] | Pending |

2. TONE & FIDELITY: Ensure the tone is highly professional. You must retain all specific metrics, dimensions, numbers, and historical context from the input.
3. OUTPUT: Output ONLY the requested Markdown. Do not include conversational filler (e.g., "Here is your summary").
"""

PROMPT_QWEN = """
You are an expert technical writer. You are receiving a detailed data extraction from a meeting. You must format it into a polished summary.

CRITICAL INSTRUCTION: You must strictly adhere to the table formatting for Action Items. You must NOT summarize away the detailed bullet points under "Key Discussion Points". Retain all data.

Follow this EXACT structure:
# Meeting Minutes: [Generate a short title]

## Participants
* [List all unique names found in the text]

## 1. Executive Summary
[Refine the provided summary text into a professional paragraph]

## 2. Key Discussion Points
[Retain all H3 sub-headers (e.g., ### A. Technical Requirements) and their bullet points. Do not omit any numbers, dimensions, or brands.]

## 3. Action Items
You MUST output a Markdown table with the following syntax:
| Task | Owner | Status |
|---|---|---|
| [Specific Task] | [Owner Name] | Pending |

Output ONLY the requested Markdown structure. Do not output anything else.
"""

PROMPT_DEFAULT = """
You are an expert technical writer. Format the provided meeting extraction into a final polished summary.

Include the following sections:
# Meeting Minutes: [Title]
## Participants
[List of names]
## 1. Executive Summary
## 2. Key Discussion Points
[Keep all existing sub-headers and detailed bullets]
## 3. Action Items
[Format as a Markdown table with columns: Task, Owner, Status (default to 'Pending')]

Output ONLY the Markdown.
"""


def get_system_prompt(model_name: str) -> str:
    """Selects the appropriate formatting prompt based on the model name."""
    model_lower = model_name.lower()
    if "gemma" in model_lower:
        print("-> Using Gemma-optimized formatter prompt.")
        return PROMPT_GEMMA
    elif "qwen" in model_lower:
        print("-> Using Qwen-optimized formatter prompt.")
        return PROMPT_QWEN
    else:
        print("-> Using Default formatter prompt.")
        return PROMPT_DEFAULT


def format_summary(input_md: Path, out_dir: Path, model: str, host: str) -> Path:
    print(f"Reading {input_md.name}...")
    content = input_md.read_text(encoding="utf-8")

    client = Client(host=host)
    selected_prompt = get_system_prompt(model)

    messages = [
        {"role": "system", "content": selected_prompt},
        {
            "role": "user",
            "content": f"Format this extracted data into the final meeting summary:\n\n{content}",
        },
    ]

    print(f"Sending to Ollama ({model}) at {host}...")
    try:
        response = client.chat(model=model, messages=messages, keep_alive=-1)
    except ResponseError as e:
        print(f"Ollama API Error: {e.error}")
        sys.exit(1)
    except Exception as e:
        print(f"Connection Error: {e}")
        sys.exit(1)

    final_text = response.get("message", {}).get("content", "").strip()

    out_dir.mkdir(parents=True, exist_ok=True)

    # Rename from _extracted to _summary
    new_stem = input_md.stem.replace("_extracted", "")
    out_path = out_dir / f"{new_stem}_summary.md"

    out_path.write_text(final_text, encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Format extracted data into a polished meeting summary using Ollama."
    )
    parser.add_argument(
        "input_md", type=Path, help="Input extracted .md transcript file"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("."), help="Output directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Ollama model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"Ollama host URL (default: {DEFAULT_HOST})",
    )

    args = parser.parse_args()

    if not args.input_md.exists():
        print(f"File not found: {args.input_md}")
        sys.exit(1)

    out_file = format_summary(args.input_md, args.out_dir, args.model, args.host)
    print(f"Successfully wrote final summary to: {out_file}")


if __name__ == "__main__":
    main()
