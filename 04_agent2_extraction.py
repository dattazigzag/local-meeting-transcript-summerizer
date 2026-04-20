#!/usr/bin/env python3
"""
Agent 2: Information Extraction
Reads a named transcript and extracts the Executive Summary,
Key Discussion Points, and Action Items using a local Ollama model.
"""

import argparse
import sys
from pathlib import Path
from ollama import Client, ResponseError

# --- Global Configurations ---
DEFAULT_MODEL = "qwen3.5:27b" # or "gemma4:26b"  # Change to your specific tag
DEFAULT_HOST = "http://192.168.178.160:11434"

# For gemma4, we need a more detailed prompt to ensure it follows the exact output format and doesn't add any extra text:
# OLD
# SYSTEM_PROMPT = """
# You are an expert executive assistant. Your task is to analyze a meeting transcript and extract the core business information.

# Strict rules to follow:
# 1. Output exactly three sections formatted exactly like this:
#    ## Executive Summary
#    [A brief 2-3 sentence summary of the meeting's overall purpose and outcome.]
   
#    ## Key Discussion Points
#    * [Main theme, decision, or important detail]
#    * [Main theme, decision, or important detail]
   
#    ## Action Items
#    * [Task] - **[Owner Name]**
   
# 2. Use the exact speaker names provided in the text to assign Action Items to their correct owners.
# 3. Do NOT output any conversational filler, introductory greetings, or concluding remarks. Output ONLY the three requested sections.
# """
# NEW
# SYSTEM_PROMPT = """
# You are an expert executive assistant. Your task is to perform a comprehensive, high-fidelity extraction of information from a meeting transcript.

# Strict rules to follow:
# 1. Output exactly three main sections:
#    ## Executive Summary
#    [Write a comprehensive paragraph explaining the meeting's purpose, main debate, and outcome.]
   
#    ## Key Discussion Points
#    [You MUST group the discussion points into 3 to 5 logical sub-categories using bolded H3 tags (e.g., ### A. Technical Requirements). Under each sub-category, provide detailed bullet points.]
   
#    ## Action Items
#    * [Task] - **[Owner Name]**
   
# 2. DATA FIDELITY: You must extract specific numbers, dates, dimensions, technical specifications, and historical context mentioned in the text. Do not generalize.
# 3. Use the exact speaker names provided in the text to assign Action Items. 
# 4. Output ONLY the requested sections. Do not include conversational filler.
# """

# For qwen3.5, we can use a more concise prompt since it has better instruction-following capabilities:
SYSTEM_PROMPT = """
You are an expert executive assistant known for meticulous, exhaustive documentation. Your task is to extract information from a meeting transcript.

CRITICAL INSTRUCTION: You tend to be too concise. You MUST NOT summarize away technical details. If a specific brand, measurement, ratio, timeframe, or historical date is mentioned in the transcript, it MUST be extracted and included in your bullet points.

Strict rules to follow:
1. REQUIRED STRUCTURE:
   ## Executive Summary
   [Provide a detailed paragraph covering the meeting's context, main debate, and outcome.]
   
   ## Key Discussion Points
   [You MUST group the discussion points into logical sub-categories using bolded H3 tags, for example: ### A. Technical Requirements. Under each header, use exhaustive bullet points that capture the full depth and specific data points of the conversation.]
   
   ## Action Items
   * [Specific Task] - **[Owner Name]**
   
2. ACTION ITEM ASSIGNMENT: Use the exact speaker names provided in the text to assign action items. If a task is implied but not explicitly assigned, note it as "Unassigned".
3. OUTPUT FORMAT: Output ONLY the requested markdown structure. Do not add conversational filler, introductory text, or concluding remarks.
"""

def extract_information(input_md: Path, out_dir: Path, model: str, host: str) -> Path:
    print(f"Reading {input_md.name}...")
    content = input_md.read_text(encoding="utf-8")

    client = Client(host=host)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Extract the information from this transcript:\n\n{content}",
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

    extracted_text = response.get("message", {}).get("content", "").strip()

    out_dir.mkdir(parents=True, exist_ok=True)

    new_stem = input_md.stem.replace("_named", "")
    out_path = out_dir / f"{new_stem}_extracted.md"

    out_path.write_text(extracted_text, encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract key info from a named transcript using Ollama."
    )
    parser.add_argument("input_md", type=Path, help="Input named .md transcript file")
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

    out_file = extract_information(args.input_md, args.out_dir, args.model, args.host)
    print(f"Successfully wrote extracted information to: {out_file}")


if __name__ == "__main__":
    main()
