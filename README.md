# README

## Description

A multi-agent, privacy-first pipeline that transforms raw `.rtf` meeting transcripts from [Moonshine.ai](https://note-taker.moonshine.ai/) into polished, corporate-grade meeting minutes using local LLMs (like Gemma4 and Qwen) via Ollama.

## Overview

Generating high-quality meeting minutes locally from 45+ minute transcripts is challenging. Passing a raw `.rtf` file with a single massive system prompt to a ~27B parameter local model often results in cognitive overload, hallucinated action items, and dropped details. 

**Our Solution:** We break the problem down into a **5-step chained pipeline**. By isolating tasks—cleanup, human-in-the-loop speaker identification, data extraction, and final formatting—we can achieve "Google-level" summary quality using local, consumer-grade hardware while keeping sensitive corporate data 100% private.

---

## Key Learnings & Architecture Decisions

During the development of this pipeline, several critical discoveries shaped the architecture:

1. **RTF Noise vs. Markdown:** Raw RTF tags consume thousands of wasted tokens and confuse LLMs. Stripping the RTF into explicit Markdown (`**Speaker 1:** text`) acts as an anchor, helping the model perfectly distinguish between the speaker and the dialogue.
2. **The "Human-in-the-Loop" Necessity:** LLMs frequently hallucinate action-item ownership if speakers don't explicitly name themselves. A fast, non-LLM CLI prompt to map generic tags (e.g., "Speaker 1") to real names (e.g., "Andro") eliminates this risk entirely.
3. **Extraction vs. Formatting:** Asking an LLM to extract data *and* format it into tables simultaneously leads to data loss. We split this: Agent 2 acts as a "Data Harvester" (extracting exhaustive, categorized bullets), and Agent 3 acts as the "Publisher" (formatting the dense data into clean tables and lists).
4. **Model Nuances (Gemma vs. Qwen):** * **Gemma (~27B)** excels at natural language smoothing and narrative flow but needs strict structural guides.
   * **Qwen (~27B)** is highly logical but tends to over-compress. It requires "negative constraints" (e.g., *CRITICAL INSTRUCTION: DO NOT summarize away technical details*) to ensure high data fidelity.
   * *Solution:* The pipeline uses **Dynamic Prompting**, automatically switching the internal system prompt based on the `--model` argument passed in the CLI.
5. **VRAM Optimization:** We utilize the `keep_alive=-1` parameter in the Ollama API to keep the LLM loaded in VRAM across the sequential scripts, drastically reducing execution time.

---

## Prerequisites

* **Python 3.8+**
* **[uv](https://github.com/astral-sh/uv)** (Python package manager)
* **[Ollama](https://ollama.com/)** running locally or on your network (Default expects: `http://<YOUR OLLAMA IP>:11434`)
* **Local Models:** Pull your preferred models in Ollama:
  
  ```bash
  ollama pull gemma4:26b
  ollama pull qwen3.5:27b
   ```

## The 5-Step Pipeline Workflow

### Step 1: Parse RTF to Markdown (Non-LLM)

Strips noisy .rtf formatting and groups consecutive speech into clean Markdown and JSON (_not used in further processing currently_).

```bash
uv run 01_convert_rtf.py transcripts/<MeetingTranscript>.rtf --out-dir output/raw_files/
```

### Step 2: Agent 1 - Transcript Cleanup

The LLM acts as a **Data Cleaner**.

It proofreads the raw markdown, removes verbal stutters, false starts, and filler words without summarizing or losing chronological context.

```bash
uv run 02_agent1_cleanup.py output/raw_files/<MeetingTranscript>.md --out-dir output/cleaned_files/
```

### Step 3: Speaker Mapping (Human-in-the-Loop)

A quick CLI script that scans for `Speaker X:` tags and pauses to ask you for their real names. 

It then performs a global find-and-replace, ensuring 100% accurate attribution for the subsequent AI steps.

```bash
uv run 03_speaker_mapping.py output/cleaned_files/<MeetingTranscript_cleaned>.md --out-dir output/named_files/
```

### Step 4: Agent 2 - Information Extraction

The LLM acts as a **Data Harvester**. 

It scans the named transcript and extracts exhaustive, high-fidelity bullet points, organizing them into logical H3 sub-categories while preserving specific metrics, dates, and brands.

```bash
uv run 04_agent2_extraction.py output/named_files/<MeetingTranscript_named>.md --out-dir output/extracted_files/
```

### Step 5: Agent 3 - Final Formatting

The LLM acts as the **Publisher**. 

It takes the dense extraction and formats it into a professional layout, generating a "Participants" list and organizing Action Items into a strict Markdown table `(Task | Owner | Status)`.

```bash
uv run 05_agent3_formatter.py output/extracted_files/<MeetingTranscript_extracted>.md --out-dir output/final_summaries/
```

---

## ⚙️ Advanced CLI Usage

All AI agents (`02`, `04`, `05`) support CLI overrides for the model and the host URL. The scripts will automatically detect if you are using a Gemma or Qwen model and apply the optimized system prompt.

> [!Warning] 
> As of April, 2026, the internal system prompts are tailored to `gemma4:26b` and `qwen3.5:27b`. If using a different model or even same family models with higher or lower weights, you might need to adjust the system prompts via experimentation or develop / tweak affordance of the respective agent scripts. 

### Mixing and Matching Models (The "Best of Both Worlds" Strategy)
Because different LLMs excel at different cognitive tasks, you are not locked into a single model for the entire pipeline. 

For example, Gemma models are historically fantastic at natural language smoothing and narrative flow, making them ideal for formatting. Qwen models are highly logical and obedient to structural constraints, making them perfect for exhaustive data extraction. 

You can leverage this by switching the `--model` argument at each step:

```bash
# Step 2: Use Gemma for natural, grammatical text cleanup
uv run 02_agent1_cleanup.py output/raw_files/Meeting.md \
    --out-dir output/cleaned_files/ \
    --model gemma4:26b

# Step 4: Switch to Qwen for rigid, exhaustive data extraction
uv run 04_agent2_extraction.py output/named_files/Meeting_named.md \
    --out-dir output/extracted_files/ \
    --model qwen3.5:27b

# Step 5: Switch back to Gemma for polished, corporate document formatting
uv run 05_agent3_formatter.py output/extracted_files/Meeting_extracted.md \
    --out-dir output/final_summaries/ \
    --model gemma4:26b
```

### Customizing the Ollama Host

If you are running Ollama on a dedicated home server, a secondary GPU rig, or within a specific Docker network, you can override the default local URL using the `--host` flag:

```bash
uv run 02_agent1_cleanup.py input.md --host http://<your_ollama_host_ADDR>:<your_ollama_host_PORT>
```

### Batch Processing and Chaining

Because every step in this pipeline is an isolated CLI command with distinct input/output directories, you can easily wrap these commands into a simple bash script (`.sh`). This allows you to process multiple transcripts sequentially, only halting the script for the human-in-the-loop speaker mapping ([Step 3](#step-3-speaker-mapping-human-in-the-loop)).


## Directory Structure

```txt
.
├── 01_convert_rtf.py
├── 02_agent1_cleanup.py
├── 03_speaker_mapping.py
├── 04_agent2_extraction.py
├── 05_agent3_formatter.py
├── output
│   ├── cleaned_files
│   │   └── README.md
│   ├── extracted_files
│   │   └── README.md
│   ├── final_summaries
│   │   └── README.md
│   ├── named_files
│   │   └── README.md
│   └── raw_files
│       └── README.md
├── pyproject.toml
├── README.md
├── transcripts
│   └── README.md
└── uv.lock
```

## License

[MIT](License)

---

<sub>Saurabh Datta · [zigzag.is](https://zigzag.is) · Berlin · April 2026</sub>