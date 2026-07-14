# Edge TTS Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Microsoft Edge TTS (free, no API key, good Vietnamese voices) as a third `voiceover.py` provider alongside ElevenLabs and Qwen3-TTS.

**Architecture:** A new standalone module `tools/edgetts.py` owns all edge-tts-package calls and returns result dicts shaped like the existing providers' (`success, output, duration_seconds, duration_frames_30fps, wpm, pacing`). `tools/voiceover.py` gets a third provider branch wired into every existing provider switch (argparse, brand config, scene-dir loop, single-file mode, dry-run/JSON output). `tools/pacing.py` is reused unchanged — it already takes a plain audio path + text and is provider-agnostic.

**Tech Stack:** Python 3.13, `edge-tts` pip package (async, wraps `asyncio.run`), ffmpeg/ffprobe (already a hard dependency via `pacing.py`).

**Testing approach:** This repo has no pytest suite for `tools/` — every existing tool (`qwen3_tts.py`, `voiceover.py`, etc.) is verified by running its CLI with `--dry-run`/`--json` and inspecting output, not unit tests. This plan follows that convention: each task's "verify" steps are real CLI invocations with concrete expected output, not a test framework. Voice generation and `--list-voices` steps make a live network call to Microsoft's edge-tts service — they require internet access in the execution environment.

## Global Constraints

- New module filename MUST be `tools/edgetts.py` (no underscore) — `voiceover.py` prepends `tools/` to `sys.path`, so a file literally named `edge_tts.py` would shadow the real `edge_tts` pip package for any `import edge_tts` issued from within `tools/`.
- `edge-tts>=6.1.0` is added to `tools/requirements.txt` as a normal (uncommented) dependency — it's free, lightweight, and needs no signup, consistent with the toolkit's "most features are free" default.
- Edge-TTS needs no API key, no `.env` entry, and no cloud GPU setup — do not add env var plumbing for it.
- The existing `--speaker` CLI flag (currently Qwen3-only) is reused as the Edge-TTS voice name. No new `--voice` flag.
- Default Edge-TTS voice is `vi-VN-HoaiMyNeural`.
- Every provider's result dict must keep including `success, output, duration_seconds, duration_frames_30fps, wpm, pacing` — `sync_timing.py` and `_apply_pacing_qc`-adjacent code depend on this shape being provider-agnostic.
- Brand voice config: only `brands/default/voice.json` gets an `edgeTts` block in this plan. Other brand profiles are untouched.
- Out of scope (do not build): tone/instruct presets for Edge-TTS, voice cloning, `edgeTts` blocks in non-default brands.
- Argparse gotcha: literal `%` characters in an `add_argument(..., help="...")` string MUST be escaped as `%%` (argparse applies `%`-substitution to every action's help text). Literal `%` in `description=`/`epilog=` strings does NOT need escaping (those are not `%`-substituted).

---

## Task 1: Core Edge-TTS module (`tools/edgetts.py`)

**Files:**
- Modify: `tools/requirements.txt`
- Create: `tools/edgetts.py`

**Interfaces:**
- Produces: `generate_audio(text: str, output_path: str, voice: str = "vi-VN-HoaiMyNeural", rate: str = "+0%", pitch: str = "+0Hz", volume: str = "+0%", max_wpm: float | None = None, verbose: bool = True) -> dict` — returns `{"success": True, "output": str, "duration_seconds": float|None, "duration_frames_30fps": int|None, "script_chars": int, "wpm": float|None, "pacing": str|None, ...}` or `{"success": False, "error": str}`.
- Produces: `list_voices(language: str | None = None) -> list[dict]` — each dict has at least `ShortName`, `Gender`, `Locale` keys (as returned by the `edge_tts` package). Returns `[]` if the package isn't installed.
- Produces: `DEFAULT_VOICE = "vi-VN-HoaiMyNeural"` module constant.
- Consumes: `tools/pacing.py`'s `clamp_pace(audio_path, text, max_wpm, verbose) -> dict` and `pace_label(text, duration_seconds) -> (wpm, label)` (already exist, unchanged).

- [ ] **Step 1: Add the dependency**

Edit `tools/requirements.txt`. Current top of file:

```
# Video Toolkit Python Dependencies
elevenlabs>=1.0.0
python-dotenv>=1.0.0
```

Change to:

```
# Video Toolkit Python Dependencies
elevenlabs>=1.0.0
edge-tts>=6.1.0  # Free Microsoft Edge TTS voices (incl. Vietnamese) — no API key needed
python-dotenv>=1.0.0
```

- [ ] **Step 2: Install and verify the package imports**

Run: `pip install edge-tts`
Then run: `python -c "import edge_tts; print('edge-tts import OK')"`
Expected: prints `edge-tts import OK` with no traceback.

- [ ] **Step 3: Write `tools/edgetts.py`**

Create the file with this exact content:

```python
#!/usr/bin/env python3
"""
Generate speech using Microsoft Edge TTS (free, no API key, no cloud GPU).

Includes high-quality Vietnamese neural voices (vi-VN-HoaiMyNeural,
vi-VN-NamMinhNeural) alongside ~300 other voices across languages. Used as
a `voiceover.py --provider edge-tts` backend; also usable standalone.

Usage:
    # Single utterance
    python tools/edgetts.py --text "Xin chao" --voice vi-VN-HoaiMyNeural --output hello.mp3

    # Adjust rate/pitch/volume
    python tools/edgetts.py --text "Hello" --voice en-US-AriaNeural --rate +10% --pitch -5Hz --output out.mp3

    # List Vietnamese voices
    python tools/edgetts.py --list-voices --language vi

    # List all voices
    python tools/edgetts.py --list-voices

Setup:
    pip install edge-tts
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Curated reference — NOT an exhaustive or enforced list. --voice/--speaker
# accepts any edge-tts voice name; these are just the toolkit's Vietnamese
# defaults, surfaced by --list-voices for convenience.
VIETNAMESE_VOICES = {
    "vi-VN-HoaiMyNeural": "Female",
    "vi-VN-NamMinhNeural": "Male",
}

DEFAULT_VOICE = "vi-VN-HoaiMyNeural"


def _get_edge_tts_import():
    """Lazy import the edge-tts package. Returns the module, or None (and
    prints an install hint) if it isn't installed."""
    try:
        import edge_tts
        return edge_tts
    except ImportError:
        print(
            "Error: edge-tts Python package not installed.\n"
            "\n"
            "  pip install edge-tts\n"
            "\n"
            "Edge TTS is free and needs no API key or account — it's a wrapper\n"
            "around Microsoft Edge's built-in read-aloud voices.",
            file=sys.stderr,
        )
        return None


def get_audio_duration(file_path: str) -> float | None:
    """Get audio duration in seconds using ffprobe."""
    import subprocess
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                file_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except (FileNotFoundError, ValueError):
        pass
    return None


async def _synthesize(text: str, output_path: str, voice: str, rate: str, pitch: str, volume: str) -> None:
    import edge_tts
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, volume=volume)
    await communicate.save(output_path)


def generate_audio(
    text: str,
    output_path: str,
    voice: str = DEFAULT_VOICE,
    rate: str = "+0%",
    pitch: str = "+0Hz",
    volume: str = "+0%",
    max_wpm: float | None = None,
    verbose: bool = True,
) -> dict:
    """Generate one audio file using Microsoft Edge TTS.

    Pacing QC: every result includes `wpm` (words per minute) and a `pacing`
    label ('fast'/'slow'/'ok'/None). If `max_wpm` is set, takes that exceed
    it are slowed in place with pitch-preserving ffmpeg atempo (floor 0.85x).
    See tools/pacing.py.

    Returns: {success, output, duration_seconds, duration_frames_30fps,
    script_chars, wpm, pacing} or {success: False, error} on failure.
    """
    if _get_edge_tts_import() is None:
        return {"success": False, "error": "edge-tts package not installed (pip install edge-tts)"}

    if not text:
        return {"success": False, "error": "text must be non-empty"}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Generating speech with Edge TTS (voice={voice})...", file=sys.stderr)

    try:
        asyncio.run(_synthesize(text, output_path, voice, rate, pitch, volume))
    except Exception as e:
        return {"success": False, "error": f"edge-tts synthesis failed: {e}"}

    duration = get_audio_duration(output_path)
    result = {
        "success": True,
        "output": output_path,
        "duration_seconds": round(duration, 2) if duration else None,
        "duration_frames_30fps": int(duration * 30) if duration else None,
        "script_chars": len(text),
    }

    from pacing import clamp_pace, pace_label
    if max_wpm:
        clamp = clamp_pace(output_path, text, max_wpm, verbose=verbose)
        if clamp.get("applied"):
            new_dur = clamp["duration_seconds"]
            result["duration_seconds"] = new_dur
            result["duration_frames_30fps"] = int(new_dur * 30) if new_dur else None
            result["pace_adjusted"] = {
                "original_wpm": clamp["original_wpm"],
                "atempo": clamp["atempo"],
            }
        elif clamp.get("error") and verbose:
            print(f"  Pace clamp skipped: {clamp['error']}", file=sys.stderr)
    wpm, label = pace_label(text, result["duration_seconds"])
    result["wpm"] = wpm
    result["pacing"] = label
    if verbose and label in ("fast", "slow"):
        print(
            f"  Pacing warning: {Path(output_path).name} is {wpm:.0f} wpm "
            f"({label}; comfortable narration is 140-160). "
            "Consider --max-wpm to auto-correct, or adjust --rate.",
            file=sys.stderr,
        )

    return result


async def _list_voices_async(language: str | None) -> list[dict]:
    import edge_tts
    voices = await edge_tts.list_voices()
    if language:
        prefix = language.lower() + "-"
        voices = [v for v in voices if v["Locale"].lower().startswith(prefix)]
    return voices


def list_voices(language: str | None = None) -> list[dict]:
    """Fetch the live edge-tts voice catalogue, optionally filtered by locale
    prefix (e.g. language="vi" matches vi-VN-*)."""
    if _get_edge_tts_import() is None:
        return []
    return asyncio.run(_list_voices_async(language))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate speech using Microsoft Edge TTS (free, no API key)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/edgetts.py --text "Xin chao" --voice vi-VN-HoaiMyNeural --output hello.mp3
  python tools/edgetts.py --text "Hello" --voice en-US-AriaNeural --rate +10% --output out.mp3
  python tools/edgetts.py --list-voices --language vi
        """,
    )
    parser.add_argument("--text", "-t", type=str, help="Text to synthesize")
    parser.add_argument("--output", "-o", type=str, help="Output audio file path (.mp3)")
    parser.add_argument(
        "--voice", "-v",
        type=str,
        default=DEFAULT_VOICE,
        help=f"Edge-TTS voice name (default: {DEFAULT_VOICE}). Use --list-voices to see options.",
    )
    parser.add_argument("--rate", type=str, default="+0%", help="Speech rate delta, e.g. '+10%%' or '-15%%' (default: +0%%)")
    parser.add_argument("--pitch", type=str, default="+0Hz", help="Pitch delta, e.g. '+5Hz' or '-10Hz' (default: +0Hz)")
    parser.add_argument("--volume", type=str, default="+0%", help="Volume delta, e.g. '+10%%' or '-20%%' (default: +0%%)")
    parser.add_argument(
        "--max-wpm",
        type=float,
        default=None,
        help="Pace clamp: if the take exceeds this words-per-minute, slow it "
             "with pitch-preserving atempo (floor 0.85x). Try 165.",
    )
    parser.add_argument("--list-voices", action="store_true", help="List voices and exit")
    parser.add_argument("--language", type=str, help="Filter --list-voices by locale prefix, e.g. 'vi', 'en'")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    verbose = not args.json

    if args.list_voices:
        voices = list_voices(args.language)
        if args.json:
            print(json.dumps(voices, indent=2))
            return
        if not voices:
            print("No voices found (or edge-tts not installed / network unavailable).")
            sys.exit(1)
        print(f"{'Voice':<28} {'Gender':<8} {'Locale'}")
        print(f"{'-'*28} {'-'*8} {'-'*8}")
        for v in voices:
            print(f"{v['ShortName']:<28} {v['Gender']:<8} {v['Locale']}")
        print()
        print("Vietnamese defaults: " + ", ".join(f"{k} ({g})" for k, g in VIETNAMESE_VOICES.items()))
        return

    if not args.text:
        print("Error: --text is required", file=sys.stderr)
        sys.exit(1)
    if not args.output:
        print("Error: --output is required", file=sys.stderr)
        sys.exit(1)

    result = generate_audio(
        text=args.text,
        output_path=args.output,
        voice=args.voice,
        rate=args.rate,
        pitch=args.pitch,
        volume=args.volume,
        max_wpm=args.max_wpm,
        verbose=verbose,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    elif result.get("success"):
        print(f"Saved to: {result['output']}", file=sys.stderr)
        if result.get("duration_seconds"):
            print(
                f"Duration: {result['duration_seconds']:.2f}s "
                f"({result['duration_frames_30fps']} frames @ 30fps, {result.get('wpm', '?')} wpm)",
                file=sys.stderr,
            )
    else:
        print(f"Error: {result.get('error', 'Unknown error')}", file=sys.stderr)

    if not result.get("success"):
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify `--help` works**

Run: `python tools/edgetts.py --help`
Expected: prints usage/options text, exits 0, no traceback (this alone catches the argparse `%`-escaping mistake — an unescaped `%` in a `help=` string raises at format time).

- [ ] **Step 5: Verify live voice listing includes both Vietnamese voices**

Run:
```bash
python tools/edgetts.py --list-voices --language vi --json > /tmp/edgetts_vi_voices.json
python -c "
import json
data = json.load(open('/tmp/edgetts_vi_voices.json'))
names = {v['ShortName'] for v in data}
assert 'vi-VN-HoaiMyNeural' in names, names
assert 'vi-VN-NamMinhNeural' in names, names
print(f'OK: {len(data)} Vietnamese voices found, including both defaults')
"
```
Expected: `OK: <N> Vietnamese voices found, including both defaults` with no AssertionError. (Requires internet access — this hits Microsoft's live voice list.)

- [ ] **Step 6: Verify real audio generation**

Run:
```bash
python tools/edgetts.py --text "Xin chào, đây là bài kiểm tra giọng đọc tiếng Việt." --voice vi-VN-HoaiMyNeural --output /tmp/edgetts_test.mp3 --json
```
Expected: JSON with `"success": true`, `"duration_seconds"` a positive number (roughly 3-6s for that sentence), `"wpm"` present, `"pacing"` one of `"ok"/"fast"/"slow"`. Confirm the file exists and is non-trivial in size:
```bash
ls -la /tmp/edgetts_test.mp3
```
Expected: file exists, size > 10KB.

- [ ] **Step 7: Commit**

```bash
git add tools/requirements.txt tools/edgetts.py
git commit -m "$(cat <<'EOF'
ADD: tools/edgetts.py — free Microsoft Edge TTS module (Vietnamese voices)

Standalone generate_audio()/list_voices() following the same result-dict
shape as qwen3_tts.py, so voiceover.py can adopt it as a third provider.
No API key or cloud GPU needed.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Wire Edge-TTS into `tools/voiceover.py`

**Files:**
- Modify: `tools/voiceover.py`

**Interfaces:**
- Consumes: `edgetts.generate_audio(text, output_path, voice, rate, pitch, volume, max_wpm, verbose) -> dict` (Task 1).
- Produces: `generate_single_audio_edge_tts(script: str, output_path: Path, voice: str = "vi-VN-HoaiMyNeural", rate: str = "+0%", pitch: str = "+0Hz", volume: str = "+0%", max_wpm: float | None = None) -> dict`, used the same way `generate_single_audio_qwen3` is used elsewhere in this file.

- [ ] **Step 1: Update the module docstring with Edge-TTS examples**

In `tools/voiceover.py`, find:

```python
"""
Generate voiceover audio using ElevenLabs or Qwen3-TTS.

Usage:
    # From script file (ElevenLabs, default)
    python tools/voiceover.py --script VOICEOVER-SCRIPT.md --output public/audio/voiceover.mp3

    # From stdin (for AI piping)
    echo "Hello world" | python tools/voiceover.py --output voiceover.mp3

    # With custom voice
    python tools/voiceover.py --script script.txt --voice-id ABC123 --output out.mp3

    # JSON output for machine parsing
    python tools/voiceover.py --script script.txt --output out.mp3 --json

    # Per-scene generation (recommended)
    python tools/voiceover.py --scene-dir public/audio/scenes --json

    # With concat for SadTalker narrator
    python tools/voiceover.py --scene-dir public/audio/scenes --concat public/audio/voiceover-concat.mp3

    # Using Qwen3-TTS provider
    python tools/voiceover.py --provider qwen3 --speaker Ryan --scene-dir public/audio/scenes --json
    python tools/voiceover.py --provider qwen3 --tone warm --scene-dir public/audio/scenes --json
    python tools/voiceover.py --provider qwen3 --instruct "Speak warmly" --script script.txt --output out.mp3
"""
```

Replace with:

```python
"""
Generate voiceover audio using ElevenLabs, Qwen3-TTS, or Edge-TTS.

Usage:
    # From script file (ElevenLabs, default)
    python tools/voiceover.py --script VOICEOVER-SCRIPT.md --output public/audio/voiceover.mp3

    # From stdin (for AI piping)
    echo "Hello world" | python tools/voiceover.py --output voiceover.mp3

    # With custom voice
    python tools/voiceover.py --script script.txt --voice-id ABC123 --output out.mp3

    # JSON output for machine parsing
    python tools/voiceover.py --script script.txt --output out.mp3 --json

    # Per-scene generation (recommended)
    python tools/voiceover.py --scene-dir public/audio/scenes --json

    # With concat for SadTalker narrator
    python tools/voiceover.py --scene-dir public/audio/scenes --concat public/audio/voiceover-concat.mp3

    # Using Qwen3-TTS provider
    python tools/voiceover.py --provider qwen3 --speaker Ryan --scene-dir public/audio/scenes --json
    python tools/voiceover.py --provider qwen3 --tone warm --scene-dir public/audio/scenes --json
    python tools/voiceover.py --provider qwen3 --instruct "Speak warmly" --script script.txt --output out.mp3

    # Using Edge-TTS provider (free, no API key — good Vietnamese voices)
    python tools/voiceover.py --provider edge-tts --scene-dir public/audio/scenes --json
    python tools/voiceover.py --provider edge-tts --speaker vi-VN-NamMinhNeural --script script.txt --output out.mp3
    python tools/voiceover.py --provider edge-tts --rate +10% --script script.txt --output out.mp3
"""
```

- [ ] **Step 2: Update the argparse description and epilog**

Find:

```python
    parser = argparse.ArgumentParser(
        description="Generate voiceover using ElevenLabs or Qwen3-TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ElevenLabs (default)
  python tools/voiceover.py --script VOICEOVER-SCRIPT.md --output public/audio/voiceover.mp3
  python tools/voiceover.py --scene-dir public/audio/scenes --json

  # Qwen3-TTS
  python tools/voiceover.py --provider qwen3 --speaker Ryan --scene-dir public/audio/scenes --json
  python tools/voiceover.py --provider qwen3 --tone warm --scene-dir public/audio/scenes --json
  python tools/voiceover.py --provider qwen3 --instruct "Speak warmly" --script script.txt --output out.mp3
        """,
    )
```

Replace with:

```python
    parser = argparse.ArgumentParser(
        description="Generate voiceover using ElevenLabs, Qwen3-TTS, or Edge-TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ElevenLabs (default)
  python tools/voiceover.py --script VOICEOVER-SCRIPT.md --output public/audio/voiceover.mp3
  python tools/voiceover.py --scene-dir public/audio/scenes --json

  # Qwen3-TTS
  python tools/voiceover.py --provider qwen3 --speaker Ryan --scene-dir public/audio/scenes --json
  python tools/voiceover.py --provider qwen3 --tone warm --scene-dir public/audio/scenes --json
  python tools/voiceover.py --provider qwen3 --instruct "Speak warmly" --script script.txt --output out.mp3

  # Edge-TTS (free, no API key, good Vietnamese voices)
  python tools/voiceover.py --provider edge-tts --scene-dir public/audio/scenes --json
  python tools/voiceover.py --provider edge-tts --speaker vi-VN-NamMinhNeural --rate +10% --script script.txt --output out.mp3
        """,
    )
```

- [ ] **Step 3: Add `edge-tts` to the `--provider` choices**

Find:

```python
    parser.add_argument(
        "--provider",
        type=str,
        default="elevenlabs",
        choices=["elevenlabs", "qwen3"],
        help="TTS provider (default: elevenlabs)",
    )
```

Replace with:

```python
    parser.add_argument(
        "--provider",
        type=str,
        default="elevenlabs",
        choices=["elevenlabs", "qwen3", "edge-tts"],
        help="TTS provider (default: elevenlabs)",
    )
```

- [ ] **Step 4: Update `--speaker` help text to mention Edge-TTS reuse**

Find:

```python
    parser.add_argument(
        "--speaker",
        type=str,
        default="Ryan",
        help="Qwen3-TTS speaker name (default: Ryan). Use 'python tools/qwen3_tts.py --list-voices' to see options.",
    )
```

Replace with:

```python
    parser.add_argument(
        "--speaker",
        type=str,
        default="Ryan",
        help="Speaker/voice name. For Qwen3-TTS: built-in speaker (default: Ryan; "
             "'python tools/qwen3_tts.py --list-voices' to see options). For Edge-TTS: "
             "a voice name like 'vi-VN-HoaiMyNeural' (defaults to that voice when unset; "
             "'python tools/edgetts.py --list-voices' to see options).",
    )
```

- [ ] **Step 5: Add `--rate`/`--pitch`/`--volume` flags**

Find:

```python
    parser.add_argument(
        "--top-p",
        type=float,
        help="Qwen3-TTS nucleus sampling (default: model default ~0.8, range: 0.1-1.0)",
    )

    # Cloud GPU provider (for Qwen3-TTS)
    parser.add_argument(
        "--cloud",
        type=str,
        default="modal",
        choices=["runpod", "modal"],
        help="Cloud GPU provider for Qwen3-TTS (default: modal)",
    )
```

Replace with:

```python
    parser.add_argument(
        "--top-p",
        type=float,
        help="Qwen3-TTS nucleus sampling (default: model default ~0.8, range: 0.1-1.0)",
    )

    # Edge-TTS-specific options
    parser.add_argument(
        "--rate",
        type=str,
        default="+0%",
        help="Edge-TTS speech rate delta, e.g. '+10%%' or '-15%%' (default: +0%%).",
    )
    parser.add_argument(
        "--pitch",
        type=str,
        default="+0Hz",
        help="Edge-TTS pitch delta, e.g. '+5Hz' or '-10Hz' (default: +0Hz).",
    )
    parser.add_argument(
        "--volume",
        type=str,
        default="+0%",
        help="Edge-TTS volume delta, e.g. '+10%%' or '-20%%' (default: +0%%).",
    )

    # Cloud GPU provider (for Qwen3-TTS)
    parser.add_argument(
        "--cloud",
        type=str,
        default="modal",
        choices=["runpod", "modal"],
        help="Cloud GPU provider for Qwen3-TTS (default: modal)",
    )
```

- [ ] **Step 6: Add the `generate_single_audio_edge_tts()` wrapper**

Find:

```python
    return [
        {"success": True, **item}
        for item in result.get("outputs", [])
    ]


def process_scene_directory(
```

Replace with:

```python
    return [
        {"success": True, **item}
        for item in result.get("outputs", [])
    ]


def generate_single_audio_edge_tts(
    script: str,
    output_path: Path,
    voice: str = "vi-VN-HoaiMyNeural",
    rate: str = "+0%",
    pitch: str = "+0Hz",
    volume: str = "+0%",
    max_wpm: float | None = None,
) -> dict:
    """Generate a single audio file from script text using Edge TTS. Returns result dict."""
    from edgetts import generate_audio

    output_path.parent.mkdir(parents=True, exist_ok=True)

    return generate_audio(
        text=script,
        output_path=str(output_path),
        voice=voice,
        rate=rate,
        pitch=pitch,
        volume=volume,
        verbose=False,
        max_wpm=max_wpm,
    )


def process_scene_directory(
```

- [ ] **Step 7: Add `rate`/`pitch`/`volume` params to `process_scene_directory()`**

Find:

```python
    # Qwen3 params
    speaker: str = "Ryan",
    language: str = "Auto",
    instruct: str = "",
    ref_audio: str | None = None,
    ref_text: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    cloud: str = "runpod",
    max_wpm: float | None = None,
) -> list[dict]:
```

Replace with:

```python
    # Qwen3 params
    speaker: str = "Ryan",
    language: str = "Auto",
    instruct: str = "",
    ref_audio: str | None = None,
    ref_text: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    cloud: str = "runpod",
    # Edge-TTS params (speaker doubles as the voice name)
    rate: str = "+0%",
    pitch: str = "+0Hz",
    volume: str = "+0%",
    max_wpm: float | None = None,
) -> list[dict]:
```

- [ ] **Step 8: Add the edge-tts branch to the serial per-scene loop**

Find:

```python
    for s in scenes:
        if not json_output:
            print(f"Generating {s['mp3_file'].name}...", file=sys.stderr)

        if provider == "qwen3":
            result = generate_single_audio_qwen3(
                script=s["script"],
                output_path=s["mp3_file"],
                speaker=speaker,
                language=language,
                instruct=s["instruct"],
                ref_audio=ref_audio,
                ref_text=ref_text,
                temperature=temperature,
                top_p=top_p,
                cloud=cloud,
                max_wpm=max_wpm,
            )
        else:
            result = generate_single_audio(
                client=client,
                script=s["script"],
                output_path=s["mp3_file"],
                voice_id=voice_id,
                model=model,
                stability=stability,
                similarity=similarity,
                style=style,
                speed=speed,
                max_wpm=max_wpm,
            )
        result["script"] = str(s["txt_file"])
        results.append(result)
```

Replace with:

```python
    for s in scenes:
        if not json_output:
            print(f"Generating {s['mp3_file'].name}...", file=sys.stderr)

        if provider == "qwen3":
            result = generate_single_audio_qwen3(
                script=s["script"],
                output_path=s["mp3_file"],
                speaker=speaker,
                language=language,
                instruct=s["instruct"],
                ref_audio=ref_audio,
                ref_text=ref_text,
                temperature=temperature,
                top_p=top_p,
                cloud=cloud,
                max_wpm=max_wpm,
            )
        elif provider == "edge-tts":
            result = generate_single_audio_edge_tts(
                script=s["script"],
                output_path=s["mp3_file"],
                voice=speaker,
                rate=rate,
                pitch=pitch,
                volume=volume,
                max_wpm=max_wpm,
            )
        else:
            result = generate_single_audio(
                client=client,
                script=s["script"],
                output_path=s["mp3_file"],
                voice_id=voice_id,
                model=model,
                stability=stability,
                similarity=similarity,
                style=style,
                speed=speed,
                max_wpm=max_wpm,
            )
        result["script"] = str(s["txt_file"])
        results.append(result)
```

- [ ] **Step 9: Pass `rate`/`pitch`/`volume` at both `process_scene_directory()` call sites**

Find (the `--dry-run` call, inside `if args.dry_run:`):

```python
            results, total_duration, total_chars = process_scene_directory(
                scene_dir=scene_dir,
                dry_run=True,
                json_output=args.json,
                provider=provider,
                client=client,
                voice_id=voice_id or "",
                model=args.model,
                stability=args.stability,
                similarity=args.similarity,
                style=args.style,
                speed=args.speed,
                speaker=args.speaker,
                language=args.language,
                instruct=args.instruct,
                ref_audio=args.ref_audio,
                ref_text=args.ref_text,
                temperature=args.temperature,
                top_p=args.top_p,
                cloud=args.cloud,
            )
```

Replace with:

```python
            results, total_duration, total_chars = process_scene_directory(
                scene_dir=scene_dir,
                dry_run=True,
                json_output=args.json,
                provider=provider,
                client=client,
                voice_id=voice_id or "",
                model=args.model,
                stability=args.stability,
                similarity=args.similarity,
                style=args.style,
                speed=args.speed,
                speaker=args.speaker,
                language=args.language,
                instruct=args.instruct,
                ref_audio=args.ref_audio,
                ref_text=args.ref_text,
                temperature=args.temperature,
                top_p=args.top_p,
                cloud=args.cloud,
                rate=args.rate,
                pitch=args.pitch,
                volume=args.volume,
            )
```

Then find (the real-generation call, NOT inside `if args.dry_run:`):

```python
        results, total_duration, total_chars = process_scene_directory(
            scene_dir=scene_dir,
            dry_run=False,
            json_output=args.json,
            provider=provider,
            client=client,
            voice_id=voice_id or "",
            model=args.model,
            stability=args.stability,
            similarity=args.similarity,
            style=args.style,
            speed=args.speed,
            speaker=args.speaker,
            language=args.language,
            instruct=args.instruct,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            temperature=args.temperature,
            top_p=args.top_p,
            cloud=args.cloud,
            max_wpm=args.max_wpm,
        )
```

Replace with:

```python
        results, total_duration, total_chars = process_scene_directory(
            scene_dir=scene_dir,
            dry_run=False,
            json_output=args.json,
            provider=provider,
            client=client,
            voice_id=voice_id or "",
            model=args.model,
            stability=args.stability,
            similarity=args.similarity,
            style=args.style,
            speed=args.speed,
            speaker=args.speaker,
            language=args.language,
            instruct=args.instruct,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            temperature=args.temperature,
            top_p=args.top_p,
            cloud=args.cloud,
            rate=args.rate,
            pitch=args.pitch,
            volume=args.volume,
            max_wpm=args.max_wpm,
        )
```

(These two blocks differ in indentation — 12 vs 8 spaces — and in the trailing `max_wpm` line, so each match is unique in the file.)

- [ ] **Step 10: Make `provider_label` three-way aware (both occurrences)**

Find (appears twice in the file — once in scene-dir mode, once in single-file mode):

```python
        provider_label = "Qwen3-TTS" if provider == "qwen3" else "ElevenLabs"
```

Replace **all** occurrences with:

```python
        provider_label = {"qwen3": "Qwen3-TTS", "edge-tts": "Edge-TTS"}.get(provider, "ElevenLabs")
```

(Use `replace_all` — the fix is identical at both call sites.)

- [ ] **Step 11: Add the `edgeTts` brand-config branch**

Find:

```python
            if qwen3_cfg.get("tone") and not args.tone and not args.instruct:
                args.tone = qwen3_cfg["tone"]
        elif provider == "elevenlabs":
```

Replace with:

```python
            if qwen3_cfg.get("tone") and not args.tone and not args.instruct:
                args.tone = qwen3_cfg["tone"]
        elif provider == "edge-tts":
            edge_cfg = voice_config.get("edgeTts", {})
            if edge_cfg.get("voice") and args.speaker == "Ryan":
                args.speaker = edge_cfg["voice"]
            if edge_cfg.get("rate") and args.rate == "+0%":
                args.rate = edge_cfg["rate"]
            if edge_cfg.get("pitch") and args.pitch == "+0Hz":
                args.pitch = edge_cfg["pitch"]
            if edge_cfg.get("volume") and args.volume == "+0%":
                args.volume = edge_cfg["volume"]
        elif provider == "elevenlabs":
```

- [ ] **Step 12: Add the global Edge-TTS default-voice fallback**

Find:

```python
        elif provider == "elevenlabs":
            # Apply voice ID from brand if not explicitly provided
            if not args.voice_id and voice_config.get("voiceId") and voice_config["voiceId"] != "YOUR_VOICE_ID_HERE":
                args.voice_id = voice_config["voiceId"]

    # Resolve tone preset → instruct text for Qwen3
    if provider == "qwen3" and (args.tone or args.instruct):
```

Replace with:

```python
        elif provider == "elevenlabs":
            # Apply voice ID from brand if not explicitly provided
            if not args.voice_id and voice_config.get("voiceId") and voice_config["voiceId"] != "YOUR_VOICE_ID_HERE":
                args.voice_id = voice_config["voiceId"]

    # Default voice for Edge-TTS when none given (CLI or brand)
    if provider == "edge-tts" and args.speaker == "Ryan":
        args.speaker = "vi-VN-HoaiMyNeural"

    # Resolve tone preset → instruct text for Qwen3
    if provider == "qwen3" and (args.tone or args.instruct):
```

- [ ] **Step 13: Make the scene-dir dry-run result block three-way aware**

Find:

```python
            if provider == "elevenlabs":
                result["voice_id"] = voice_id
                result["model"] = args.model
                result["settings"] = {
                    "stability": args.stability,
                    "similarity": args.similarity,
                    "style": args.style,
                    "speed": args.speed,
                }
            else:
                result["speaker"] = args.speaker
                result["language"] = args.language
                if args.instruct:
                    result["instruct"] = args.instruct
                if args.temperature is not None:
                    result["temperature"] = args.temperature
                if args.top_p is not None:
                    result["top_p"] = args.top_p
            if args.concat:
                result["concat_output"] = args.concat
            if args.json:
                print(json.dumps(result, indent=2))
            return
```

Replace with:

```python
            if provider == "elevenlabs":
                result["voice_id"] = voice_id
                result["model"] = args.model
                result["settings"] = {
                    "stability": args.stability,
                    "similarity": args.similarity,
                    "style": args.style,
                    "speed": args.speed,
                }
            elif provider == "edge-tts":
                result["voice"] = args.speaker
                result["rate"] = args.rate
                result["pitch"] = args.pitch
                result["volume"] = args.volume
            else:
                result["speaker"] = args.speaker
                result["language"] = args.language
                if args.instruct:
                    result["instruct"] = args.instruct
                if args.temperature is not None:
                    result["temperature"] = args.temperature
                if args.top_p is not None:
                    result["top_p"] = args.top_p
            if args.concat:
                result["concat_output"] = args.concat
            if args.json:
                print(json.dumps(result, indent=2))
            return
```

- [ ] **Step 14: Make the scene-dir final result block three-way aware**

Find:

```python
        if provider == "elevenlabs":
            result["voice_id"] = voice_id
            result["model"] = args.model

        # Concat if requested
```

Replace with:

```python
        if provider == "elevenlabs":
            result["voice_id"] = voice_id
            result["model"] = args.model
        elif provider == "edge-tts":
            result["voice"] = args.speaker
            result["rate"] = args.rate
            result["pitch"] = args.pitch
            result["volume"] = args.volume

        # Concat if requested
```

- [ ] **Step 15: Make the single-file dry-run block three-way aware**

Find:

```python
        if provider == "elevenlabs":
            result["voice_id"] = voice_id
            result["model"] = args.model
            result["settings"] = {
                "stability": args.stability,
                "similarity": args.similarity,
                "style": args.style,
                "speed": args.speed,
            }
        else:
            result["speaker"] = args.speaker
            result["language"] = args.language
            if args.instruct:
                result["instruct"] = args.instruct
            if args.temperature is not None:
                result["temperature"] = args.temperature
            if args.top_p is not None:
                result["top_p"] = args.top_p
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Would generate voiceover:")
            if provider == "elevenlabs":
                print(f"  Voice ID: {voice_id}")
                print(f"  Model: {args.model}")
            else:
                print(f"  Speaker: {args.speaker}")
                print(f"  Language: {args.language}")
            print(f"  Script: {len(script)} characters")
            print(f"  Output: {output_path}")
        return
```

Replace with:

```python
        if provider == "elevenlabs":
            result["voice_id"] = voice_id
            result["model"] = args.model
            result["settings"] = {
                "stability": args.stability,
                "similarity": args.similarity,
                "style": args.style,
                "speed": args.speed,
            }
        elif provider == "edge-tts":
            result["voice"] = args.speaker
            result["rate"] = args.rate
            result["pitch"] = args.pitch
            result["volume"] = args.volume
        else:
            result["speaker"] = args.speaker
            result["language"] = args.language
            if args.instruct:
                result["instruct"] = args.instruct
            if args.temperature is not None:
                result["temperature"] = args.temperature
            if args.top_p is not None:
                result["top_p"] = args.top_p
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Would generate voiceover:")
            if provider == "elevenlabs":
                print(f"  Voice ID: {voice_id}")
                print(f"  Model: {args.model}")
            elif provider == "edge-tts":
                print(f"  Voice: {args.speaker}")
                print(f"  Rate: {args.rate}, Pitch: {args.pitch}, Volume: {args.volume}")
            else:
                print(f"  Speaker: {args.speaker}")
                print(f"  Language: {args.language}")
            print(f"  Script: {len(script)} characters")
            print(f"  Output: {output_path}")
        return
```

- [ ] **Step 16: Add the edge-tts branch to single-file generation + its final result block**

Find:

```python
    if provider == "qwen3":
        result = generate_single_audio_qwen3(
            script=script,
            output_path=output_path,
            speaker=args.speaker,
            language=args.language,
            instruct=args.instruct,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            temperature=args.temperature,
            top_p=args.top_p,
            cloud=args.cloud,
            max_wpm=args.max_wpm,
        )
    else:
        result = generate_single_audio(
            client=client,
            script=script,
            output_path=output_path,
            voice_id=voice_id,
            model=args.model,
            stability=args.stability,
            similarity=args.similarity,
            style=args.style,
            speed=args.speed,
            max_wpm=args.max_wpm,
        )

    result["mode"] = "single"
    result["provider"] = provider
    if provider == "elevenlabs":
        result["voice_id"] = voice_id
        result["model"] = args.model
```

Replace with:

```python
    if provider == "qwen3":
        result = generate_single_audio_qwen3(
            script=script,
            output_path=output_path,
            speaker=args.speaker,
            language=args.language,
            instruct=args.instruct,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            temperature=args.temperature,
            top_p=args.top_p,
            cloud=args.cloud,
            max_wpm=args.max_wpm,
        )
    elif provider == "edge-tts":
        result = generate_single_audio_edge_tts(
            script=script,
            output_path=output_path,
            voice=args.speaker,
            rate=args.rate,
            pitch=args.pitch,
            volume=args.volume,
            max_wpm=args.max_wpm,
        )
    else:
        result = generate_single_audio(
            client=client,
            script=script,
            output_path=output_path,
            voice_id=voice_id,
            model=args.model,
            stability=args.stability,
            similarity=args.similarity,
            style=args.style,
            speed=args.speed,
            max_wpm=args.max_wpm,
        )

    result["mode"] = "single"
    result["provider"] = provider
    if provider == "elevenlabs":
        result["voice_id"] = voice_id
        result["model"] = args.model
    elif provider == "edge-tts":
        result["voice"] = args.speaker
        result["rate"] = args.rate
        result["pitch"] = args.pitch
        result["volume"] = args.volume
```

- [ ] **Step 17: Verify `--help` and regression-check the other two providers' dry-run paths**

Run: `python tools/voiceover.py --help`
Expected: no traceback, prints usage including `edge-tts` in the `--provider` choices line.

Run: `python tools/voiceover.py --provider elevenlabs --script tools/requirements.txt --output /tmp/vo_el.mp3 --dry-run --json`
Expected: valid JSON, `"provider": "elevenlabs"`, no traceback (proves the ElevenLabs branch is untouched by the three-way refactor — no real API key needed since this only exercises the dry-run/argparse path, and `--script` just needs any readable text file).

Run: `python tools/voiceover.py --provider qwen3 --script tools/requirements.txt --output /tmp/vo_qw.mp3 --dry-run --json`
Expected: valid JSON, `"provider": "qwen3"`, `"speaker": "Ryan"`, no traceback.

- [ ] **Step 18: Verify Edge-TTS dry-run output**

Run: `python tools/voiceover.py --provider edge-tts --script tools/requirements.txt --output /tmp/vo_edge.mp3 --dry-run --json`
Expected: valid JSON containing `"provider": "edge-tts"`, `"voice": "vi-VN-HoaiMyNeural"` (the fallback default kicking in with no `--speaker` given), `"rate": "+0%"`, `"pitch": "+0Hz"`, `"volume": "+0%"`.

- [ ] **Step 19: Verify real single-file generation through `voiceover.py`**

Create a small Vietnamese test script and generate through the full `voiceover.py` path:
```bash
printf 'Chào mừng bạn đến với bộ công cụ sản xuất video. Đây là bản kiểm tra giọng đọc tiếng Việt.' > /tmp/vo_test_script.txt
python tools/voiceover.py --provider edge-tts --script /tmp/vo_test_script.txt --output /tmp/vo_test.mp3 --json
```
Expected: JSON with `"success": true`, `"provider": "edge-tts"`, `"voice": "vi-VN-HoaiMyNeural"`, `"duration_seconds"` a positive number, `"wpm"`/`"pacing"` present.

- [ ] **Step 20: Verify scene-dir mode with `--max-wpm`**

```bash
mkdir -p /tmp/vo_scenes
printf 'Cảnh một. Giới thiệu về sản phẩm mới của chúng tôi.' > /tmp/vo_scenes/01_intro.txt
printf 'Cảnh hai. Các tính năng chính bao gồm tốc độ và độ chính xác cao.' > /tmp/vo_scenes/02_features.txt
python tools/voiceover.py --provider edge-tts --scene-dir /tmp/vo_scenes --max-wpm 165 --json
```
Expected: JSON with `"mode": "per_scene"`, `"provider": "edge-tts"`, a `"scenes"` array of length 2, each with `"success": true` and a `.mp3` written next to its `.txt` (i.e. `/tmp/vo_scenes/01_intro.mp3` and `/tmp/vo_scenes/02_features.mp3` both exist).

- [ ] **Step 21: Confirm `sync_timing.py` compatibility (no code change expected)**

`sync_timing.py` reads `duration_seconds`/`wpm` fields from `voiceover.py`'s JSON output — fields already produced identically for every provider by `_apply_pacing_qc` (ElevenLabs path) and each provider module's own pacing block (Qwen3/Edge-TTS). Confirm by inspecting the Step 19 JSON output already contains `duration_seconds` and `wpm` at the top level — no further action needed if so. If either field is missing, that's a bug in Task 1/Task 2 to fix before proceeding (do not modify `sync_timing.py` itself; it's provider-agnostic by design).

- [ ] **Step 22: Commit**

```bash
git add tools/voiceover.py
git commit -m "$(cat <<'EOF'
ADD: wire Edge-TTS as a third voiceover.py provider

Adds --provider edge-tts alongside elevenlabs/qwen3: argparse choices,
--rate/--pitch/--volume flags, a generate_single_audio_edge_tts()
wrapper, and a three-way branch through the scene-dir loop, single-file
generation, dry-run/JSON output, and brand voice-config resolution.
--speaker is reused as the Edge-TTS voice name (default vi-VN-HoaiMyNeural).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Brand voice config (`brands/default/voice.json`)

**Files:**
- Modify: `brands/default/voice.json`
- Create (temporary, deleted at the end of this task): `brands/_edgetts_plan_test/brand.json`, `brands/_edgetts_plan_test/voice.json`

**Interfaces:**
- Consumes: the `elif provider == "edge-tts":` brand-resolution branch added in Task 2 Step 11, which reads `voice_config.get("edgeTts", {})` with keys `voice`/`rate`/`pitch`/`volume`.

- [ ] **Step 1: Add the `edgeTts` block to the default brand**

In `brands/default/voice.json`, find:

```json
  "qwen3": {
    "speaker": "Ryan",
    "language": "Auto",
    "tone": "",
    "instruct": "",
    "clone": null
  }
}
```

Replace with:

```json
  "qwen3": {
    "speaker": "Ryan",
    "language": "Auto",
    "tone": "",
    "instruct": "",
    "clone": null
  },
  "edgeTts": {
    "voice": "vi-VN-HoaiMyNeural",
    "rate": "+0%",
    "pitch": "+0Hz",
    "volume": "+0%"
  }
}
```

- [ ] **Step 2: Verify the JSON is still valid**

Run: `python -c "import json; json.load(open('brands/default/voice.json')); print('valid JSON')"`
Expected: prints `valid JSON`, no traceback.

- [ ] **Step 3: Create a throwaway test brand with a deliberately distinct voice**

This proves the brand-config code path actually reads `edgeTts` from the JSON, rather than just always falling back to the hardcoded default (which happens to also be `vi-VN-HoaiMyNeural`, so testing against `brands/default` alone wouldn't distinguish "brand config worked" from "fallback kicked in regardless").

Create `brands/_edgetts_plan_test/brand.json`:
```json
{
  "name": "Edge TTS Plan Test (throwaway, safe to delete)"
}
```

Create `brands/_edgetts_plan_test/voice.json`:
```json
{
  "edgeTts": {
    "voice": "en-US-AriaNeural",
    "rate": "-10%",
    "pitch": "+2Hz",
    "volume": "+5%"
  }
}
```

- [ ] **Step 4: Verify brand config overrides CLI defaults**

Run:
```bash
python tools/voiceover.py --provider edge-tts --brand _edgetts_plan_test --script tools/requirements.txt --output /tmp/vo_brand_test.mp3 --dry-run --json
```
Expected: JSON containing `"voice": "en-US-AriaNeural"`, `"rate": "-10%"`, `"pitch": "+2Hz"`, `"volume": "+5%"` — proving these came from `brands/_edgetts_plan_test/voice.json`, not the CLI/global defaults (`vi-VN-HoaiMyNeural` / `+0%` / `+0Hz` / `+0%`).

Then run without `--brand` to confirm the global default still applies on its own:
```bash
python tools/voiceover.py --provider edge-tts --script tools/requirements.txt --output /tmp/vo_nobrand_test.mp3 --dry-run --json
```
Expected: JSON containing `"voice": "vi-VN-HoaiMyNeural"`, `"rate": "+0%"`.

- [ ] **Step 5: Delete the throwaway test brand**

```bash
rm -rf brands/_edgetts_plan_test
```
Expected: `git status` no longer lists `brands/_edgetts_plan_test` (it was untracked, so this fully removes it — nothing to unstage).

- [ ] **Step 6: Commit**

```bash
git add brands/default/voice.json
git commit -m "$(cat <<'EOF'
ADD: edgeTts block to default brand voice config

Sets the default brand's Edge-TTS voice to vi-VN-HoaiMyNeural, matching
the toolkit-wide fallback added in voiceover.py.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Docs and registry updates

**Files:**
- Modify: `CLAUDE.md`
- Modify: `_internal/toolkit-registry.json`

**Interfaces:** None (documentation only — no code consumes these).

- [ ] **Step 1: Update the top-level capabilities bullet in `CLAUDE.md`**

Find:

```
- AI voiceover generation with ElevenLabs or Qwen3-TTS
```

Replace with:

```
- AI voiceover generation with ElevenLabs, Qwen3-TTS, or Edge TTS (free, incl. Vietnamese)
```

- [ ] **Step 2: Add an Edge-TTS example to the "Voiceover Generation" section**

Find:

```
### Voiceover Generation

```bash
# Per-scene generation (recommended)
python tools/voiceover.py --scene-dir public/audio/scenes --json

# Using Qwen3-TTS (self-hosted, free alternative to ElevenLabs)
python tools/voiceover.py --provider qwen3 --tone warm --scene-dir public/audio/scenes --json

# Single file (legacy)
python tools/voiceover.py --script SCRIPT.md --output out.mp3
```
```

Replace with:

```
### Voiceover Generation

```bash
# Per-scene generation (recommended)
python tools/voiceover.py --scene-dir public/audio/scenes --json

# Using Qwen3-TTS (self-hosted, free alternative to ElevenLabs)
python tools/voiceover.py --provider qwen3 --tone warm --scene-dir public/audio/scenes --json

# Using Edge TTS (free, no API key — Vietnamese voices: vi-VN-HoaiMyNeural, vi-VN-NamMinhNeural)
python tools/voiceover.py --provider edge-tts --scene-dir public/audio/scenes --json
python tools/voiceover.py --provider edge-tts --speaker vi-VN-NamMinhNeural --rate +10% --script SCRIPT.md --output out.mp3

# Single file (legacy)
python tools/voiceover.py --script SCRIPT.md --output out.mp3
```
```

(Note: the outer ` ```bash ... ``` ` fence is unchanged — only the lines inside it gain the two new Edge-TTS example lines.)

- [ ] **Step 3: Update the `voiceover` tool entry in the registry**

In `_internal/toolkit-registry.json`, find:

```json
    "voiceover": {
      "path": "tools/voiceover.py",
      "description": "Generate TTS voiceovers using ElevenLabs or Qwen3-TTS",
      "usage": "python tools/voiceover.py --script SCRIPT.md --output out.mp3",
      "status": "stable",
      "created": "2025-12-08",
      "updated": "2026-06-09",
      "options": {
        "pacingQC": "results include wpm + pacing label (fast/slow/ok); --max-wpm clamps fast takes with pitch-preserving atempo (floor 0.85x)"
      }
    },
```

Replace with:

```json
    "voiceover": {
      "path": "tools/voiceover.py",
      "description": "Generate TTS voiceovers using ElevenLabs, Qwen3-TTS, or Edge TTS (free, incl. Vietnamese)",
      "usage": "python tools/voiceover.py --script SCRIPT.md --output out.mp3",
      "status": "stable",
      "created": "2025-12-08",
      "updated": "2026-07-13",
      "options": {
        "pacingQC": "results include wpm + pacing label (fast/slow/ok); --max-wpm clamps fast takes with pitch-preserving atempo (floor 0.85x)"
      }
    },
```

- [ ] **Step 4: Verify the registry JSON is still valid**

Run: `python -c "import json; json.load(open('_internal/toolkit-registry.json')); print('valid JSON')"`
Expected: prints `valid JSON`, no traceback.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md _internal/toolkit-registry.json
git commit -m "$(cat <<'EOF'
DOCS: document Edge-TTS provider in CLAUDE.md and toolkit registry

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Final check

- [ ] Run `python tools/voiceover.py --provider edge-tts --scene-dir public/audio/scenes --json --dry-run` from a real project directory (or confirm the Task 2 Step 20 scratch run already covered the scene-dir path) to make sure nothing in this plan only works against `/tmp` fixtures.
- [ ] Skim `git log --oneline -6` and confirm four new commits exist (Task 1, Task 2, Task 3, Task 4), each with a clean `git status`.
