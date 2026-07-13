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
import time
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


def _synthesize_with_retries(
    text: str, output_path: str, voice: str, rate: str, pitch: str, volume: str,
    retries: int, verbose: bool,
) -> str | None:
    """Synthesize, retrying transient backend failures. Returns None on success,
    or an error string once the attempts are exhausted.

    Microsoft's Edge TTS backend intermittently returns no audio for an
    otherwise valid request ("No audio was received"); a retry usually clears
    it. Left unretried, a single flaky scene silently drops out of a multi-scene
    voiceover run.

    On unrecoverable failure the output file is removed: a failed save can leave
    a ZERO-BYTE .mp3 behind, and an empty file in a scene directory is worse than
    no file — ffprobe reports no duration, Remotion renders silence rather than
    erroring, and the file's mere presence reads as success.

    Bad parameters (unknown voice, malformed rate/pitch/volume) surface as
    ValueError from edge-tts's client-side validation, before any network call.
    Those are deterministic, so they fail immediately rather than burning retries.
    """
    last_error: object = None
    for attempt in range(1, retries + 1):
        try:
            asyncio.run(_synthesize(text, output_path, voice, rate, pitch, volume))
        except ValueError as e:
            Path(output_path).unlink(missing_ok=True)
            return f"edge-tts rejected the request: {e}"
        except Exception as e:
            last_error = e
        else:
            out = Path(output_path)
            if out.exists() and out.stat().st_size > 0:
                return None
            last_error = "edge-tts reported success but wrote no audio"

        if attempt < retries:
            if verbose:
                print(
                    f"  Attempt {attempt}/{retries} failed ({last_error}) — retrying...",
                    file=sys.stderr,
                )
            time.sleep(attempt)  # 1s, then 2s

    Path(output_path).unlink(missing_ok=True)
    return f"edge-tts synthesis failed after {retries} attempts: {last_error}"


def generate_audio(
    text: str,
    output_path: str,
    voice: str = DEFAULT_VOICE,
    rate: str = "+0%",
    pitch: str = "+0Hz",
    volume: str = "+0%",
    max_wpm: float | None = None,
    verbose: bool = True,
    retries: int = 3,
) -> dict:
    """Generate one audio file using Microsoft Edge TTS.

    Pacing QC: every result includes `wpm` (words per minute) and a `pacing`
    label ('fast'/'slow'/'ok'/None). If `max_wpm` is set, takes that exceed
    it are slowed in place with pitch-preserving ffmpeg atempo (floor 0.85x).
    See tools/pacing.py.

    Transient backend failures are retried (see _synthesize_with_retries); a
    run that ultimately fails leaves no file behind.

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

    error = _synthesize_with_retries(
        text, output_path, voice, rate, pitch, volume, retries, verbose
    )
    if error:
        return {"success": False, "error": error}

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
