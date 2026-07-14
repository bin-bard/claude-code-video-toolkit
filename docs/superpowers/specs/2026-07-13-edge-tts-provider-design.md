# Edge TTS Provider — Design Spec

**Date:** 2026-07-13
**Status:** Approved for planning

## Problem

The toolkit's voiceover generation (`tools/voiceover.py`) supports two TTS providers:
ElevenLabs (paid, cloud) and Qwen3-TTS (self-hosted via RunPod/Modal). Neither has good
Vietnamese support — Qwen3-TTS's `SUPPORTED_LANGUAGES` list excludes Vietnamese entirely.
Microsoft Edge TTS (via the `edge-tts` Python package) is free, requires no API key or
cloud GPU, and includes high-quality Vietnamese neural voices (`vi-VN-HoaiMyNeural`,
`vi-VN-NamMinhNeural`), plus ~300 other voices across languages.

## Goal

Add Edge TTS as a third `voiceover.py` provider, following the existing provider pattern
(ElevenLabs / Qwen3-TTS), so it participates in the full pipeline: `--scene-dir` batch
generation, `--max-wpm` pacing QC, `sync_timing.py`, and brand voice config.

## Scope decisions (from brainstorming)

1. **General-purpose, not Vietnamese-only.** Support any edge-tts voice via `--speaker`,
   defaulting to a Vietnamese voice. Not hardcoded to just 2 Vietnamese speakers.
2. **Expose `--rate` / `--pitch` / `--volume`.** These map directly to edge-tts's native
   parameters (e.g. `+10%`, `-5Hz`) and are additive to the existing `--max-wpm` pacing
   clamp (which operates after generation via `atempo`).
3. **Brand config now.** Add an `edgeTts` block to `voice.json`, read the same way as the
   existing `qwen3` block.

## Architecture

Mirrors the Qwen3-TTS integration: a dedicated module owns the provider-specific API
calls and returns a result dict in the same shape used by every other provider
(`success`, `output`, `duration_seconds`, `duration_frames_30fps`, `wpm`, `pacing`).
`voiceover.py` stays the single CLI entry point and orchestrator; `pacing.py` stays
provider-agnostic and unchanged.

### 1. New module: `tools/edgetts.py`

**Naming note:** the file must NOT be named `edge_tts.py`. `voiceover.py` prepends
`tools/` to `sys.path`, so a same-named local module would shadow the real `edge_tts`
pip package for any script doing `import edge_tts` from within `tools/`. Filename:
`tools/edgetts.py` (no underscore).

```python
def generate_audio(
    text: str,
    output_path: str,
    voice: str = "vi-VN-HoaiMyNeural",
    rate: str = "+0%",
    pitch: str = "+0Hz",
    volume: str = "+0%",
    max_wpm: float | None = None,
    verbose: bool = True,
) -> dict:
    """Generate one audio file via Microsoft Edge TTS. Returns a result dict
    matching the shape used by qwen3_tts.generate_audio()."""
```

Behavior:
- Lazy-imports the `edge_tts` package; on `ImportError`, prints a install hint
  (`pip install edge-tts`) and returns `{"success": False, "error": ...}` — mirrors
  `voiceover.py`'s existing `_get_elevenlabs_imports()` error style, but simpler
  (single free option, no "or use X instead" branching needed).
- Runs `edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, volume=volume).save(output_path)`
  via `asyncio.run(...)`.
- Probes duration with ffprobe (reuse the same subprocess pattern already used in
  `voiceover.py` / `pacing.py`).
- Applies pacing QC internally (matches Qwen3's `generate_audio`, which self-applies
  `clamp_pace`/`pace_label` rather than leaving it to the caller): if `max_wpm` is set,
  calls `pacing.clamp_pace`; always sets `wpm`/`pacing` via `pacing.pace_label`.
- Catches edge-tts exceptions (bad voice name, network error) and returns
  `{"success": False, "error": str(e)}`.

```python
async def list_voices(language: str | None = None) -> list[dict]:
    """Fetch the live edge-tts voice catalogue, optionally filtered by locale
    prefix (e.g. language="vi" matches vi-VN-*)."""
```

Standalone CLI (matching `qwen3_tts.py`'s self-contained usability):
```bash
python tools/edgetts.py --text "Xin chào" --voice vi-VN-HoaiMyNeural --output out.mp3
python tools/edgetts.py --list-voices --language vi
```

A small `VIETNAMESE_VOICES` dict (`HoaiMy`: female, `NamMinh`: male) is included as a
convenience reference for `--list-voices` output and docs — not a hard restriction on
`--speaker`, which accepts any edge-tts voice name.

### 2. Changes to `voiceover.py`

- **`--provider` choices** → add `"edge-tts"`.
- **`--speaker`** (existing flag, currently Qwen3-only) is reused for the edge-tts voice
  name. Its argparse default stays `"Ryan"`. After parsing, if
  `provider == "edge-tts" and args.speaker == "Ryan"` (i.e. untouched default), swap to
  `"vi-VN-HoaiMyNeural"` — same sentinel-swap trick already used for brand defaults
  (see `main()`'s existing `qwen3_cfg.get("speaker") and args.speaker == "Ryan"` check).
- **New flags:** `--rate` (default `"+0%"`), `--pitch` (default `"+0Hz"`), `--volume`
  (default `"+0%"`) — edge-tts-specific, passed straight through.
- **`generate_single_audio_edge_tts()`** — new wrapper function, same shape as
  `generate_single_audio_qwen3()`, delegates to `edgetts.generate_audio()`.
- **`process_scene_directory()`** — add a third branch in the serial fallback loop
  (`provider == "edge-tts"` → call the new wrapper). No batch path: edge-tts has no
  batch API, so it always takes the serial per-scene loop, same as ElevenLabs today.
  `can_batch_qwen3`'s condition is already `provider == "qwen3"`-gated, so no change
  needed there.
- **Brand config (`main()`)** — add an `elif provider == "edge-tts":` branch alongside
  the existing `qwen3`/`elevenlabs` brand-resolution blocks: read
  `voice_config.get("edgeTts", {})` and apply `voice`/`rate`/`pitch`/`volume` defaults
  when the corresponding CLI args are still at their defaults.
- **Dry-run / JSON result blocks** — extend the existing `if provider == "elevenlabs": ... else: ...`
  conditionals (there are three: single dry-run, scene-dir dry-run, final result) to be
  three-way aware, emitting `voice`/`rate`/`pitch`/`volume` for edge-tts instead of the
  Qwen3 fields.
- **`provider_label` strings** (`"Qwen3-TTS"`/`"ElevenLabs"`) — extend to include
  `"Edge-TTS"`.
- Update the module docstring and argparse epilog with Edge TTS examples.

### 3. `brands/default/voice.json`

Add an `edgeTts` block, same style as the existing `qwen3` block:

```json
"edgeTts": {
  "voice": "vi-VN-HoaiMyNeural",
  "rate": "+0%",
  "pitch": "+0Hz",
  "volume": "+0%"
}
```

Other brand profiles are left untouched (they don't have a `qwen3` block either — only
`brands/default/voice.json` currently carries one).

### 4. `tools/requirements.txt`

Add `edge-tts>=6.1.0` as a normal (uncommented) dependency, alongside `elevenlabs`. It's
lightweight (no torch/heavy deps), free, and needs no signup — consistent with the
toolkit's "most features are free" default.

### 5. Docs / registry

- `_internal/toolkit-registry.json` → update the `voiceover` tool entry's `description`
  and `updated` date to mention Edge TTS / Vietnamese.
- `CLAUDE.md` → add an Edge TTS example block under "Voiceover Generation", and mention
  it in the top-level capabilities list.

## Error handling

- Missing `edge-tts` package → friendly `pip install edge-tts` message, `sys.exit(1)`
  in CLI mode / `{"success": False, "error": ...}` in library mode.
- Invalid voice name / network failure from `edge_tts.Communicate` → caught, surfaced as
  `{"success": False, "error": str(e)}`, consistent with how Qwen3/ElevenLabs failures
  propagate through `process_scene_directory()`.
- Malformed `--rate`/`--pitch`/`--volume` strings are validated by edge-tts itself; its
  exception message is surfaced as-is (no extra validation layer — YAGNI).

## Testing plan

1. `pip install edge-tts`, then `python tools/edgetts.py --list-voices --language vi` —
   confirm live voice list resolves and includes `vi-VN-HoaiMyNeural` / `vi-VN-NamMinhNeural`.
2. `python tools/edgetts.py --text "Xin chào, đây là bài kiểm tra." --voice vi-VN-HoaiMyNeural --output /tmp/test.mp3`
   — inspect duration/wpm output.
3. `python tools/voiceover.py --provider edge-tts --script test.txt --output out.mp3 --json`
   — full path through `voiceover.py`.
4. `python tools/voiceover.py --provider edge-tts --scene-dir public/audio/scenes --json --max-wpm 165`
   — per-scene batch + pacing clamp.
5. Confirm `sync_timing.py` consumes the JSON output unchanged (it's already
   provider-agnostic — reads `duration_seconds` / `wpm` fields common to all providers).

## Out of scope (deferred)

- Tone/instruct presets for Edge TTS (Qwen3 has `--tone`/`--instruct`; edge-tts has no
  equivalent natural-language style control — only rate/pitch/volume).
- Voice cloning (edge-tts doesn't support it).
- `edgeTts` blocks in brand profiles other than `default`.
