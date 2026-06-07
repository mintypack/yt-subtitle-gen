"""
Transcribe audio and align words
"""
import importlib.util
import os
import sys
from pathlib import Path

from subtitles import Segment, write_srt


def _ensure_cuda_libs() -> None:
    """Make pip-installed cuBLAS/cuDNN discoverable, re-execing if needed.

    CTranslate2 (the ASR backend) needs libcublas/libcudnn at runtime, but the
    dynamic loader only reads LD_LIBRARY_PATH at process startup. If the nvidia
    lib dirs are not already on the path, add them and re-exec this process once.
    """
    lib_dirs = []
    for pkg in ("nvidia.cublas", "nvidia.cudnn"):
        spec = importlib.util.find_spec(pkg)
        if spec and spec.submodule_search_locations:
            lib_dirs.append(os.path.join(spec.submodule_search_locations[0], "lib"))

    current = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if not lib_dirs or all(d in current for d in lib_dirs):
        return

    os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + [p for p in current if p])
    os.execv(sys.executable, [sys.executable, *sys.argv])


# Languages written without spaces between words
_LANGUAGES_WITHOUT_SPACES = {"ja", "zh"}

# Suffixes that end a sentence; a cue breaks after a word ending in one of these.
_SENT_END = ("。", "｡", "．", ".", "！", "!", "？", "?", "…", "‼", "⁉")
_CLOSING_PUNCT = "」』”’\"')）〕］】》"


def _ends_sentence(text: str) -> bool:
    """Return whether text ends with sentence punctuation, ignoring closing quotes."""
    return text.strip().rstrip(_CLOSING_PUNCT).endswith(_SENT_END)


def resegment(wx_segments: list[dict], language: str, max_gap: float = 0.7,
              max_chars: int | None = None, max_dur: float = 6.0) -> list[Segment]:
    """Split WhisperX segments into shorter cues using word-level timestamps.

    WhisperX groups speech into coarse segments whose granularity depends on the
    language's punctuation (long runs for spaceless languages like Japanese).
    This re-splits them from the per-word timing produced by alignment, starting
    a new cue on a silence gap, sentence-ending punctuation, a speaker change, or
    when the running cue would exceed a character or duration cap.

    Args:
        wx_segments: result["segments"] from whisperx.align (each may carry a
            "words" list of {"word", "start", "end", "speaker"} dicts).
        language: Detected language code; selects the word join and char cap.
        max_gap: Silence in seconds between two words that forces a break.
        max_chars: Max characters per cue before forcing a break; defaults to 20
            for spaceless languages, 42 otherwise.
        max_dur: Max duration in seconds per cue before forcing a break.

    Returns:
        A list of Segment objects, shorter and pause-aligned. Segments without
        word timing are passed through whole.
    """

    sep = "" if language in _LANGUAGES_WITHOUT_SPACES else " "
    if max_chars is None:
        max_chars = 20 if sep == "" else 42

    cues: list[Segment] = []        # Finished segments
    buf: list[dict] = []            # words for the cue being built

    def flush() -> None:
        text = sep.join(w["word"].strip() for w in buf).strip()
        starts = [w["start"] for w in buf if w.get("start") is not None]
        ends = [w["end"] for w in buf if w.get("end") is not None]
        if text and starts and ends:
            cues.append(Segment(starts[0], ends[-1], text, buf[0].get("speaker")))
        buf.clear()

    for seg in wx_segments:
        words = seg.get("words")
        # no per-word timing for this segment; pass it through whole
        if not words:
            # Alignment produced no word timing for this segment; keep it whole.
            flush()
            if seg.get("text", "").strip() and seg.get("start") is not None:
                cues.append(Segment(seg["start"], seg["end"], seg["text"].strip(),
                                    seg.get("speaker")))
            continue

        for w in words:
            if buf:
                prev = buf[-1]
                # Silence before this word
                gap = (w["start"] - prev["end"]
                       if w.get("start") is not None and prev.get("end") is not None else 0.0)
                # Text built so far
                cur_text = sep.join(x["word"].strip() for x in buf)
                # Get start time of the first word in the cue (should be the first word in the segment)
                cur_start = next((x["start"] for x in buf if x.get("start") is not None), None)
                # Duration of the cue so far
                cur_dur = (prev["end"] - cur_start
                           if cur_start is not None and prev.get("end") is not None else 0.0)
                ends_sentence = _ends_sentence(prev["word"])
                speaker_changed = bool(w.get("speaker") and buf[0].get("speaker")
                                       and w["speaker"] != buf[0]["speaker"])
                # Break on silence gap, sentence end, speaker change, or character/duration cap
                if (gap > max_gap or ends_sentence or speaker_changed
                        or len(cur_text) >= max_chars or cur_dur >= max_dur):
                    flush()
            buf.append(w)

    flush()
    return cues


def transcribe(audio_path: str | Path, model_size: str = "large-v3", device: str = "cuda",
               compute_type: str = "float16", diarize: bool = False,
               language: str | None = None,
               min_speakers: int | None = None,
               max_speakers: int | None = None,
               isolate_vocals: bool = False) -> tuple[list[Segment], str]:
    """Transcribe audio into word-timed, optionally speaker-labelled Segments.

    Runs WhisperX: faster-whisper ASR, wav2vec2 forced alignment
    for word-level timestamps, and optional pyannote diarization for speaker
    labels. Demucs vocal isolation can be enabled for music- or noise-heavy
    audio before ASR.

    Args:
        audio_path: Path to the input audio file.
        model_size: faster-whisper model name, e.g. "large-v3".
        device: Compute device, "cuda" or "cpu".
        compute_type: CTranslate2 precision, e.g. "float16" or "int8" to reduce memory usage.
        diarize: Whether to run speaker diarization.
        min_speakers: Lower bound on speaker count, or None to let pyannote decide.
        max_speakers: Upper bound on speaker count, or None for no limit.
        isolate_vocals: Whether to run Demucs vocal isolation before ASR. Helps
            on music- or noise-heavy audio; leave off for clean speech to save time.

    Returns:
        A tuple (segments, language). segments is a list of Segment objects with
        start/end seconds, text, and a speaker label (None when diarization is
        off or no speaker was assigned). language is the detected language code.

    Raises:
        SystemExit: If diarize is True but HF_TOKEN is not set.
    """
    import whisperx

    if isolate_vocals:
        from separate import isolate_vocals as run_demucs

        print("Isolating vocals with Demucs")
        vocals_path = run_demucs(audio_path, device=device)
        try:
            audio = whisperx.load_audio(str(vocals_path))
        finally:
            vocals_path.unlink(missing_ok=True)
    else:
        audio = whisperx.load_audio(str(audio_path))

    # ASR
    if language:
        print(f"Using language {language} for transcription")
        model = whisperx.load_model(model_size, device, compute_type=compute_type, language=language)
    else:
        model = whisperx.load_model(model_size, device, compute_type=compute_type)
    result = model.transcribe(audio, batch_size=16)
    language = result["language"]

    # Word-level timestamps via forced alignment
    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(result["segments"], align_model, metadata, audio, device,
                            return_char_alignments=False)

    # Speaker labels
    if diarize:
        from whisperx.diarize import DiarizationPipeline

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("HF_TOKEN not set (needed for diarization). Add it to backend/.env.")
        diarizer = DiarizationPipeline(token=token, device=device)
        diarize_segments = diarizer(audio, min_speakers=min_speakers, max_speakers=max_speakers)
        result = whisperx.assign_word_speakers(diarize_segments, result)

    segments = resegment(result["segments"], language)
    return segments, language


if __name__ == "__main__":
    _ensure_cuda_libs()
    from dotenv import load_dotenv

    load_dotenv()
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    diarize = "--diarize" in sys.argv
    isolate_vocals = "--isolate-vocals" in sys.argv and "--no-isolate-vocals" not in sys.argv
    audio = Path(paths[0])

    lang_index = sys.argv.index('--language') + 1 if '--language' in sys.argv else None
    if lang_index and lang_index < len(sys.argv):
        language = sys.argv[lang_index]
        segments, language = transcribe(audio, diarize=diarize, language=language,
                                        isolate_vocals=isolate_vocals)
    else:
        language = None
        segments, language = transcribe(audio, diarize=diarize,
                                        isolate_vocals=isolate_vocals)

    srt_path = audio.with_suffix(".srt")
    write_srt(segments, srt_path)
    print(f"Language: {language}  diarization: {'on' if diarize else 'off'}  "
          f"vocal isolation: {'on' if isolate_vocals else 'off'}")
    print(f"Wrote {len(segments)} segments to {srt_path}")
