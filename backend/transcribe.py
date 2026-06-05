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


def transcribe(audio_path: str | Path, model_size: str = "large-v3", device: str = "cuda",
               compute_type: str = "float16", diarize: bool = False,
               min_speakers: int | None = None,
               max_speakers: int | None = None) -> tuple[list[Segment], str]:
    """Transcribe audio into word-timed, optionally speaker-labelled Segments.

    Runs WhisperX: faster-whisper ASR, wav2vec2 forced alignment
    for word-level timestamps, and optional pyannote diarization for speaker
    labels.

    Args:
        audio_path: Path to the input audio file.
        model_size: faster-whisper model name, e.g. "large-v3".
        device: Compute device, "cuda" or "cpu".
        compute_type: CTranslate2 precision, e.g. "float16" or "int8" to reduce memory usage.
        diarize: Whether to run speaker diarization.
        min_speakers: Lower bound on speaker count, or None to let pyannote decide.
        max_speakers: Upper bound on speaker count, or None for no limit.

    Returns:
        A tuple (segments, language). segments is a list of Segment objects with
        start/end seconds, text, and a speaker label (None when diarization is
        off or no speaker was assigned). language is the detected language code.

    Raises:
        SystemExit: If diarize is True but HF_TOKEN is not set.
    """
    import whisperx

    audio = whisperx.load_audio(str(audio_path))

    # ASR
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

    segments = [
        Segment(seg["start"], seg["end"], seg["text"], seg.get("speaker"))
        for seg in result["segments"]
        if "start" in seg and "end" in seg
    ]
    return segments, language


if __name__ == "__main__":
    _ensure_cuda_libs()
    from dotenv import load_dotenv

    load_dotenv()
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    diarize = "--diarize" in sys.argv
    audio = Path(paths[0])
    segments, language = transcribe(audio, diarize=diarize)
    srt_path = audio.with_suffix(".srt")
    write_srt(segments, srt_path)
    print(f"Language: {language}  diarization: {'on' if diarize else 'off'}")
    print(f"Wrote {len(segments)} segments to {srt_path}")
