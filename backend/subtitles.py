"""
Generate SRT subtitles from timed segments.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Segment:
    start: float  # seconds
    end: float  # seconds
    text: str
    speaker: str | None = None  # e.g. "SPEAKER_00" from diarization, or None


def format_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp: HH:MM:SS,mmm.

    Args:
        seconds: Time offset in seconds; negative values are clamped to zero.

    Returns:
        The timestamp string, e.g. "01:23:45,678".
    """
    if seconds < 0:
        seconds = 0.0
    ms = round(seconds * 1000)
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def format_speaker(speaker: str) -> str:
    """Turn a diarization label like "SPEAKER_00" into "Speaker 1".

    Args:
        speaker: A diarization label such as "SPEAKER_00".

    Returns:
        A human-readable label like "Speaker 1", or the input unchanged if it
        is not a recognised "SPEAKER_NN" label.
    """
    if speaker and speaker.startswith("SPEAKER_"):
        try:
            return f"Speaker {int(speaker.split('_')[1]) + 1}"
        except ValueError:
            pass
    return speaker


def to_srt(segments: List[Segment]) -> str:
    """Render timed segments as an SRT document.

    Accepts any objects exposing start, end (seconds) and text. If a segment
    has a non-empty speaker attribute, the line is prefixed with the speaker.

    Args:
        segments: Timed segments to render, numbered in order starting at 1.

    Returns:
        The full SRT document as a single string.
    """
    blocks = []
    for index, seg in enumerate(segments, start=1):
        start = format_timestamp(seg.start)
        end = format_timestamp(seg.end)
        text = seg.text.strip()
        speaker = getattr(seg, "speaker", None)
        if speaker:
            text = f"[{format_speaker(speaker)}] {text}"
        blocks.append(f"{index}\n{start} --> {end}\n{text}\n")
    return "\n".join(blocks)


def write_srt(segments: List[Segment], path: str | Path) -> None:
    """Write segments to an SRT file (UTF-8).

    Args:
        segments: Timed segments to render.
        path: Destination file path (str or os.PathLike).
    """
    Path(path).write_text(to_srt(segments), encoding="utf-8")
