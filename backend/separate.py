"""
Isolate vocals from audio using Demucs.
"""
import gc
import os
import tempfile
from pathlib import Path


def isolate_vocals(audio_path: str | Path, device: str = "cuda",
                   model_name: str = "htdemucs",
                   out_path: str | Path | None = None,
                   progress: bool = False) -> Path:
    """Separate the vocal stem from an audio file with Demucs.

    Removes music, drums, and other accompaniment so the ASR model sees a
    cleaner speech signal. Most useful for music- or noise-heavy content (movie
    trailers, gameplay); on already-clean speech it adds compute for little gain.

    The Demucs model is loaded and freed within this call, releasing its GPU
    memory before the caller loads the ASR model on the same device.

    Args:
        audio_path: Path to the input audio file.
        device: Compute device, "cuda" or "cpu".
        model_name: Demucs model name; "htdemucs" is the v4 hybrid transformer.
        out_path: Where to write the vocal stem. When None, a temporary WAV is
            created and the caller is responsible for deleting it.
        progress: Whether Demucs should print chunk-level progress.

    Returns:
        Path to the WAV file holding only the vocal stem.
    """
    import torch
    from demucs.apply import apply_model
    from demucs.audio import save_audio
    from demucs.pretrained import get_model
    from demucs.separate import load_track

    model = get_model(model_name)
    model.cpu()
    model.eval()

    try:
        wav = load_track(Path(audio_path), model.audio_channels, model.samplerate)
    except SystemExit as exc:
        raise RuntimeError(f"Could not load audio file: {audio_path}") from exc

    # Normalize to zero mean and unit variance the way the Demucs CLI does; the
    # shift and scale are undone on the output so the vocal levels are preserved.
    ref = wav.mean(0)
    wav -= ref.mean()
    scale = ref.std().clamp_min(1e-8)
    wav /= scale

    with torch.inference_mode():
        sources = apply_model(model, wav[None], device=device, shifts=1, split=True,
                              overlap=0.25, progress=progress)[0]
    sources *= scale
    sources += ref.mean()

    vocals = sources[model.sources.index("vocals")]

    if out_path is None:
        fd, tmp = tempfile.mkstemp(suffix=".vocals.wav")
        os.close(fd)
        out_path = Path(tmp)
    else:
        out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_audio(vocals, out_path, samplerate=model.samplerate)

    # Release the model and intermediate tensors before the ASR model loads.
    del model, wav, sources, vocals
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()

    return out_path


if __name__ == "__main__":
    import sys

    audio = Path(sys.argv[1])
    out = audio.with_suffix(".vocals.wav")
    isolate_vocals(audio, out_path=out)
    print(f"Wrote isolated vocals to {out}")
