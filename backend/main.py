import subprocess
import sys
from pathlib import Path

URLS = ['https://www.youtube.com/watch?v=dQw4w9WgXcQ']

# Set output location
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

cmd = [
    sys.executable, '-m', 'yt_dlp',
    '--extract-audio',
    '--audio-format', 'm4a',
    '--format', 'm4a/bestaudio/best',
    '--js-runtimes', 'node',
    '--output', str(OUTPUT_DIR / '%(title)s.%(ext)s'),
    *URLS,
]

subprocess.run(cmd, check=True)
