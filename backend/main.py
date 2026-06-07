import subprocess
import sys
from pathlib import Path
import argparse

def download_audio(url: str) -> None:
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
        url,
    ]

    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download audio from one or more YouTube URLs.')
    parser.add_argument('urls', nargs='+', help='One or more YouTube URLs')
    args = parser.parse_args()

    for url in args.urls:
        download_audio(url)
