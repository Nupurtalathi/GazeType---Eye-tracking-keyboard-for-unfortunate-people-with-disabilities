"""
Model Downloader Utility
Ensures shape_predictor_68_face_landmarks.dat is present in the models/ directory.
If absent, streams and decompresses it with proper User-Agent headers.
"""

import os
import sys
import bz2
import urllib.request
import logging

logger = logging.getLogger(__name__)

MODEL_FILENAME = "shape_predictor_68_face_landmarks.dat"
ARCHIVE_FILENAME = "shape_predictor_68_face_landmarks.dat.bz2"
PRIMARY_URL = "https://raw.githubusercontent.com/davisking/dlib-models/master/shape_predictor_68_face_landmarks.dat.bz2"
FALLBACK_URL = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"


def get_default_model_path():
    """Returns absolute path to shape_predictor_68_face_landmarks.dat in the models folder."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "models", MODEL_FILENAME)


def ensure_model_exists(target_path=None, progress_callback=None):
    """
    Checks if model exists at target_path. If not, downloads and extracts it.
    Returns the absolute path to the valid model file.
    """
    if target_path is None:
        target_path = get_default_model_path()

    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir, exist_ok=True)

    if os.path.exists(target_path) and os.path.getsize(target_path) > 50_000_000:
        return target_path

    archive_path = os.path.join(target_dir, ARCHIVE_FILENAME)

    print(f"[Model Downloader] Target model path: {target_path}")
    print(f"[Model Downloader] Downloading {ARCHIVE_FILENAME}...")

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    for url in [PRIMARY_URL, FALLBACK_URL]:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response, open(archive_path, 'wb') as out_file:
                total_size = int(response.headers.get('Content-Length', 64000000))
                downloaded = 0
                chunk_size = 256 * 1024

                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    pct = min(100.0, downloaded * 100.0 / total_size)
                    print(f"\rDownloading model: {pct:.1f}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)", end="", flush=True)
                    if progress_callback:
                        progress_callback(pct)

            print("\n[Model Downloader] Download complete. Decompressing bz2...")
            break
        except Exception as e:
            print(f"\n[Model Downloader] Attempt with {url} failed: {e}. Trying next...")
            if os.path.exists(archive_path):
                try:
                    os.remove(archive_path)
                except OSError:
                    pass
    else:
        raise RuntimeError("Failed to download model from all mirrors.")

    try:
        with bz2.BZ2File(archive_path, 'rb') as source, open(target_path, 'wb') as dest:
            chunk_size = 1024 * 1024
            while True:
                chunk = source.read(chunk_size)
                if not chunk:
                    break
                dest.write(chunk)

        if os.path.exists(archive_path):
            os.remove(archive_path)

        size_mb = os.path.getsize(target_path) // (1024 * 1024)
        print(f"[Model Downloader] Successfully unpacked to {target_path} ({size_mb} MB).")
        return target_path

    except Exception as e:
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
            except OSError:
                pass
        raise RuntimeError(f"Failed to decompress {archive_path}: {e}")


if __name__ == "__main__":
    ensure_model_exists()
