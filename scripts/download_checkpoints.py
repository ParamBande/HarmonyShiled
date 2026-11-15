"""
HarmonyShield - Checkpoint Downloader & Setup Utility
Downloads the 5 ResNet18 LOGO checkpoints from GitHub Release assets
and verifies their SHA-256 integrity using verify_models.py.
"""

import os
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = PROJECT_ROOT / "experiments" / "resnet18_generalization" / "checkpoints"

GITHUB_REPO = "ParamBande/HarmonyShiled"
RELEASE_TAG = "v1.0.0-review-ready"
BASE_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}"

CHECKPOINTS = [
    "LOGO_audioldm2_model.pt",
    "LOGO_MusicGen_medium_model.pt",
    "LOGO_musicldm_model.pt",
    "LOGO_mustango_model.pt",
    "LOGO_stable_audio_open_model.pt",
]


def download_file(url: str, dest_path: Path):
    req = urllib.request.Request(url, headers={"User-Agent": "HarmonyShield-Setup"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_file:
        total_size = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024 # 1 MB
        last_pct = -1

        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            pct = int(downloaded * 100 / total_size) if total_size else 0
            if pct != last_pct and pct % 20 == 0:
                print(f"  Downloaded: {pct}% ({downloaded / (1024 * 1024):.1f} / {total_size / (1024 * 1024):.1f} MB)")
                last_pct = pct


def download_checkpoints():
    print("=" * 70)
    print(" HarmonyShield — Checkpoint Downloader")
    print(f" Source: https://github.com/{GITHUB_REPO}/releases/tag/{RELEASE_TAG}")
    print(f" Target: {CKPT_DIR}")
    print("=" * 70)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    for ckpt in CHECKPOINTS:
        target_path = CKPT_DIR / ckpt
        if target_path.exists():
            print(f"• [EXISTS] {ckpt} already present ({target_path.stat().st_size:,} bytes). Skipping.")
            continue

        url = f"{BASE_URL}/{ckpt}"
        print(f"• Downloading {ckpt}...")
        try:
            download_file(url, target_path)
            print("  [DONE] Download complete.")
        except Exception as e:
            print(f"\n[-] Failed to download {ckpt} from {url}: {e}")
            print("    Please ensure the release assets are published or copy the files manually.")
            return False

    print("\nRunning integrity verification...")
    import subprocess
    result = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "verify_models.py")])
    return result.returncode == 0


if __name__ == "__main__":
    success = download_checkpoints()
    sys.exit(0 if success else 1)
