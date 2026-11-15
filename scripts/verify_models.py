"""
HarmonyShield - Model Artifact Verification Script
Verifies presence, file size, and SHA-256 integrity of all required model weights.
Exit code 0 on complete PASS; exit code 1 on any FAIL.
"""

import os
import sys
import json
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "models" / "model_manifest.json"


def compute_sha256(file_path: Path) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest().lower()


def verify_artifacts() -> bool:
    print("=" * 70)
    print(" HarmonyShield — Model Artifact Integrity Verification")
    print("=" * 70)

    if not MANIFEST_PATH.exists():
        print(f"[-] ERROR: Manifest file not found at {MANIFEST_PATH}")
        return False

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    artifacts = manifest.get("artifacts", [])
    all_passed = True

    print(f"[*] Verifying {len(artifacts)} required model artifacts...\n")

    for art in artifacts:
        name = art["name"]
        rel_path = art["path"]
        expected_size = art["size_bytes"]
        expected_hash = art["sha256"].lower()
        full_path = PROJECT_ROOT / rel_path

        print(f"• Checking: {name}")
        print(f"  Target:   {rel_path}")

        if not full_path.exists():
            print(f"  Result:   [FAIL] File missing from disk!")
            all_passed = False
            print("-" * 70)
            continue

        actual_size = full_path.stat().st_size
        if actual_size != expected_size:
            print(f"  Result:   [FAIL] Size mismatch! (Expected {expected_size} B, found {actual_size} B)")
            all_passed = False
            print("-" * 70)
            continue

        actual_hash = compute_sha256(full_path)
        if actual_hash != expected_hash:
            print(f"  Result:   [FAIL] SHA-256 mismatch!")
            print(f"            Expected: {expected_hash}")
            print(f"            Found:    {actual_hash}")
            all_passed = False
            print("-" * 70)
            continue

        print(f"  Size:     {actual_size:,} bytes")
        print(f"  SHA-256:  {actual_hash[:16]}...{actual_hash[-16:]}")
        print(f"  Result:   [PASS] Verified successfully.")
        print("-" * 70)

    # Note on Hugging Face AST model
    print("\n• External AST Model (Plan B):")
    print("  Model ID: AI-Music-Detection/ai_music_detection_large_60s")
    print("  Status:   Managed automatically via Hugging Face Transformers cache.")
    print("=" * 70)

    if all_passed:
        print(">>> ALL MODEL ARTIFACTS VERIFIED: PASS <<<")
        return True
    else:
        print(">>> ONE OR MORE MODEL ARTIFACTS FAILED VERIFICATION: FAIL <<<")
        return False


if __name__ == "__main__":
    success = verify_artifacts()
    sys.exit(0 if success else 1)
