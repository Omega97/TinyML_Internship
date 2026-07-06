#!/usr/bin/env python3
"""Download Lc0 teacher binary + BT4 network for SARDINE labeling."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tinymlinternship.config.settings import (
    LC0_BINARY,
    LC0_NETWORK_BT4,
    TEACHER_DIR,
    TEACHER_MANIFEST,
)

CHUNK_SIZE = 1024 * 1024
USER_AGENT = "SARDINE-TinyMLInternship/0.1 (+teacher download)"

LC0_RELEASE = "v0.32.1"
LC0_WINDOWS_CPU_URL = (
    "https://github.com/LeelaChessZero/lc0/releases/download/"
    f"{LC0_RELEASE}/lc0-{LC0_RELEASE}-windows-cpu-openblas.zip"
)

BT4_URL = (
    "https://storage.lczero.org/files/networks-contrib/big-transformers/"
    "BT4-1024x15x32h-swa-6147500.pb.gz"
)
BT4_FILENAME = "BT4-1024x15x32h-swa-6147500.pb.gz"


def ensure_dirs() -> None:
    TEACHER_DIR.mkdir(parents=True, exist_ok=True)
    LC0_BINARY.parent.mkdir(parents=True, exist_ok=True)
    LC0_NETWORK_BT4.parent.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK_SIZE):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: Path, *, label: str) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip (exists): {dest.name}")
        return

    print(f"  downloading {label} …")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            while block := resp.read(CHUNK_SIZE):
                out.write(block)
                done += len(block)
                if total:
                    pct = 100 * done / total
                    print(f"\r  … {done // (1024 * 1024)} MiB / {total // (1024 * 1024)} MiB ({pct:.0f}%)", end="", flush=True)
            print()
    except (HTTPError, URLError) as exc:
        if tmp.exists():
            tmp.unlink()
        raise SystemExit(f"Download failed ({url}): {exc}") from exc
    tmp.replace(dest)
    print(f"  saved: {dest} ({dest.stat().st_size // (1024 * 1024)} MiB)")


def extract_lc0_zip(zip_path: Path, dest_dir: Path) -> Path:
    print(f"  extracting {zip_path.name} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    exe = dest_dir / "lc0.exe"
    if not exe.exists():
        candidates = list(dest_dir.rglob("lc0.exe"))
        if not candidates:
            raise SystemExit(f"lc0.exe not found inside {zip_path}")
        exe = candidates[0]
    return exe.resolve()


def write_manifest(lc0_exe: Path, network: Path, *, lc0_zip: Path) -> None:
    manifest = {
        "teacher": "lc0-bt4",
        "lc0_release": LC0_RELEASE,
        "lc0_binary": str(lc0_exe.relative_to(TEACHER_DIR.parent.parent)),
        "lc0_zip": str(lc0_zip.relative_to(TEACHER_DIR.parent.parent)) if lc0_zip.exists() else None,
        "network": str(network.relative_to(TEACHER_DIR.parent.parent)),
        "network_url": BT4_URL,
        "label_formula": "expected_reward = (W - L) / 1000  # UCI wdl permille",
        "uci_nodes_labeling": 1,
    }
    if network.exists():
        manifest["network_sha256"] = sha256_file(network)
        manifest["network_size_bytes"] = network.stat().st_size
    if lc0_exe.exists():
        manifest["lc0_binary_sha256"] = sha256_file(lc0_exe)
    TEACHER_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  manifest: {TEACHER_MANIFEST}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-only", action="store_true", help="Skip lc0 binary download")
    parser.add_argument("--binary-only", action="store_true", help="Skip network download")
    args = parser.parse_args()

    ensure_dirs()
    lc0_zip = TEACHER_DIR / "lc0" / f"lc0-{LC0_RELEASE}-windows-cpu-openblas.zip"

    print("SARDINE teacher install →", TEACHER_DIR)

    if not args.network_only:
        download(LC0_WINDOWS_CPU_URL, lc0_zip, label="lc0 windows-cpu-openblas")
        lc0_exe = extract_lc0_zip(lc0_zip, LC0_BINARY.parent)
    else:
        lc0_exe = LC0_BINARY

    if not args.binary_only:
        download(BT4_URL, LC0_NETWORK_BT4, label="BT4 network")

    if not lc0_exe.exists():
        raise SystemExit(f"lc0 binary missing: {lc0_exe}")
    if not LC0_NETWORK_BT4.exists():
        raise SystemExit(f"network missing: {LC0_NETWORK_BT4}")

    write_manifest(lc0_exe, LC0_NETWORK_BT4, lc0_zip=lc0_zip)
    print("Done. Verify: py -3.12 scripts/smoke_test_teacher.py")


if __name__ == "__main__":
    main()