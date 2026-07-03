#!/usr/bin/env python3
"""Download a curated Lc0 training-data subset (~1–2 GB) from storage.lczero.org."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tinymlinternship.config.settings import (
    LC0_CHUNKS_DIR,
    LC0_MANIFEST,
    LC0_RAW_DIR,
    LC0_TARS_DIR,
    PROJECT_ROOT,
)
from tinymlinternship.data.lc0_shards import (
    DEFAULT_SHARDS,
    EXTRA_SHARDS,
    LC0_BASE_URL,
    Lc0Shard,
    select_shards,
)

CHUNK_SIZE = 1024 * 1024  # 1 MiB
USER_AGENT = "SARDINE-TinyMLInternship/0.1 (+https://github.com/LeelaChessZero/lc0; training-data subset)"


def ensure_dirs() -> None:
    LC0_RAW_DIR.mkdir(parents=True, exist_ok=True)
    LC0_TARS_DIR.mkdir(parents=True, exist_ok=True)
    LC0_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(CHUNK_SIZE)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_manifest() -> dict:
    if LC0_MANIFEST.exists():
        return json.loads(LC0_MANIFEST.read_text(encoding="utf-8"))
    return {"base_url": LC0_BASE_URL, "shards": {}}


def save_manifest(manifest: dict) -> None:
    LC0_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def download_shard(shard: Lc0Shard, *, force: bool = False) -> Path:
    """Download one tar with HTTP Range resume and size verification."""
    dest = LC0_TARS_DIR / shard.name
    if dest.exists() and not force:
        size = dest.stat().st_size
        if size == shard.size_bytes:
            print(f"  skip (complete): {shard.name}")
            return dest
        if size == 0:
            dest.unlink()
            size = 0
        elif size > shard.size_bytes:
            print(f"  re-download (oversized): {shard.name}")
            dest.unlink()
            size = 0

    offset = dest.stat().st_size if dest.exists() else 0
    if offset and offset < shard.size_bytes:
        print(f"  resume: {shard.name} ({offset:,} / {shard.size_bytes:,} bytes)")
    else:
        print(f"  fetch: {shard.name} ({shard.size_bytes:,} bytes)")

    headers: dict[str, str] = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"

    request = Request(shard.url, headers=headers)
    try:
        with urlopen(request, timeout=120) as response:
            status = getattr(response, "status", response.getcode())
            if offset and status not in (206, 200):
                print(f"  server did not resume; restarting {shard.name}")
                dest.unlink(missing_ok=True)
                offset = 0
                request = Request(shard.url, headers={"User-Agent": USER_AGENT})
                response = urlopen(request, timeout=120)

            mode = "ab" if offset else "wb"
            downloaded = offset
            next_report = ((downloaded // (50 * CHUNK_SIZE)) + 1) * 50 * CHUNK_SIZE
            with dest.open(mode) as handle:
                while True:
                    block = response.read(CHUNK_SIZE)
                    if not block:
                        break
                    handle.write(block)
                    handle.flush()
                    downloaded += len(block)
                    if downloaded >= next_report:
                        pct = 100.0 * downloaded / shard.size_bytes
                        print(
                            f"    {shard.name}: {downloaded:,} / {shard.size_bytes:,} "
                            f"({pct:.1f}%)"
                        )
                        next_report += 50 * CHUNK_SIZE
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Download failed for {shard.name}: {exc}") from exc

    final_size = dest.stat().st_size
    if final_size != shard.size_bytes:
        raise RuntimeError(
            f"Size mismatch for {shard.name}: got {final_size:,}, expected {shard.size_bytes:,}"
        )

    print(f"  done: {shard.name} ({final_size:,} bytes)")
    return dest


def extract_shard(tar_path: Path, *, force: bool = False) -> Path:
    """Extract gzipped training chunks from a downloaded tar."""
    out_dir = LC0_CHUNKS_DIR / tar_path.stem
    marker = out_dir / ".extracted"
    if marker.exists() and not force:
        print(f"  skip extract (done): {tar_path.name}")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as archive:
        archive.extractall(path=out_dir, filter="data")
    marker.write_text(f"source={tar_path.name}\n", encoding="utf-8")
    print(f"  extracted: {tar_path.name} -> {out_dir}")
    return out_dir


def update_manifest_entry(shard: Lc0Shard, tar_path: Path, extracted: bool) -> None:
    manifest = load_manifest()
    manifest["shards"][shard.name] = {
        "url": shard.url,
        "size_bytes": shard.size_bytes,
        "sha256": sha256_file(tar_path),
        "tar_path": str(tar_path.relative_to(PROJECT_ROOT)),
        "extracted": extracted,
        "chunks_dir": str((LC0_CHUNKS_DIR / tar_path.stem).relative_to(PROJECT_ROOT))
        if extracted
        else None,
        "note": shard.note,
    }
    save_manifest(manifest)


def summarize(shards: list[Lc0Shard]) -> None:
    total = sum(s.size_bytes for s in shards)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output:       {LC0_RAW_DIR}")
    print(f"Shards:       {len(shards)}")
    print(f"Total size:   {total / (1024**3):.2f} GiB ({total:,} bytes)")
    for shard in shards:
        print(f"  - {shard.name}  ({shard.size_gb:.2f} GiB)  {shard.note}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download curated Lc0 training-data shards (~1–2 GB subset)"
    )
    parser.add_argument(
        "--max-gb",
        type=float,
        default=2.0,
        help="Byte budget when using default shard list (default: 2.0)",
    )
    parser.add_argument(
        "--shard",
        action="append",
        dest="shards",
        metavar="NAME",
        help="Download specific tar name(s); ignores --max-gb when set",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List curated shards and exit",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download tars only; skip extraction to chunks/",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract even if files already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned downloads without fetching",
    )
    args = parser.parse_args(argv)

    if args.list:
        print("Default subset:")
        summarize(list(DEFAULT_SHARDS))
        print("\nOptional extras (--shard):")
        for shard in EXTRA_SHARDS:
            print(f"  - {shard.name}  ({shard.size_gb:.2f} GiB)  {shard.note}")
        return 0

    try:
        chosen = select_shards(DEFAULT_SHARDS, args.max_gb, args.shards)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not chosen:
        print("No shards selected for the given budget.", file=sys.stderr)
        return 1

    summarize(chosen)
    if args.dry_run:
        print("Dry run — no download.")
        return 0

    ensure_dirs()
    print("\nDownloading...")
    for shard in chosen:
        tar_path = download_shard(shard, force=args.force)
        extracted = False
        if not args.no_extract:
            extract_shard(tar_path, force=args.force)
            extracted = True
        update_manifest_entry(shard, tar_path, extracted)

    manifest = load_manifest()
    chunk_files = list(LC0_CHUNKS_DIR.rglob("*.gz"))
    print("\nComplete.")
    print(f"Manifest: {LC0_MANIFEST}")
    print(f"Tars:     {len(list(LC0_TARS_DIR.glob('*.tar')))} under {LC0_TARS_DIR}")
    print(f"Chunks:   {len(chunk_files)} .gz files under {LC0_CHUNKS_DIR}")
    print(f"Recorded: {len(manifest.get('shards', {}))} shard(s) in manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())