#!/usr/bin/env python3
"""Download a Lichess standard monthly PGN dump (.pgn.zst) into data/raw/lichess/dumps/.

Source: https://database.lichess.org/  (standard rated games, one month per file).
Keeps the archive compressed; conversion to FEN is a later step.

Example::

    py -3.12 scripts/download_lichess_dump.py
    py -3.12 scripts/download_lichess_dump.py --month 2026-07
    py -3.12 scripts/download_lichess_dump.py --list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tinymlinternship.config.settings import (
    LICHESS_DUMP_BASE_URL,
    LICHESS_DUMP_LIST_URL,
    LICHESS_DUMP_MANIFEST,
    LICHESS_DUMP_SHA256_URL,
    LICHESS_DUMPS_DIR,
    LICHESS_RAW_DIR,
    PROJECT_ROOT,
)

CHUNK_SIZE = 1024 * 1024  # 1 MiB
PROGRESS_EVERY = 256 * CHUNK_SIZE  # 256 MiB
USER_AGENT = "SARDINE-TinyMLInternship/0.1 (+https://database.lichess.org/)"
MONTH_RE = re.compile(r"^(\d{4}-\d{2})$")
FILENAME_RE = re.compile(r"lichess_db_standard_rated_(\d{4}-\d{2})\.pgn\.zst$")


def ensure_dirs() -> None:
    LICHESS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    LICHESS_DUMPS_DIR.mkdir(parents=True, exist_ok=True)


def http_get_text(url: str, *, timeout: int = 60) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def dump_filename(month: str) -> str:
    return f"lichess_db_standard_rated_{month}.pgn.zst"


def dump_url(month: str) -> str:
    return f"{LICHESS_DUMP_BASE_URL}/{dump_filename(month)}"


def parse_months_from_list(text: str) -> list[str]:
    months: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        name = line.rsplit("/", 1)[-1]
        match = FILENAME_RE.search(name)
        if not match:
            continue
        month = match.group(1)
        if month not in seen:
            seen.add(month)
            months.append(month)
    return months


def parse_sha256_map(text: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        digest, name = parts[0], parts[-1]
        mapping[Path(name).name] = digest.lower()
    return mapping


def head_content_length(url: str) -> int | None:
    request = Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=60) as response:
            raw = response.headers.get("Content-Length")
            return int(raw) if raw else None
    except (HTTPError, URLError, ValueError, TimeoutError):
        return None


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
    if LICHESS_DUMP_MANIFEST.exists():
        return json.loads(LICHESS_DUMP_MANIFEST.read_text(encoding="utf-8"))
    return {"base_url": LICHESS_DUMP_BASE_URL, "dumps": {}}


def save_manifest(manifest: dict) -> None:
    LICHESS_DUMP_MANIFEST.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def _human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{n} B"


def download_file(
    url: str,
    dest: Path,
    *,
    expected_size: int | None,
    force: bool = False,
) -> Path:
    if dest.exists() and not force:
        size = dest.stat().st_size
        if expected_size is not None and size == expected_size:
            print(f"  skip (complete): {dest.name} ({_human_bytes(size)})")
            return dest
        if size == 0:
            dest.unlink()
        elif expected_size is not None and size > expected_size:
            print(f"  re-download (oversized): {dest.name}")
            dest.unlink()

    offset = dest.stat().st_size if dest.exists() else 0
    if offset and expected_size:
        print(f"  resume: {dest.name} ({_human_bytes(offset)} / {_human_bytes(expected_size)})")
    elif offset:
        print(f"  resume: {dest.name} ({_human_bytes(offset)} so far)")
    else:
        size_note = f" ({_human_bytes(expected_size)})" if expected_size else ""
        print(f"  fetch: {dest.name}{size_note}")

    headers: dict[str, str] = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"

    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=120) as response:
            status = getattr(response, "status", response.getcode())
            if offset and status not in (206, 200):
                print(f"  server did not resume; restarting {dest.name}")
                dest.unlink(missing_ok=True)
                offset = 0
                request = Request(url, headers={"User-Agent": USER_AGENT})
                response = urlopen(request, timeout=120)
                status = getattr(response, "status", response.getcode())
            if offset and status == 200:
                print(f"  Range ignored (HTTP 200); restarting {dest.name}")
                dest.unlink(missing_ok=True)
                offset = 0

            mode = "ab" if offset else "wb"
            downloaded = offset
            next_report = ((downloaded // PROGRESS_EVERY) + 1) * PROGRESS_EVERY
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open(mode) as handle:
                while True:
                    block = response.read(CHUNK_SIZE)
                    if not block:
                        break
                    handle.write(block)
                    downloaded += len(block)
                    if downloaded >= next_report:
                        if expected_size:
                            pct = 100.0 * downloaded / expected_size
                            print(
                                f"    {dest.name}: {_human_bytes(downloaded)} / "
                                f"{_human_bytes(expected_size)} ({pct:.1f}%)"
                            )
                        else:
                            print(f"    {dest.name}: {_human_bytes(downloaded)}")
                        next_report += PROGRESS_EVERY
                    if expected_size and downloaded > expected_size + CHUNK_SIZE:
                        raise RuntimeError(
                            f"Download exceeded expected size for {dest.name}: "
                            f"{downloaded:,} > {expected_size:,}"
                        )
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Download failed for {dest.name}: {exc}") from exc

    final_size = dest.stat().st_size
    if expected_size is not None and final_size != expected_size:
        raise RuntimeError(
            f"Size mismatch for {dest.name}: got {final_size:,}, expected {expected_size:,}"
        )
    print(f"  done: {dest.name} ({_human_bytes(final_size)})")
    return dest


def resolve_month(requested: str | None) -> str:
    listing = http_get_text(LICHESS_DUMP_LIST_URL)
    months = parse_months_from_list(listing)
    if not months:
        raise RuntimeError(f"No monthly dumps listed at {LICHESS_DUMP_LIST_URL}")
    if requested is None:
        return months[0]
    if not MONTH_RE.match(requested):
        raise ValueError(f"month must be YYYY-MM, got {requested!r}")
    if requested not in months:
        raise ValueError(
            f"month {requested} is not in the Lichess standard dump list "
            f"(latest is {months[0]})"
        )
    return requested


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download a Lichess standard rated monthly PGN dump (.pgn.zst)"
    )
    parser.add_argument(
        "--month",
        metavar="YYYY-MM",
        help="Dump month (default: latest published on database.lichess.org)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print available months (latest first) and exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already has the expected size",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve month and size without downloading",
    )
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Skip SHA256 check after download",
    )
    args = parser.parse_args(argv)

    try:
        listing = http_get_text(LICHESS_DUMP_LIST_URL)
        months = parse_months_from_list(listing)
        if not months:
            print(f"Error: no dumps listed at {LICHESS_DUMP_LIST_URL}", file=sys.stderr)
            return 1
        if args.list:
            print(f"Standard rated dumps ({len(months)} months), latest first:")
            for month in months:
                print(f"  {month}  {dump_url(month)}")
            return 0
        month = resolve_month(args.month)
    except (HTTPError, URLError, TimeoutError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    filename = dump_filename(month)
    url = dump_url(month)
    dest = LICHESS_DUMPS_DIR / filename
    expected_size = head_content_length(url)
    sha_map: dict[str, str] = {}
    try:
        sha_map = parse_sha256_map(http_get_text(LICHESS_DUMP_SHA256_URL))
    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"Warning: could not fetch sha256sums.txt ({exc})")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output:       {dest}")
    print(f"Month:        {month}")
    print(f"URL:          {url}")
    if expected_size is not None:
        print(f"Size:         {_human_bytes(expected_size)} ({expected_size:,} bytes)")
    else:
        print("Size:         unknown (HEAD had no Content-Length)")
    if filename in sha_map:
        print(f"SHA256:       {sha_map[filename]}")

    if args.dry_run:
        print("Dry run — no download.")
        return 0

    ensure_dirs()
    try:
        download_file(url, dest, expected_size=expected_size, force=args.force)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    digest: str | None = None
    expected_digest = sha_map.get(filename)
    if not args.skip_sha256 and expected_digest:
        print("  verifying SHA256…")
        digest = sha256_file(dest)
        if digest != expected_digest:
            print(
                f"Error: SHA256 mismatch for {filename}\n"
                f"  expected {expected_digest}\n"
                f"  got      {digest}",
                file=sys.stderr,
            )
            return 1
        print(f"  SHA256 ok: {digest}")
    elif not args.skip_sha256:
        print("  skip SHA256 (checksum not in sha256sums.txt)")

    final_size = dest.stat().st_size
    manifest = load_manifest()
    manifest["dumps"][filename] = {
        "month": month,
        "url": url,
        "path": dest.relative_to(PROJECT_ROOT).as_posix(),
        "size_bytes": final_size,
        "sha256": digest or expected_digest,
        "sha256_verified": bool(digest and expected_digest and digest == expected_digest),
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "Keep compressed; stream-decompress when converting to FEN.",
    }
    save_manifest(manifest)
    print("Complete.")
    print(f"Manifest: {LICHESS_DUMP_MANIFEST}")
    print(f"Dump:     {dest} ({_human_bytes(final_size)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
