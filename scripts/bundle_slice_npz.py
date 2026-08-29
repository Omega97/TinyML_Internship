#!/usr/bin/env python3
"""Copy every slice ``features.npz`` into one folder, then zip that tree.

Default layout under ``data/processed/board_eval/fen_value_visits/``::

    npz_bundle/<slice>/features.npz
    npz_bundle.zip                  # same paths inside the archive

The bundle folder is skipped when scanning slices. ``.npz`` is already a zip;
the archive uses stored (no extra deflate) unless ``--deflate``.

    py -3.12 -u scripts/bundle_slice_npz.py
    py -3.12 -u scripts/bundle_slice_npz.py --no-copy
    py -3.12 -u scripts/bundle_slice_npz.py --clean
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tinymlinternship.config.settings import PROCESSED_DATA_DIR, PROJECT_ROOT
from tinymlinternship.data.board_store import BOARD_EVAL_DIR_NAME, FEN_VALUE_VISITS_DIR_NAME
from tinymlinternship.nnue.dataset import SLICE_FEATURES_NPZ

DEFAULT_ROOT = PROCESSED_DATA_DIR / BOARD_EVAL_DIR_NAME / FEN_VALUE_VISITS_DIR_NAME
DEFAULT_DEST_NAME = "npz_bundle"
ARC_NPZ = SLICE_FEATURES_NPZ


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def discover_slice_npz(
    root: Path, *, skip: set[Path] | None = None
) -> list[tuple[str, Path]]:
    """Immediate child dirs that contain ``features.npz`` (dest folder excluded)."""
    skip_res = {p.resolve() for p in (skip or set())}
    found: list[tuple[str, Path]] = []
    if not root.is_dir():
        return found
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.resolve() in skip_res:
            continue
        npz = child / SLICE_FEATURES_NPZ
        if npz.is_file():
            found.append((child.name, npz))
    return found


def copy_npz(items: list[tuple[str, Path]], dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, src in items:
        out = dest / name / ARC_NPZ
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        written.append(out)
    return written


def write_zip(
    items: list[tuple[str, Path]],
    zip_path: Path,
    *,
    deflate: bool = False,
) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    compression = zipfile.ZIP_DEFLATED if deflate else zipfile.ZIP_STORED
    with zipfile.ZipFile(
        zip_path, "w", compression=compression, allowZip64=True
    ) as zf:
        for name, src in items:
            zf.write(src, arcname=f"{name}/{ARC_NPZ}")
    return zip_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Copy slice features.npz into one folder and zip them"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Parent of per-slice folders",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help=f"Copy target (default: <root>/{DEFAULT_DEST_NAME})",
    )
    parser.add_argument(
        "--zip",
        dest="zip_path",
        type=Path,
        default=None,
        help="Zip path (default: <root>/npz_bundle.zip)",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Write the zip from the slice files; do not copy into --dest",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Copy into --dest only; do not write a zip",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete --dest before copying",
    )
    parser.add_argument(
        "--deflate",
        action="store_true",
        help="ZIP_DEFLATED (slow; npz is already compressed)",
    )
    args = parser.parse_args(argv)

    if args.no_copy and args.no_zip:
        print("nothing to do: both --no-copy and --no-zip", file=sys.stderr)
        return 2

    root = _resolve(args.root)
    dest = _resolve(args.dest) if args.dest is not None else (root / DEFAULT_DEST_NAME)
    zip_path = (
        _resolve(args.zip_path)
        if args.zip_path is not None
        else (root / f"{DEFAULT_DEST_NAME}.zip")
    )

    skip = {dest}
    items = discover_slice_npz(root, skip=skip)
    if not items:
        print(f"no {SLICE_FEATURES_NPZ} under {root}", file=sys.stderr)
        return 1

    bytes_src = sum(src.stat().st_size for _, src in items)
    print(f"slices with npz: {len(items):,}  ({bytes_src / (1024 ** 3):.2f} GiB)")

    zip_sources = items
    if not args.no_copy:
        if args.clean and dest.exists():
            shutil.rmtree(dest)
        copied = copy_npz(items, dest)
        print(f"copied {len(copied):,} → {_rel(dest)}/<slice>/{ARC_NPZ}")
        zip_sources = [(p.parent.name, p) for p in copied]

    if not args.no_zip:
        write_zip(zip_sources, zip_path, deflate=args.deflate)
        zsize = zip_path.stat().st_size
        print(f"zip → {_rel(zip_path)}  ({zsize / (1024 ** 3):.2f} GiB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
