"""Bundle per-slice features.npz into a folder + zip."""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "bundle_slice_npz.py"


def _load():
    spec = importlib.util.spec_from_file_location("bundle_slice_npz", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_slice(root: Path, name: str, payload: bytes) -> Path:
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    npz = folder / "features.npz"
    npz.write_bytes(payload)
    return npz


def test_discover_skips_dest_and_non_npz(tmp_path: Path):
    root = tmp_path / "fen_value_visits"
    _write_slice(root, "slice_a", b"aaa")
    _write_slice(root, "slice_b", b"bbb")
    dest = root / "npz_bundle"
    _write_slice(dest, "stale", b"nope")
    (root / "notes.txt").write_text("x", encoding="utf-8")
    found = _load().discover_slice_npz(root, skip={dest})
    assert [name for name, _ in found] == ["slice_a", "slice_b"]


def test_copy_and_zip_layout(tmp_path: Path):
    root = tmp_path / "fen_value_visits"
    _write_slice(root, "slice_a", b"aaa")
    _write_slice(root, "slice_b", b"bbb")
    dest = root / "npz_bundle"
    zip_path = root / "npz_bundle.zip"
    mod = _load()
    copied = mod.copy_npz(mod.discover_slice_npz(root, skip={dest}), dest)
    assert (dest / "slice_a" / "features.npz").read_bytes() == b"aaa"
    assert (dest / "slice_b" / "features.npz").read_bytes() == b"bbb"
    mod.write_zip([(p.parent.name, p) for p in copied], zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert zf.read("slice_a/features.npz") == b"aaa"
        assert zf.read("slice_b/features.npz") == b"bbb"
    assert names == {"slice_a/features.npz", "slice_b/features.npz"}


def test_main_no_copy_writes_zip_from_slices(tmp_path: Path):
    root = tmp_path / "fen_value_visits"
    _write_slice(root, "slice_a", b"aaa")
    dest = root / "npz_bundle"
    zip_path = root / "out.zip"
    rc = _load().main(
        [
            "--root",
            str(root),
            "--dest",
            str(dest),
            "--zip",
            str(zip_path),
            "--no-copy",
        ]
    )
    assert rc == 0
    assert not dest.exists()
    with zipfile.ZipFile(zip_path) as zf:
        assert zf.read("slice_a/features.npz") == b"aaa"
