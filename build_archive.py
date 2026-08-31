"""Create a deterministic ZIP and an internal SHA-256 manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parent
VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
ARCHIVE = ROOT.parent / f"hcc_tme_reproducibility_v{VERSION}.zip"
CHECKSUMS = ROOT / "SHA256SUMS.txt"
FIXED_TIMESTAMP = (2026, 8, 30, 0, 0, 0)


def included_files():
    excluded_names = {"SHA256SUMS.txt", ".DS_Store"}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if path.name in excluded_names or path.suffix in {".pyc", ".zip"}:
            continue
        if "__pycache__" in relative.parts:
            continue
        yield path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build():
    files = list(included_files())
    empty = [path for path in files if path.stat().st_size == 0]
    if empty:
        raise RuntimeError(f"Refusing to archive empty files: {empty}")
    checksum_text = "".join(
        f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in files
    )
    CHECKSUMS.write_text(checksum_text, encoding="utf-8")
    files.append(CHECKSUMS)

    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(files):
            relative = Path(ROOT.name) / path.relative_to(ROOT)
            info = zipfile.ZipInfo(relative.as_posix(), date_time=FIXED_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

    with zipfile.ZipFile(ARCHIVE, "r") as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP integrity test failed at {bad}")
        names = archive.namelist()
    print(f"created {ARCHIVE}")
    print(f"files: {len(names)}; bytes: {ARCHIVE.stat().st_size}; sha256: {sha256(ARCHIVE)}")
    return ARCHIVE


if __name__ == "__main__":
    build()
