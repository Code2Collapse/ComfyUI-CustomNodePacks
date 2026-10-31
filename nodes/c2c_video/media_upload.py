"""Upload and scan for Load Video (C2C)'s unified sources (owner A21, Rev 3 Part 3.4 / 3.5; ledger L9.89).

Core's upload widget takes only video/webm and video/mp4 (frontend 1.53.10: ``accept``, plus a ``video/*`` MIME
filter), and EXR / DPX have no browser MIME type at all. This is the dedicated upload: any format an installed
backend reads, several files at once (a sequence), a folder, with the file's own bytes checked against its extension.

Safety (Rev 3 3.4 "File safety"):

* the name is reduced to a bare file name - no folders, no control characters, no ``..``, no Windows device names;
* the target's real path must stay inside ComfyUI's input folder;
* the extension must be a readable format AND the first bytes must be that format (an .exr must start 76 2f 31 01);
* a per-file size cap (``C2C_UPLOAD_MAX_MB``, default 20000), streamed to a ``.part`` file and moved into place only
  when complete - a cancelled upload leaves nothing behind;
* duplicates: the same name with the same bytes is reused, never written twice; a single file with other bytes gets
  core's ``name (1).ext``; a SEQUENCE is never renamed file by file (that breaks its numbering) - the whole set goes
  to a new folder ``shot (2)/`` unless ``overwrite`` is set.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import uuid

try:
    from ...c2c_core.video import media as MEDIA
    from ...c2c_core.video.sequence import detect_from_file, scan_directory
except (ImportError, ValueError):
    from c2c_core.video import media as MEDIA
    from c2c_core.video.sequence import detect_from_file, scan_directory

_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
_CTRL = re.compile(r"[\x00-\x1f\x7f<>:\"|?*]")
STAGING = ".c2c_upload"


class UploadError(Exception):
    def __init__(self, status: int, message: str) -> None:
        self.status = int(status)
        self.message = str(message)
        super().__init__(message)


def max_bytes() -> int:
    try:
        return int(float(os.environ.get("C2C_UPLOAD_MAX_MB", "20000")) * 1024 * 1024)
    except ValueError:
        return 20000 * 1024 * 1024


def allowed_extensions() -> set[str]:
    return set(MEDIA.image_extensions()) | {"." + e for e in MEDIA.VIDEO_EXTENSIONS}


def safe_name(name: str) -> str:
    """A bare, harmless file name, or UploadError(400)."""
    base = re.split(r"[\\/]", str(name or ""))[-1].strip()
    base = _CTRL.sub("_", base).rstrip(" .")
    stem = base.split(".", 1)[0].lower()
    if not base or base in (".", "..") or stem in _RESERVED or len(base) > 200:
        raise UploadError(400, f"{name!r} is not a usable file name")
    return base


def safe_folder(name: str) -> str:
    """One folder level below input/ (sub-sub folders are not created by an upload)."""
    s = str(name or "").strip().replace("\\", "/").strip("/")
    if not s:
        return ""
    parts = [p for p in s.split("/") if p]
    if len(parts) != 1:
        raise UploadError(400, f"subfolder {name!r}: one folder name, no slashes")
    return safe_name(parts[0])


def sniff_ok(ext: str, head: bytes) -> bool:
    """Do the first bytes look like this extension's format? (never trusts the name alone)"""
    e = ext.lower().lstrip(".")
    h = head or b""
    if e == "exr":
        return h[:4] == b"\x76\x2f\x31\x01"
    if e == "png":
        return h[:8] == b"\x89PNG\r\n\x1a\n"
    if e in ("jpg", "jpeg"):
        return h[:3] == b"\xff\xd8\xff"
    if e in ("tif", "tiff"):
        return h[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")
    if e == "dpx":
        return h[:4] in (b"SDPX", b"XPDS")
    if e == "bmp":
        return h[:2] == b"BM"
    if e == "hdr":
        return h.startswith(b"#?RADIANCE") or h.startswith(b"#?RGBE")
    if e == "webp":
        return h[:4] == b"RIFF" and h[8:12] == b"WEBP"
    if e == "tga":                               # no magic: colour-map type and image type must be legal
        return len(h) >= 3 and h[1] in (0, 1) and h[2] in (1, 2, 3, 9, 10, 11)
    if e in ("mp4", "mov", "m4v", "3gp"):
        return h[4:8] in (b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip", b"pnot")
    if e in ("mkv", "webm"):
        return h[:4] == b"\x1a\x45\xdf\xa3"
    if e == "avi":
        return h[:4] == b"RIFF" and h[8:12] == b"AVI "
    if e == "mxf":
        return h[:4] == b"\x06\x0e\x2b\x34"
    if e == "gif":
        return h[:6] in (b"GIF87a", b"GIF89a")
    if e in ("ts", "mts", "m2ts"):
        return (len(h) > 0 and h[0] == 0x47) or (len(h) > 4 and h[4] == 0x47)
    if e == "wmv":
        return h[:4] == b"\x30\x26\xb2\x75"
    if e == "flv":
        return h[:3] == b"FLV"
    if e in ("mpg", "mpeg"):
        return h[:4] in (b"\x00\x00\x01\xba", b"\x00\x00\x01\xb3")
    if e == "ogv":
        return h[:4] == b"OggS"
    if e == "y4m":
        return h[:9] == b"YUV4MPEG2"
    return False


def _sha256(path: str) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            d.update(block)
    return d.hexdigest()


def _inside(root: str, path: str) -> bool:
    r, p = os.path.realpath(root), os.path.realpath(path)
    try:
        return os.path.commonpath([r, p]) == r
    except ValueError:
        return False


class Staging:
    """One upload request: files are streamed into input/.c2c_upload/<id>/ first, placed when all arrived."""

    def __init__(self, input_dir: str) -> None:
        self.input_dir = os.path.abspath(input_dir)
        self.dir = os.path.join(self.input_dir, STAGING, uuid.uuid4().hex)
        os.makedirs(self.dir, exist_ok=True)
        self.files: list[tuple[str, str, str]] = []   # (safe name, folder from a folder upload, staged path)

    def add(self, filename: str, relpath: str = "") -> tuple[str, object]:
        name = safe_name(filename)
        ext = os.path.splitext(name)[1].lower()
        if ext not in allowed_extensions():
            raise UploadError(415, f"{name}: {ext or 'no extension'} is not a format this ComfyUI can read "
                                   f"(readable: {', '.join(sorted(allowed_extensions()))})")
        folder = ""
        rel = str(relpath or "").replace("\\", "/").strip("/")
        if "/" in rel:                               # a folder upload: keep the top folder's name
            folder = safe_folder(rel.split("/", 1)[0])
        staged = os.path.join(self.dir, f"{len(self.files):06d}_{name}.part")
        self.files.append((name, folder, staged))
        return staged, _Writer(staged, ext, name)

    def cleanup(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)
        try:
            os.rmdir(os.path.dirname(self.dir))      # input/.c2c_upload when it is empty
        except OSError:
            pass

    def place(self, subfolder: str = "", overwrite: bool = False) -> dict:
        """Move the staged files into input/ (rules in the module notes); returns what the UI needs."""
        if not self.files:
            raise UploadError(400, "no files in the upload")
        folders = {f for _n, f, _p in self.files}
        folder = safe_folder(subfolder) if subfolder else (folders.pop() if len(folders) == 1 else "")
        names = [n for n, _f, _p in self.files]
        if len(set(n.lower() for n in names)) != len(names):
            raise UploadError(400, "two uploaded files have the same name")
        multi = len(self.files) > 1
        if multi and not folder:
            seqs, _ = scan_directory(self.dir, names=[os.path.basename(p)[7:-5] for _n, _f, p in self.files])
            big = max(seqs, key=lambda s: s.count) if seqs else None
            stem = (big.prefix.rstrip(" ._-") if big is not None and big.count > 1 else "") or "upload"
            folder = safe_name(stem)
        target_dir = os.path.join(self.input_dir, folder) if folder else self.input_dir
        if multi:
            target_dir = self._free_set_folder(target_dir, overwrite)
        os.makedirs(target_dir, exist_ok=True)
        if not _inside(self.input_dir, target_dir):
            raise UploadError(400, "the upload would land outside the input folder")
        out = []
        for name, _f, staged in self.files:
            dest = os.path.join(target_dir, name)
            status = "written"
            if os.path.exists(dest):
                if _sha256(dest) == _sha256(staged):
                    status = "reused"
                elif overwrite or multi:             # a set: _free_set_folder already chose a clean folder
                    status = "overwritten"
                else:
                    dest, status = _free_name(dest), "renamed"
            if status == "reused":
                os.remove(staged)
            else:
                os.replace(staged, dest)
            out.append({"name": os.path.basename(dest), "status": status})
        rel_folder = os.path.relpath(target_dir, self.input_dir).replace("\\", "/")
        rel_folder = "" if rel_folder == "." else rel_folder
        prefix = rel_folder + "/" if rel_folder else ""
        seqs, _ = scan_directory(target_dir, exts=MEDIA.image_extensions())
        mine = {o["name"] for o in out}
        found = [s for s in seqs if s.count > 1 and mine & {os.path.basename(f) for f in s.files}]
        if len(out) == 1:
            value = prefix + out[0]["name"]
            first, _rel = detect_from_file(os.path.join(target_dir, out[0]["name"]))
            if first is not None and first.count > 1:
                value = prefix + first.pattern          # one frame of a sequence that is already there: the sequence
        elif found:
            value = prefix + max(found, key=lambda s: s.count).pattern
        else:
            value = prefix + out[0]["name"]
        return {"files": out, "subfolder": rel_folder, "value": value,
                "sequences": [dict(s.to_dict(), pattern=prefix + s.pattern) for s in found]}

    def _free_set_folder(self, target_dir: str, overwrite: bool) -> str:
        """A folder where every file of the set can keep its name: this one, unless a same-named file with other bytes
        is there (then 'name (2)', 'name (3)' ...); with overwrite, always this one."""
        if overwrite:
            return target_dir

        def clashes(d: str) -> bool:
            for name, _f, staged in self.files:
                dest = os.path.join(d, name)
                if os.path.exists(dest) and _sha256(dest) != _sha256(staged):
                    return True
            return False

        if not clashes(target_dir):
            return target_dir
        if target_dir == self.input_dir:
            raise UploadError(409, "files with these names and other contents are already in the input folder - "
                                   "upload into a subfolder or tick overwrite")
        i = 2
        while True:
            cand = f"{target_dir} ({i})"
            if not os.path.exists(cand) or not clashes(cand):
                return cand
            i += 1


def _free_name(dest: str) -> str:
    stem, ext = os.path.splitext(dest)
    i = 1
    while os.path.exists(f"{stem} ({i}){ext}"):
        i += 1
    return f"{stem} ({i}){ext}"


class _Writer:
    """Streams one part to its .part file: the size cap, then the magic-byte check on the first bytes."""

    def __init__(self, path: str, ext: str, name: str) -> None:
        self.path, self.ext, self.name = path, ext, name
        self.size = 0
        self.head = b""
        self.f = open(path, "wb")

    def write(self, chunk: bytes) -> None:
        self.size += len(chunk)
        if self.size > max_bytes():
            self.close()
            raise UploadError(413, f"{self.name} is larger than C2C_UPLOAD_MAX_MB "
                                   f"({max_bytes() // (1024 * 1024)} MB)")
        if len(self.head) < 32:
            self.head += chunk[: 32 - len(self.head)]
        self.f.write(chunk)

    def close(self) -> None:
        if not self.f.closed:
            self.f.close()

    def finish(self) -> None:
        self.close()
        if self.size == 0:
            raise UploadError(400, f"{self.name} is empty")
        if not sniff_ok(self.ext, self.head):
            raise UploadError(415, f"{self.name} does not contain {self.ext} data (its first bytes say otherwise) - "
                                   f"was it renamed?")


async def handle_upload(request, input_dir: str) -> dict:
    """The aiohttp side of POST /c2c/media/upload: fields ``subfolder`` / ``overwrite`` (before the files) and any
    number of ``file`` parts, each optionally preceded by a ``relpath`` field (a folder upload's relative path)."""
    reader = await request.multipart()
    stage = Staging(input_dir)
    subfolder, overwrite, relpath = "", False, ""
    try:
        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name == "subfolder":
                subfolder = (await part.text()).strip()
            elif part.name == "overwrite":
                overwrite = (await part.text()).strip().lower() in ("1", "true", "yes")
            elif part.name == "relpath":
                relpath = (await part.text()).strip()
            elif part.name == "file" and part.filename:
                _staged, w = stage.add(part.filename, relpath)
                relpath = ""
                try:
                    while True:
                        chunk = await part.read_chunk(1 << 20)
                        if not chunk:
                            break
                        w.write(chunk)
                    w.finish()
                finally:
                    w.close()
            else:
                await part.release()
        return stage.place(subfolder, overwrite)
    finally:
        stage.cleanup()


def scan_candidates(path: str) -> dict:
    """Candidate sequences for a picked frame or a folder (Rev 3 3.5: show them, never merge them)."""
    images = MEDIA.image_extensions()
    if os.path.isdir(path):
        seqs, _ = scan_directory(path, exts=images)
        return {"chosen": None, "candidates": [s.to_dict() for s in seqs if s.count > 1]}
    chosen, related = detect_from_file(path, exts=images)
    return {"chosen": chosen.to_dict() if chosen is not None else None,
            "candidates": [s.to_dict() for s in related if s.count > 1 or s is chosen]}
