"""Give each output its own name under one shared version.

THE SHAPE THIS SERVES. Four Video Combines in one workflow - a mov, an exr, a
png sequence and a masked mov - and one Folder Version Incrementer feeding all
of them. Each output needs its own prefix or suffix; all four have to land
under the SAME version, because they are one render.

The incrementer decides the version. This takes what it produced and re-tags
it for one output, which is why there is one of these per Video Combine rather
than a fixed number of suffix slots on the incrementer: four outputs today,
seven next week, and each one visible on the canvas with its own name on it.

WHERE THE TAG GOES is the same three choices the incrementer already has, and
they produce genuinely different trees:

    filename    shot_a/09-22-2026/v004/shot_a_masked.mov
                -> everything in one folder, distinguished by name

    subfolder   shot_a/09-22-2026/v004/masked/shot_a.mov
                -> one folder per output INSIDE the version. This is usually
                   what you want for exr sequences, which are thousands of
                   files and should not share a directory with anything.

    folder      shot_a_masked/09-22-2026/v004/shot_a.mov
                -> a separate top-level tree. Still the same version, because
                   the incrementer's lease ignores the suffix when deciding
                   which outputs belong together.

A PREFIX is also offered, because some facilities name by prefix rather than
suffix, and a prefix that has to be faked with a suffix ends up in the wrong
sort order - which is the entire reason anyone uses a prefix.

This node never touches the disk. It rewrites a path string, so it cannot
create a version, cannot skip one, and cannot disagree with the incrementer
about which one is current.
"""

from __future__ import annotations

SUFFIX_MODES = ("filename", "subfolder", "folder")


class VersionVariantError(ValueError):
    pass


def _sep_of(path: str) -> str:
    """Whatever separator the incrementer used, so the two agree.

    Rebuilding with the wrong one produces a path that works on one OS and
    silently makes a file literally named "v004\\shot.mov" on the other.
    """
    if "\\" in path and "/" not in path:
        return "\\"
    return "/"


def split_path(filename_prefix: str) -> tuple[list[str], str]:
    """(directory parts, basename) from an incrementer's filename_prefix."""
    sep = _sep_of(filename_prefix)
    parts = [p for p in filename_prefix.split(sep) if p != ""]
    if not parts:
        raise VersionVariantError(
            "The filename prefix is empty. Connect this node's `filename_prefix` "
            "input to the Folder Version Incrementer's `filename_prefix` output.")
    return parts[:-1], parts[-1]


def apply_variant(filename_prefix: str, *, prefix: str = "", suffix: str = "",
                  mode: str = "subfolder", version_string: str = "",
                  extension: str = "") -> dict:
    """Re-tag one output's path. Returns every string a save node might want.

    `version_string` is only used to find where the version sits in the path,
    so a `subfolder` tag can be placed immediately after it. Without it the
    tag goes on the end, which is the same thing whenever the incrementer's
    own suffix_mode was not already `subfolder`.
    """
    if mode not in SUFFIX_MODES:
        raise VersionVariantError(
            f"Unknown mode {mode!r}. Use one of: {', '.join(SUFFIX_MODES)}.")

    tag_prefix = (prefix or "").strip()
    tag_suffix = (suffix or "").strip()
    sep = _sep_of(filename_prefix)
    dirs, base = split_path(filename_prefix)

    if not tag_prefix and not tag_suffix:
        # Nothing to do, and saying so beats silently returning the input as
        # though a tag had been applied.
        out_dirs, out_base = dirs, base
    elif mode == "filename":
        out_dirs = dirs
        out_base = f"{tag_prefix}{base}{tag_suffix}"
    elif mode == "subfolder":
        # A leading separator character reads as a join hint, not part of the
        # name: "_masked" becomes the folder "masked".
        folder = (tag_prefix + tag_suffix).lstrip("_-. ")
        if not folder:
            raise VersionVariantError(
                "The tag is only separator characters, so it would make a "
                "folder with no name. Use something like 'masked' or '_masked'.")
        out_dirs = list(dirs)
        # Placed straight after the version directory when one is identifiable,
        # so the tree reads v004/masked/... rather than .../masked buried at
        # the bottom of whatever the incrementer already nested.
        if version_string and version_string in out_dirs:
            at = out_dirs.index(version_string) + 1
            out_dirs.insert(at, folder)
        else:
            out_dirs.append(folder)
        out_base = base
    else:  # folder
        folder_tag = (tag_prefix + tag_suffix)
        if not dirs:
            raise VersionVariantError(
                "There is no folder to tag - the prefix has no directory part. "
                "Use mode 'filename' for a bare name.")
        out_dirs = list(dirs)
        # The TOP folder, which is the one the incrementer treats as the shot.
        out_dirs[0] = out_dirs[0] + folder_tag
        out_base = base

    subfolder_path = sep.join(out_dirs)
    new_prefix = sep.join(out_dirs + [out_base])
    ext = (extension or "").strip()
    if ext and not ext.startswith("."):
        ext = "." + ext
    output_filename = f"{new_prefix}{ext}" if ext else new_prefix

    return {
        "filename_prefix": new_prefix,
        "subfolder_path": subfolder_path,
        "output_filename": output_filename,
        "basename": out_base,
        "folder_name": out_dirs[0] if out_dirs else "",
    }


def describe(result: dict, *, mode: str, prefix: str, suffix: str,
             version_string: str) -> str:
    """The resulting path, and what stayed shared."""
    tag = (prefix or "") + (suffix or "")
    lines = []
    if not tag.strip():
        lines.append(
            "No prefix or suffix set, so this output is unchanged from the "
            "incrementer. That is fine for the one output that needs no tag - "
            "usually the main mov.")
    else:
        where = {
            "filename": "the file name",
            "subfolder": "a folder inside the version",
            "folder": "the top-level folder",
        }[mode]
        lines.append(f"Tagged {where} with {tag.strip()!r}.")
    lines.append(f"Path: {result['filename_prefix']}")
    if version_string:
        lines.append(
            f"Version {version_string} comes from the incrementer and is NOT "
            "changed here - this node only renames. Every variant fed by the "
            "same incrementer therefore lands under the same version, which is "
            "the point of using one incrementer for several outputs.")
    if mode == "folder":
        lines.append(
            "'folder' mode writes to a separate top-level tree. It still "
            "shares the version: the incrementer's lease ignores the suffix "
            "when working out which outputs belong to the same shot.")
    return "\n".join(lines)
