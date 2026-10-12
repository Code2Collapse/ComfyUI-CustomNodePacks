"""Weight downloads only with the user's permission (L7.97, owner 2026-10-12).

Owner's rule: detect weights that are already installed and reuse them; download only when the node's "Allow
Download" is ticked (default off); save into the model's own subfolder; when weights are missing and downloading is
off, run the labelled model-free method where one exists and say which model is needed and where it goes.

A node runs its work inside ``with allow_downloads(flag):``; every loader asks ``downloads_allowed()``. Outside any
node the answer is False, so a probe, a test or a background scan can never start a multi-GB download. A download the
user starts on purpose (the Model Browser, an explicit "download" button) does not go through here.
"""
from __future__ import annotations

import contextlib
import contextvars
import os
from typing import Iterator, List, Optional

_ALLOWED: contextvars.ContextVar[bool] = contextvars.ContextVar("c2c_downloads_allowed", default=False)
# Installed copies the resolvers found during the current node run (None outside a node)
_FOUND: contextvars.ContextVar[Optional[List[str]]] = contextvars.ContextVar("c2c_models_found", default=None)


@contextlib.contextmanager
def allow_downloads(flag: bool) -> Iterator[None]:
    token = _ALLOWED.set(bool(flag) and os.environ.get("C2C_NO_DOWNLOADS") != "1")
    found = _FOUND.set([])
    try:
        yield
    finally:
        _FOUND.reset(found)
        _ALLOWED.reset(token)


def downloads_allowed() -> bool:
    return bool(_ALLOWED.get()) and os.environ.get("C2C_NO_DOWNLOADS") != "1"


def note_found(path: str) -> None:
    """A resolver reports an installed copy it is about to hand out (not a download)."""
    seen = _FOUND.get()
    if seen is None or not path:
        return
    p = os.path.abspath(str(path))
    if p not in seen:
        seen.append(p)


def found_paths() -> List[str]:
    return list(_FOUND.get() or [])


def model_status(*, fallback_why: Optional[str] = None, path: str = "") -> str:
    """The status line a node puts in its info output. Owner 2026-10-12: "Show clear status messages: Indicate
    whether the model was found, loaded, missing, or requires user permission to download." ``fallback_why`` is the
    reason a model-free result was used; ``path`` defaults to the installed copies found during this run."""
    if fallback_why is not None:
        if "downloading is off" in str(fallback_why):
            return "needs permission - model-free result"
        return "missing - model-free result"
    where = path or ", ".join(found_paths())
    return "loaded" + (f" - found at {where}" if where else "")


class ModelNotDownloaded(FileNotFoundError):
    """The model is not installed and the node may not download it. The message says what, where and how."""

    def __init__(self, model: str, *, target: str = "", source: str = "", widget: str = "allow_download",
                 file_name: str = ""):
        # target = the absolute FOLDER the weights belong in; file_name = the file expected there (empty for a model
        # that is a whole folder, e.g. a Transformers repo). Not `filename`: that is OSError's own attribute, and
        # setting it turns str(error) into "[Errno None] None: '<name>'".
        self.model, self.target, self.source, self.file_name = model, target, source, file_name
        if target and file_name:
            where = f" Expected file '{file_name}' in folder: {target}"
        elif target:
            where = f" Expected in folder: {target}"
        else:
            where = ""
        frm = f" (official source: {source})" if source else ""
        super().__init__(
            f"Model '{model}' is not installed and downloading is off.{where}{frm}. Place it there, or tick "
            f"'{widget}' on the node to download it there.")

    @property
    def expected_path(self) -> str:
        """The full path the missing file is expected at ('' when only a folder is known)."""
        return os.path.join(self.target, self.file_name) if self.target and self.file_name else ""


def hf_local_only() -> bool:
    """For transformers / huggingface_hub calls: ``local_files_only=hf_local_only()``."""
    return not downloads_allowed()


def hf_cached_snapshot(repo_id: str) -> Optional[str]:
    """A repo already in the Hugging Face cache (another node may have fetched it): its folder, or None. No network."""
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(repo_id=repo_id, local_files_only=True)
    except Exception:
        return None


def hf_cached_file(repo_id: str, filename: str) -> Optional[str]:
    """One file of a repo already in the Hugging Face cache, or None. No network."""
    try:
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(repo_id=repo_id, filename=filename)
        return p if isinstance(p, str) and os.path.isfile(p) else None
    except Exception:
        return None


def hf_kwargs(repo_or_path) -> dict:
    """Extra kwargs for a transformers ``from_pretrained``: a repo id (not a local folder) may be fetched only with
    the node's permission; without it the call uses the Hugging Face cache or fails with transformers' own message,
    which the callers treat as a missing model."""
    if repo_or_path and not os.path.isdir(str(repo_or_path)) and not downloads_allowed():
        return {"local_files_only": True}
    return {}


def require_hub_repo(repo: str, *, target_hint: str = "", weights_missing: bool = False) -> None:
    """torch.hub fetches a GitHub repo (and, with pretrained=True, its weights) on first use. Without the node's
    permission: stop with the missing-model message unless the repo is cached and no weights would be fetched."""
    if downloads_allowed():
        return
    if weights_missing:
        raise ModelNotDownloaded(f"{repo} weights", target=target_hint, source=f"https://github.com/{repo}")
    try:
        import torch
        hub = torch.hub.get_dir()
    except Exception:
        hub = ""
    owner, _, name = repo.partition("/")
    cached = hub and any(os.path.isdir(os.path.join(hub, f"{owner}_{name}_{b}")) for b in ("master", "main"))
    if not cached:
        raise ModelNotDownloaded(repo, target=target_hint or (hub or "the torch hub folder"),
                                 source=f"https://github.com/{repo}")


def sam_model_status(model_info) -> str:
    """Owner D7.91 status line for a SAM_MODEL from the SAM Model Loader."""
    if isinstance(model_info, dict) and model_info.get("model_free"):
        return model_status(fallback_why=(model_info.get("fallback_from") or {}).get("why", ""))
    return model_status(path=model_info.get("model_path", "") if isinstance(model_info, dict) else "")
