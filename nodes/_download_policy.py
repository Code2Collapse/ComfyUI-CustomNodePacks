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
from typing import Iterator, Optional

_ALLOWED: contextvars.ContextVar[bool] = contextvars.ContextVar("c2c_downloads_allowed", default=False)


@contextlib.contextmanager
def allow_downloads(flag: bool) -> Iterator[None]:
    token = _ALLOWED.set(bool(flag) and os.environ.get("C2C_NO_DOWNLOADS") != "1")
    try:
        yield
    finally:
        _ALLOWED.reset(token)


def downloads_allowed() -> bool:
    return bool(_ALLOWED.get()) and os.environ.get("C2C_NO_DOWNLOADS") != "1"


class ModelNotDownloaded(FileNotFoundError):
    """The model is not installed and the node may not download it. The message says what, where and how."""

    def __init__(self, model: str, *, target: str = "", source: str = "", widget: str = "allow_download"):
        self.model, self.target, self.source = model, target, source
        where = f" Put it in {target}" if target else ""
        frm = f" (from {source})" if source else ""
        super().__init__(
            f"Model '{model}' is not installed and downloading is off.{where}{frm}, or tick '{widget}' on the node "
            "to download it there.")


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
