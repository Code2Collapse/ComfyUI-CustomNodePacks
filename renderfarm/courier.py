"""C2C Farm invisible courier — media handoff between local graph and cloud backend.

Outbound:  scan the prompt JSON for local media references, upload each file
once through the configured storage adapter, and rewrite the prompt so the
backend pulls the media from a presigned URL instead of a local path.

Two rewrite modes (per-backend, `url_loader_map` in backends.json):
  * mapped   — {"LoadImage": {"class_type": "LoadImageFromUrl", "input": "url"}}
               swaps the loader node for a URL-capable class on the backend image.
  * in-place — no map entry: the path STRING is replaced by the URL (the backend
               image must ship URL-tolerant loaders).

Inbound: `download_results` pulls any http(s) media URL from a job's outputs
back into the local ComfyUI input folder (input/c2c_farm_results/<job_id>/) so the
next node in the local graph can consume it.
"""

from __future__ import annotations

import copy
import logging
import os

log = logging.getLogger("C2C.Farm.courier")

MEDIA_EXTS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".exr", ".dpx",
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".gif",
    ".wav", ".mp3", ".flac", ".ogg", ".m4a",
    ".latent", ".safetensors", ".npy",
}


def _default_resolver(value: str) -> str | None:
    """Resolve a workflow string to an existing local media file, else None."""
    v = value.strip()
    # ComfyUI annotated names: "sub/file.png [input]" / "[output]" / "[temp]"
    annotated = v.endswith("]") and "[" in v
    base = v[: v.rfind("[")].strip() if annotated else v
    if os.path.splitext(base)[1].lower() not in MEDIA_EXTS:
        return None
    if os.path.isabs(base):
        return base if os.path.isfile(base) else None
    try:
        import folder_paths
        if annotated:
            p = folder_paths.get_annotated_filepath(v)
            return p if p and os.path.isfile(p) else None
        p = os.path.join(folder_paths.get_input_directory(), base)
        return p if os.path.isfile(p) else None
    except Exception:  # noqa: BLE001 — outside ComfyUI (tests) only abs paths resolve
        return None


def find_local_media(prompt: dict, resolver=_default_resolver) -> list[dict]:
    """[{node_id, input_key, value, path}] for every resolvable media reference."""
    found = []
    for node_id, node in prompt.items():
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if not isinstance(inputs, dict):
            continue
        for key, value in inputs.items():
            if not isinstance(value, str) or not value:
                continue
            path = resolver(value)
            if path:
                found.append({"node_id": node_id, "input_key": key,
                              "value": value, "path": path})
    return found


def prepare_prompt(prompt: dict, backend_cfg: dict, storage=None,
                   resolver=_default_resolver) -> tuple[dict, dict]:
    """Upload referenced media + rewrite the prompt. Returns (new_prompt, upload_map).

    `storage` is created lazily ONLY if local media is actually found, so
    URL-free workflows never require storage configuration.
    """
    refs = find_local_media(prompt, resolver)
    if not refs:
        return prompt, {}
    if storage is None:
        from .storage import get_storage
        storage = get_storage()

    new_prompt = copy.deepcopy(prompt)
    loader_map = backend_cfg.get("url_loader_map") or {}
    uploads: dict[str, str] = {}  # local path -> url (each file uploaded once)

    for ref in refs:
        path = ref["path"]
        if path not in uploads:
            log.info("courier upload: %s", path)
            uploads[path] = storage.upload(path)
        url = uploads[path]
        node = new_prompt[ref["node_id"]]
        mapping = loader_map.get(node.get("class_type", ""))
        if mapping:
            node["class_type"] = mapping["class_type"]
            node["inputs"] = {mapping["input"]: url}
        else:
            node["inputs"][ref["input_key"]] = url
    return new_prompt, uploads


def results_dir(job_id: str) -> str:
    """Local landing folder for a job's outputs (inside ComfyUI input/)."""
    try:
        import folder_paths
        base = folder_paths.get_input_directory()
    except Exception:  # noqa: BLE001 — tests
        base = os.path.join(os.getcwd(), "input")
    d = os.path.join(base, "c2c_farm_results", job_id)
    os.makedirs(d, exist_ok=True)
    return d


def download_results(urls: list[str], job_id: str, storage=None) -> list[str]:
    """Pull output media URLs back into input/c2c_farm_results/<job_id>/."""
    from .storage.base_storage import _http_download
    out = []
    dest = results_dir(job_id)
    for i, url in enumerate(urls):
        name = os.path.basename(url.split("?", 1)[0]) or f"result_{i}"
        local = os.path.join(dest, name)
        if storage is not None:
            out.append(storage.download(url, local))
        else:
            out.append(_http_download(url, local))
    return out
