"""Reference-image sockets → one upload per picture.

Kept free of torch on purpose: the split is duck-typed over `dim()` / `shape` /
indexing so the socket arithmetic stays testable where ComfyUI's runtime is
not installed (the unittest suites run stdlib-only)."""

from typing import Any, List, Optional


def frames(*images: Optional[Any]) -> List[Any]:
    """Every picture wired into the reference sockets, in socket order, with a
    batched IMAGE ([B, H, W, C]) contributing each of its B frames. A user who
    joins pictures with ComfyUI's Batch Images node expects every one of them
    to reach the model, not just the first."""
    result: List[Any] = []
    for image in images:
        if image is None:
            continue
        if image.dim() == 4:
            result.extend(image[index] for index in range(image.shape[0]))
        else:
            result.append(image)
    return result
