"""C2C Distributed nodes: GPU Inventory, Sequence Parallel."""
from __future__ import annotations

import json

from . import enabled, kill_switch_reason
from . import inventory as inv
from . import usp

CATEGORY = "C2C/Distributed"


def _refuse_if_off():
    if not enabled():
        raise RuntimeError(kill_switch_reason())


class C2CGPUInventory:
    """What this machine can do with several GPUs: models, architecture, links, measured backends, usable modes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "probe_backends": ("BOOLEAN", {"default": True, "tooltip": "Really create a NCCL / gloo process group in "
                                           "a subprocess to see whether it works (a few seconds, cached until "
                                           "ComfyUI restarts). Off: report only what the hardware says."}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("summary", "report_json")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = ("Lists the GPUs (architecture Turing to Blackwell, memory, BF16 / FP8 / FP4 support, whether this "
                   "torch build has kernels for them), how they are linked (P2P, NVLink), Grace / aarch64 hosts, "
                   "whether NCCL / gloo really work here, and which multi-GPU modes this box can run.")

    @classmethod
    def IS_CHANGED(cls, probe_backends):
        return f"{probe_backends}"

    def run(self, probe_backends):
        _refuse_if_off()
        rep = inv.inventory(probe_backends=bool(probe_backends))
        text = inv.summary(rep)
        return {"ui": {"text": [text]}, "result": (text, json.dumps(rep, indent=1, default=str))}


class C2CSequenceParallel:
    """Split every model call over several GPUs: batch items for any model, tokens for Wan / MiniMax H3."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "gpus": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip": "How many GPUs to use; 0 = all that "
                             "ComfyUI sees. Each GPU holds a full copy of the model."}),
            "strategy": (["auto", "ulysses", "ring"], {"default": "auto", "tooltip": "auto: Ulysses as far as the "
                         "head count and fast links allow, Ring for the rest (USP). ulysses: prefer head-parallel "
                         "all-to-all (best on NVLink). ring: K/V blocks passed around (best across slow links)."}),
            "min_tokens_per_gpu": ("INT", {"default": 1024, "min": 1, "max": 1 << 20, "tooltip": "Calls with fewer "
                                   "tokens per GPU run on one GPU: below this the exchange costs more than it saves."}),
        }, "optional": {
            "fast_link_gpus": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip": "How many GPUs share fast "
                               "links (NVLink domain). 0 = all of them. Ulysses stays inside this many GPUs."}),
        }}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "plan")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Splits every model call over the GPUs of this machine. Any model: the call's batch items (cond "
                   "and uncond, an image batch, Tile Split's tiles) go to different GPUs, with no exchange at all. "
                   "Wan 2.1 / 2.2 and MiniMax H3 also split the tokens of one item (Ulysses all-to-all / Ring, exact). "
                   "One model copy per GPU, made by ComfyUI's own multi-GPU clones. Works on Windows and Linux; with "
                   "one GPU the model passes through unchanged. Put it last in the model chain (after LoRA / attention "
                   "nodes). Replaces core's 'MultiGPU CFG Split' for this model.")

    def run(self, model, gpus, strategy, min_tokens_per_gpu, fast_link_gpus=0):
        _refuse_if_off()
        m, report, _state = usp.apply(model, gpus=gpus, prefer=strategy, min_tokens_per_gpu=min_tokens_per_gpu,
                                      fast_domain=fast_link_gpus)
        text = "\n".join(report)
        return {"ui": {"text": [text]}, "result": (m, text)}


NODE_CLASS_MAPPINGS = {
    "C2CGPUInventory": C2CGPUInventory,
    "C2CSequenceParallel": C2CSequenceParallel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "C2CGPUInventory": "C2C GPU Inventory",
    "C2CSequenceParallel": "C2C Multi-GPU Split",
}
