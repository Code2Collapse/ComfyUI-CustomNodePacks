"""What multi-GPU work can this box do? GPUs, architecture, links, and which parallel modes can actually run.

Everything here is MEASURED where it can be (a backend counts as usable only after a process group was really
created in a subprocess), and reported as unknown otherwise - never assumed from the OS name.
Nothing runs at import time; the probes run when the Inventory node (or a caller) asks.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

# compute capability -> (architecture, typical parts). Sources: NVIDIA CUDA GPU compute capability table.
_ARCH = {
    (7, 0): ("Volta", "V100"),
    (7, 2): ("Volta", "Jetson Xavier"),
    (7, 5): ("Turing", "RTX 20, T4, Quadro RTX"),
    (8, 0): ("Ampere", "A100, A30"),
    (8, 6): ("Ampere", "RTX 30, A40, A10, RTX A6000"),
    (8, 7): ("Ampere", "Jetson Orin"),
    (8, 9): ("Ada Lovelace", "RTX 40, L40S, L4, RTX 6000 Ada"),
    (9, 0): ("Hopper", "H100, H200, GH200"),
    (10, 0): ("Blackwell", "B200, GB200"),
    (10, 3): ("Blackwell Ultra", "B300, GB300"),
    (11, 0): ("Blackwell", "Jetson Thor"),
    (12, 0): ("Blackwell", "RTX 50, RTX PRO 6000 Blackwell"),
    (12, 1): ("Blackwell", "GB10 (DGX Spark)"),
}

# Arm CPU part ids (from /proc/cpuinfo) that identify NVIDIA superchip hosts.
_ARM_PARTS = {"0xd4f": "NVIDIA Grace (Neoverse V2)", "0xd85": "Cortex-X925 (GB10 / DGX Spark)",
              "0xd87": "Cortex-A725 (GB10 / DGX Spark)"}


def arch_info(major: int, minor: int) -> dict:
    name, parts = _ARCH.get((major, minor), (None, None))
    if name is None:
        name = {7: "Volta/Turing", 8: "Ampere/Ada", 9: "Hopper", 10: "Blackwell", 11: "Blackwell",
                12: "Blackwell"}.get(major, f"sm_{major}{minor}")
    cc = (major, minor)
    return {
        "compute_capability": f"{major}.{minor}",
        "sm": f"sm_{major}{minor}",
        "architecture": name,
        "typical_parts": parts,
        "bf16": cc >= (8, 0),
        "fp8_tensor_cores": cc >= (8, 9),
        "fp4_tensor_cores": cc >= (10, 0),
        "flash_attention_3": cc == (9, 0),
    }


def kernel_support(major: int, minor: int, arch_list: List[str]) -> Tuple[str, str]:
    """Does this torch build carry kernels for the GPU? -> (status, detail)."""
    want = f"sm_{major}{minor}"
    if want in arch_list:
        return "native", f"{want} cubins in this torch build"
    same = sorted(a for a in arch_list if a.startswith(f"sm_{major}") and len(a) == len(want)
                  and a[len(f'sm_{major}'):].isdigit() and int(a[len(f'sm_{major}'):]) <= minor)
    if same:
        return "compatible", f"runs the {same[-1]} cubins (same major, binary compatible)"
    ptx = sorted(a for a in arch_list if a.startswith("compute_") and a[8:].isdigit()
                 and int(a[8:]) <= major * 10 + minor)
    if ptx:
        return "ptx-jit", f"JIT from {ptx[-1]} PTX (slow first run, then cached)"
    return "unsupported", f"this torch build has no kernels for {want} (has {', '.join(arch_list) or 'none'})"


def host_info() -> dict:
    info = {"os": platform.system(), "machine": platform.machine(), "python": platform.python_version(),
            "cpu": platform.processor() or None, "superchip_host": None}
    if info["machine"].lower() in ("aarch64", "arm64") and os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
                parts = {line.split(":", 1)[1].strip().lower() for line in fh if line.startswith("CPU part")}
            names = sorted({_ARM_PARTS[p] for p in parts if p in _ARM_PARTS})
            info["superchip_host"] = ", ".join(names) or None
        except OSError:
            pass
    return info


def _nvml_details() -> Dict[str, dict]:
    """Per PCI bus id: memory, NVLink peers, addressing mode (ATS / HMM on Grace). {} when NVML is absent."""
    try:
        import pynvml
    except Exception:  # noqa: BLE001
        return {}
    out: Dict[str, dict] = {}
    try:
        pynvml.nvmlInit()
    except Exception:  # noqa: BLE001
        return {}
    try:
        driver = pynvml.nvmlSystemGetDriverVersion()
        driver = driver.decode() if isinstance(driver, bytes) else driver
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            bus = pynvml.nvmlDeviceGetPciInfo(h).busId
            bus = (bus.decode() if isinstance(bus, bytes) else bus).lower()
            d = {"driver": driver, "nvlink_peers": [], "nvlink_links": 0}
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                d["memory_total_gb"] = round(mem.total / 2 ** 30, 2)
                d["memory_free_gb"] = round(mem.free / 2 ** 30, 2)
            except Exception:  # noqa: BLE001
                pass
            for link in range(getattr(pynvml, "NVML_NVLINK_MAX_LINKS", 18)):
                try:
                    if pynvml.nvmlDeviceGetNvLinkState(h, link) != pynvml.NVML_FEATURE_ENABLED:
                        continue
                    remote = pynvml.nvmlDeviceGetNvLinkRemotePciInfo(h, link).busId
                    d["nvlink_links"] += 1
                    d["nvlink_peers"].append((remote.decode() if isinstance(remote, bytes) else remote).lower())
                except Exception:  # noqa: BLE001 - link absent / unsupported
                    break
            try:
                mode = pynvml.nvmlDeviceGetAddressingMode(h)
                d["addressing_mode"] = {1: "HMM", 2: "ATS"}.get(getattr(mode, "value", mode), str(mode))
            except Exception:  # noqa: BLE001 - older NVML
                pass
            out[_norm_bus(bus)] = d
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass
    return out


def _norm_bus(bus: str) -> str:
    """'00000000:65:00.0' and '0000:65:00.0' name the same device."""
    bus = bus.lower()
    dom, _, rest = bus.partition(":")
    return f"{int(dom, 16):04x}:{rest}" if rest and all(c in "0123456789abcdef" for c in dom) else bus


def gpus() -> List[dict]:
    import torch
    if not torch.cuda.is_available():
        return []
    arch_list = list(torch.cuda.get_arch_list())
    nvml = _nvml_details()
    out = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        g = {"index": i, "name": p.name, "memory_gb": round(p.total_memory / 2 ** 30, 2),
             "multiprocessors": p.multi_processor_count}
        g.update(arch_info(p.major, p.minor))
        g["kernels"], g["kernels_detail"] = kernel_support(p.major, p.minor, arch_list)
        try:                                # torch exposes the PCI location as integers
            bus = "{:04x}:{:02x}:{:02x}.0".format(int(p.pci_domain_id), int(p.pci_bus_id), int(p.pci_device_id))
        except Exception:  # noqa: BLE001 - older torch
            bus = None
        if bus:
            g["pci_bus"] = _norm_bus(str(bus))
            g.update({k: v for k, v in nvml.get(g["pci_bus"], {}).items() if k != "nvlink_peers"})
            g["_nvlink_peers"] = nvml.get(g["pci_bus"], {}).get("nvlink_peers", [])
        out.append(g)
    by_bus = {g.get("pci_bus"): g["index"] for g in out}
    for g in out:
        g["nvlink_to"] = sorted({by_bus[b] for b in (_norm_bus(x) for x in g.pop("_nvlink_peers", []))
                                 if b in by_bus})
    return out


def p2p_matrix(n: int) -> List[List[bool]]:
    import torch
    m = [[i == j for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                try:
                    m[i][j] = bool(torch.cuda.can_device_access_peer(i, j))
                except Exception:  # noqa: BLE001
                    m[i][j] = False
    return m


_PROBE = r"""
import json, os, sys, tempfile
import torch, torch.distributed as dist
backend = sys.argv[1]
res = {"backend": backend, "available": getattr(dist, f"is_{backend}_available")(), "usable": False, "error": None}
if res["available"]:
    store = os.path.join(tempfile.mkdtemp(), "store")
    try:
        if backend == "nccl":
            torch.cuda.set_device(0)
        dist.init_process_group(backend, init_method="file:///" + store.replace(os.sep, "/"), rank=0, world_size=1)
        t = torch.ones(4, device="cuda" if backend == "nccl" else "cpu")
        dist.all_reduce(t)
        res["usable"] = bool(t.sum().item() == 4)
        dist.destroy_process_group()
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {exc}"[:300]
print("C2C_PROBE " + json.dumps(res))
"""


@lru_cache(maxsize=4)
def backend_usable(backend: str, timeout: float = 90.0) -> dict:
    """Create a one-rank process group with `backend` in a fresh subprocess. Cached for the process lifetime."""
    if backend not in ("nccl", "gloo"):
        raise ValueError(backend)
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", env.get("CUDA_VISIBLE_DEVICES", "0"))
    try:
        cp = subprocess.run([sys.executable, "-c", _PROBE, backend], capture_output=True, text=True,
                            timeout=timeout, env=env, cwd=tempfile.gettempdir())
    except Exception as exc:  # noqa: BLE001
        return {"backend": backend, "available": None, "usable": False, "error": f"probe failed to run: {exc}"}
    for line in (cp.stdout or "").splitlines():
        if line.startswith("C2C_PROBE "):
            return json.loads(line[len("C2C_PROBE "):])
    tail = (cp.stderr or cp.stdout or "").strip().splitlines()[-1:] or ["no output"]
    return {"backend": backend, "available": None, "usable": False, "error": f"probe crashed: {tail[0][:300]}"}


def comfy_devices() -> dict:
    """What ComfyUI itself will use (its device list and whether core multi-GPU is present)."""
    out = {"core_multigpu": False, "devices": []}
    try:
        import comfy.model_management as mm
        out["devices"] = [str(d) for d in mm.get_all_torch_devices(exclude_current=False)]
    except Exception:  # noqa: BLE001
        pass
    try:
        import comfy.multigpu  # noqa: F401
        out["core_multigpu"] = True
    except Exception:  # noqa: BLE001
        pass
    return out


def modes(gpu_list: List[dict], nccl: dict, comfy: dict, p2p: List[List[bool]]) -> List[dict]:
    """Which parallel modes can run here, and how to use them."""
    n = len(gpu_list)
    multi = n >= 2
    any_p2p = any(p2p[i][j] for i in range(n) for j in range(n) if i != j) if multi else False
    nccl_ok = bool(nccl.get("usable"))
    rows = [
        {"mode": "data parallel (whole jobs per GPU / per machine)", "can_run": True,
         "how": "C2C Farm backends (renderfarm/) or one ComfyUI per GPU; any OS"},
        {"mode": "CFG / batch split (one box)", "can_run": multi and comfy.get("core_multigpu", False),
         "how": "core node 'MultiGPU CFG Split' (advanced/multigpu) after the model loader"
         if comfy.get("core_multigpu") else "needs ComfyUI core with comfy/multigpu.py",
         "why_not": None if multi else "one GPU"},
        {"mode": "sequence parallel, one box (Ulysses / Ring / USP)", "can_run": multi,
         "how": "C2C Sequence Parallel node (this package): threads, one model copy per GPU, tensors exchanged "
                "GPU-to-GPU" + ("" if any_p2p or not multi else " - NOTE: no P2P between these GPUs, exchanges go "
                                                                    "through host memory (slower)"),
         "why_not": None if multi else "one GPU"},
        {"mode": "sequence parallel / FSDP across processes or machines", "can_run": multi and nccl_ok,
         "how": "torch.distributed with NCCL (Linux)",
         "why_not": None if nccl_ok else (nccl.get("error") or "NCCL is not available in this torch build")},
    ]
    return rows


def inventory(probe_backends: bool = True) -> dict:
    import torch
    gl = gpus()
    rep = {
        "host": host_info(),
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda,
                  "arch_list": list(torch.cuda.get_arch_list()) if torch.cuda.is_available() else []},
        "gpus": gl,
        "p2p": p2p_matrix(len(gl)) if gl else [],
        "comfy": comfy_devices(),
        "backends": {},
    }
    if probe_backends:
        rep["backends"] = {"nccl": backend_usable("nccl") if gl else
                           {"backend": "nccl", "available": False, "usable": False, "error": "no CUDA GPU"},
                           "gloo": backend_usable("gloo")}
    rep["modes"] = modes(gl, rep["backends"].get("nccl", {}), rep["comfy"], rep["p2p"])
    return rep


def summary(rep: dict) -> str:
    lines = []
    h = rep["host"]
    lines.append(f"Host: {h['os']} {h['machine']}" + (f" - {h['superchip_host']}" if h.get("superchip_host") else ""))
    t = rep["torch"]
    lines.append(f"torch {t['version']} (CUDA {t['cuda']}), kernels: {', '.join(t['arch_list']) or 'none'}")
    if not rep["gpus"]:
        lines.append("No CUDA GPU visible.")
    for g in rep["gpus"]:
        extra = []
        if g.get("nvlink_to"):
            extra.append(f"NVLink to {g['nvlink_to']}")
        if g.get("addressing_mode"):
            extra.append(f"addressing {g['addressing_mode']}")
        flags = [f for f, k in (("BF16", "bf16"), ("FP8", "fp8_tensor_cores"), ("FP4", "fp4_tensor_cores")) if g[k]]
        lines.append(f"GPU {g['index']}: {g['name']} - {g['architecture']} {g['sm']}, {g['memory_gb']} GB, "
                     f"{'/'.join(flags) or 'FP16 only'}, kernels {g['kernels']}"
                     + (f" ({'; '.join(extra)})" if extra else ""))
    n = len(rep["gpus"])
    if n > 1:
        pairs = [f"{i}<->{j}" for i in range(n) for j in range(i + 1, n) if rep["p2p"][i][j]]
        lines.append("P2P: " + (", ".join(pairs) if pairs else "none (exchanges go through host memory)"))
    for name, b in (rep.get("backends") or {}).items():
        lines.append(f"{name}: " + ("usable" if b.get("usable") else f"not usable - {b.get('error') or 'not in this torch build'}"))
    lines.append("Modes:")
    for m in rep["modes"]:
        lines.append(f"  [{'yes' if m['can_run'] else 'no '}] {m['mode']}: "
                     + (m["how"] if m["can_run"] else (m.get("why_not") or m["how"])))
    return "\n".join(lines)
