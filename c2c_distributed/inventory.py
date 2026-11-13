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
         "how": "C2C Multi-GPU Split node, any model: cond / uncond, image batches and tiles over all GPUs "
                "(core's 'MultiGPU CFG Split' uses at most one GPU per cond)"
         if comfy.get("core_multigpu") else "needs ComfyUI core with comfy/multigpu.py (0.39+)",
         "why_not": None if multi else "one GPU"},
        {"mode": "sequence parallel, one box (Ulysses / Ring / USP)", "can_run": multi,
         "how": "C2C Multi-GPU Split node, Wan / MiniMax H3: threads, one model copy per GPU, tensors exchanged "
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


# ── A26 L9.213: real qualification ─────────────────────────────────────────────────────────────────────────────
# A one-rank process group proves nothing about two GPUs exchanging data. qualify() runs one PROCESS PER PHYSICAL
# DEVICE, checks the devices are distinct (UUIDs), runs all-reduce / all-gather / reduce-scatter with correctness
# checks, measures latency and bandwidth, and checks that a missing rank is detected within the timeout.
# Labels: MULTI_GPU_VERIFIED only for >= 2 distinct physical GPUs that passed; SINGLE_GPU_ONLY with one GPU (the
# harness itself is then exercised on CPU ranks and labelled SIMULATED); FAILED when real GPUs did not pass.

_RANK = r"""
import datetime, json, os, statistics, sys, time
import torch, torch.distributed as dist
rank, world, store, backend, kind, timeout_s, out = (int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4],
                                                     sys.argv[5], float(sys.argv[6]), sys.argv[7])
res = {"rank": rank, "ok": {}, "errors": {}}
try:
    dev = torch.device("cpu")
    if kind == "cuda":
        torch.cuda.set_device(rank)
        dev = torch.device("cuda", rank)
        res["uuid"] = str(getattr(torch.cuda.get_device_properties(rank), "uuid", ""))
        res["device"] = torch.cuda.get_device_name(rank)
    dist.init_process_group(backend, init_method="file:///" + store.replace(os.sep, "/"), rank=rank,
                            world_size=world, timeout=datetime.timedelta(seconds=timeout_s))
    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
    # correctness
    t = torch.full((1024,), float(rank + 1), device=dev)
    dist.all_reduce(t)
    res["ok"]["all_reduce"] = bool(torch.allclose(t, torch.full_like(t, world * (world + 1) / 2)))
    parts = [torch.empty(256, device=dev) for _ in range(world)]
    dist.all_gather(parts, torch.full((256,), float(rank), device=dev))
    res["ok"]["all_gather"] = all(bool(torch.all(p == i)) for i, p in enumerate(parts))
    try:
        inp = torch.arange(world * 128, dtype=torch.float32, device=dev) + rank
        o = torch.empty(128, device=dev)
        dist.reduce_scatter_tensor(o, inp)
        want = (torch.arange(world * 128, dtype=torch.float32, device=dev) * world
                + sum(range(world)))[rank * 128:(rank + 1) * 128]
        res["ok"]["reduce_scatter"] = bool(torch.allclose(o, want))
    except Exception as exc:
        res["ok"]["reduce_scatter"] = None
        res["errors"]["reduce_scatter"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    # latency (small) and bandwidth (64 MiB)
    small = torch.ones(8, device=dev)
    lat = []
    for _ in range(25):
        sync(); t0 = time.perf_counter(); dist.all_reduce(small); sync(); lat.append(time.perf_counter() - t0)
    big = torch.ones(16 * 1024 * 1024, device=dev)
    bw = []
    for _ in range(5):
        sync(); t0 = time.perf_counter(); dist.all_reduce(big); sync(); bw.append(time.perf_counter() - t0)
    nbytes = big.numel() * 4
    res["latency_us"] = statistics.median(lat[5:]) * 1e6
    res["bandwidth_gbps"] = (2 * (world - 1) / world * nbytes) / statistics.median(bw[1:]) / 1e9   # bus bandwidth
    # a missing rank must be detected within the timeout, not hang
    if rank == world - 1:
        res["timeout_role"] = "absent"                       # alive but silent: a HUNG rank, not a closed socket
        open(out, "w").write(json.dumps(res))
        time.sleep(timeout_s + 5)
    else:
        t0 = time.perf_counter()
        try:
            dist.all_reduce(torch.ones(4, device=dev))
            sync()
            res["timeout_detected"] = False
        except Exception as exc:
            res["timeout_detected"] = True
            res["timeout_error"] = f"{type(exc).__name__}"
        res["timeout_seconds"] = time.perf_counter() - t0
except Exception as exc:
    res["errors"]["setup"] = f"{type(exc).__name__}: {str(exc)[:300]}"
open(out, "w").write(json.dumps(res))
os._exit(0)
"""


def _driver_version() -> Optional[str]:
    try:
        cp = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True, timeout=20)
        return cp.stdout.strip().splitlines()[0] if cp.returncode == 0 and cp.stdout.strip() else None
    except Exception:  # noqa: BLE001
        return None


def topology() -> Optional[str]:
    """`nvidia-smi topo -m` as printed (PCIe / NVLink matrix), or None where unavailable."""
    try:
        cp = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=20)
        return cp.stdout.strip() if cp.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def physical_devices() -> List[dict]:
    import torch
    out = []
    if not torch.cuda.is_available():
        return out
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        out.append({"index": i, "uuid": str(getattr(p, "uuid", "")) or None, "name": p.name,
                    "compute_capability": f"{p.major}.{p.minor}", "total_gb": round(total / 2 ** 30, 2),
                    "free_gb": round(free / 2 ** 30, 2)})
    return out


def run_ranks(world: int, backend: str, kind: str, timeout_s: float = 15.0, wall_s: float = 240.0) -> dict:
    """One subprocess per rank (rank r on device r for kind 'cuda'); returns per-rank results."""
    d = tempfile.mkdtemp(prefix="c2c_qual_")
    store = os.path.join(d, "store")
    procs, outs = [], []
    for r in range(world):
        o = os.path.join(d, f"rank{r}.json")
        outs.append(o)
        procs.append(subprocess.Popen([sys.executable, "-c", _RANK, str(r), str(world), store, backend, kind,
                                       str(timeout_s), o], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                      cwd=tempfile.gettempdir()))
    errs = []
    for pr in procs:
        try:
            _, e = pr.communicate(timeout=wall_s)
            errs.append((e or b"").decode(errors="replace")[-400:])
        except subprocess.TimeoutExpired:
            pr.kill()
            errs.append("rank process hung past the wall clock and was killed")
    ranks = []
    for r, o in enumerate(outs):
        try:
            with open(o, encoding="utf-8") as fh:
                ranks.append(json.load(fh))
        except Exception:  # noqa: BLE001
            ranks.append({"rank": r, "ok": {}, "errors": {"process": errs[r] or "no result"}})
    return {"world": world, "backend": backend, "device_kind": kind, "ranks": ranks}


def _assess(run: dict) -> dict:
    ranks = run["ranks"]
    ops = ("all_reduce", "all_gather", "reduce_scatter")
    res = {}
    setup_failed = any(k in r.get("errors", {}) for r in ranks for k in ("setup", "process")) or not ranks
    for op in ops:
        vals = [r["ok"].get(op) for r in ranks]
        if setup_failed:
            res[op] = "not run (the process group could not start)"
        elif all(v is None for v in vals) and all(op in r.get("errors", {}) for r in ranks):
            res[op] = "unsupported by this backend"
        else:
            res[op] = all(v is True for v in vals)
    waiting = [r for r in ranks if r.get("timeout_role") != "absent"]
    res["timeout_detected"] = bool(waiting) and all(r.get("timeout_detected") for r in waiting)
    res["timeout_seconds"] = max((r.get("timeout_seconds") or 0) for r in waiting) if waiting else None
    lat = [r["latency_us"] for r in ranks if r.get("latency_us")]
    bw = [r["bandwidth_gbps"] for r in ranks if r.get("bandwidth_gbps")]
    res["latency_us"] = round(max(lat), 1) if lat else None
    res["bus_bandwidth_gbps"] = round(min(bw), 3) if bw else None
    res["errors"] = {r["rank"]: r["errors"] for r in ranks if r.get("errors")}
    res["passed"] = all(res[op] is True or res[op] == "unsupported by this backend" for op in ops) \
        and res["all_reduce"] is True and res["all_gather"] is True and res["timeout_detected"] \
        and not any("setup" in e or "process" in e for e in res["errors"].values())
    return res


def worker_mechanism(devs: List[dict]) -> dict:
    """Can C2C's thread-per-GPU workers (the Multi-GPU Split mechanism) compute on each device at once?"""
    import threading

    import torch
    out: Dict[int, object] = {}

    def work(i):
        try:
            d = torch.device("cuda", i)
            a = torch.randn(512, 512, device=d, generator=torch.Generator(device=d).manual_seed(i))
            ref = (a.double() @ a.double()).float()
            out[i] = bool(torch.allclose(a @ a, ref, rtol=1e-2, atol=1e-2))
        except Exception as exc:  # noqa: BLE001
            out[i] = f"{type(exc).__name__}: {exc}"[:200]
    th = [threading.Thread(target=work, args=(d["index"],)) for d in devs]
    [t.start() for t in th]
    [t.join(120) for t in th]
    return {"per_device": out, "ok": bool(devs) and all(v is True for v in out.values())}


def qualify(timeout_s: float = 15.0) -> dict:
    """The A26 4.2 qualification. Runs real subprocess ranks; takes tens of seconds."""
    import torch
    devs = physical_devices()
    uuids = [d["uuid"] for d in devs]
    distinct = len({u for u in uuids if u}) if uuids else 0
    nccl = backend_usable("nccl") if devs else {"usable": False}
    gloo = backend_usable("gloo")
    rep = {"versions": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                        "driver": _driver_version()},
           "devices": devs, "distinct_physical_devices": distinct,
           "p2p": p2p_matrix(len(devs)) if devs else [], "topology": topology(),
           "backends": {"nccl": bool(nccl.get("usable")), "gloo": bool(gloo.get("usable"))},
           "worker_mechanism": worker_mechanism(devs) if devs else {"ok": False, "per_device": {}}}
    if distinct >= 2:
        backend = "nccl" if rep["backends"]["nccl"] else "gloo"
        run = run_ranks(distinct, backend, "cuda", timeout_s)
        seen = {r.get("uuid") for r in run["ranks"] if r.get("uuid")}
        rep["multi_rank"] = {**run, **_assess(run), "distinct_uuids_seen": len(seen)}
        ok = rep["multi_rank"]["passed"] and len(seen) == distinct and rep["worker_mechanism"]["ok"]
        rep["status"] = "MULTI_GPU_VERIFIED" if ok else "FAILED"
    else:
        run = run_ranks(2, "gloo", "cpu", timeout_s)                  # the harness itself, on CPU ranks
        rep["harness_selftest"] = {**run, **_assess(run), "label": "SIMULATED"}
        rep["status"] = "SINGLE_GPU_ONLY" if distinct == 1 else "NO_GPU"
    h = rep.get("harness_selftest")
    harness = "" if h is None else (
        " (the harness passed on CPU ranks: SIMULATED, not a GPU result)" if h.get("passed") else
        " (the harness could not run multi-process collectives with this torch build - e.g. Windows gloo has no "
        "transport; run the qualification on Linux / WSL)")
    rep["status_meaning"] = {
        "MULTI_GPU_VERIFIED": "collectives passed between distinct physical GPUs on this machine",
        "SINGLE_GPU_ONLY": "one physical GPU: nothing multi-GPU can be verified here" + harness,
        "NO_GPU": "no CUDA GPU" + harness,
        "FAILED": "several GPUs, but a collective, the timeout check or the worker mechanism failed"}[rep["status"]]
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
    q = rep.get("qualification")
    if q:
        lines.append(f"Qualification: {q['status']} - {q['status_meaning']}")
        mr = q.get("multi_rank") or q.get("harness_selftest")
        if mr:
            lab = mr.get("label", "real devices")
            lines.append(f"  {mr['world']} ranks, {mr['backend']} on {mr['device_kind']} ({lab}): all-reduce "
                         f"{mr['all_reduce']}, all-gather {mr['all_gather']}, reduce-scatter {mr['reduce_scatter']}, "
                         f"missing rank detected {mr['timeout_detected']}"
                         + (f" in {mr['timeout_seconds']:.1f} s" if mr.get("timeout_seconds") else "")
                         + (f", latency {mr['latency_us']} us, bus bandwidth {mr['bus_bandwidth_gbps']} GB/s"
                            if mr.get("latency_us") else ""))
    return "\n".join(lines)
