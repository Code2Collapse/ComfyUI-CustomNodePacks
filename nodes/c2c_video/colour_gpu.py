"""OpenColorIO's own GPU processor for Load / Save Video colour management (L9.63, owner 2026-10-26: "Exact OCIO GPU
Acceleration + Automatic CPU Fallback"). Design: docs/LOAD_SAVE_VIDEO_CM_DESIGN.md section 10.

The SAME OCIO transform the CPU path runs - its generated GLSL 4.0 and LUT textures - compiled once per transform in a
headless OpenGL context (moderngl) that one worker thread owns, applied frame by frame (upload, one full-screen pass per
strip, float readback). Used only after this machine's GPU result matched the exact CPU processor within a strict
tolerance - worst case over ~1.9 million validation samples AND the clip's first frame - and, in Auto, was measured
faster end to end. Pixels outside the validated input domain are recomputed on the CPU. Anything missing or failing ->
exact CPU, with the reason. Imports only numpy / PyOpenColorIO / moderngl, so it runs outside ComfyUI too (the
evidence harness docs/evidence/L9.63/gpu_harness.py).
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Any, Callable

import numpy as np

log = logging.getLogger("c2c_video.colour_gpu")

KILL_SWITCH = "C2C_VIDEO_CM_GPU"
MODES = ("Auto", "Exact CPU", "OCIO GPU")
TOL_DISPLAY_ABS = 1e-4          # display-referred results in 0-1
TOL_LOG_ABS = 1e-3              # log / video code values
TOL_REL = 1e-3                  # scene-linear results, display results above 1: relative to the pixel's largest channel
REL_FLOOR = 1e-2                # ... absolute (1e-5) below this
SPEEDUP_NEEDED = 1.2            # Auto: the GPU must be at least this much faster end to end
# The validated input domains (MEASURED, docs/evidence/L9.63): far above ~160 scene-linear, very saturated colours go
# through ACES 2.0's numerics differently on the GPU and the CPU (0 vs 1 outputs); at display peak white the inverse
# tone scale is ill-conditioned. Pixels outside go to the CPU.
LIN_DOMAIN = (-0.025, 100.0)    # scene-linear values (a log space: the code values of these in it)
VIDEO_DOMAIN = (-0.05, 0.99)    # display / video code values
CODE_DOMAIN = (-0.05, 1.05)     # a log space the config cannot map, and an encoding the config does not name
TILE_PIXELS = 1 << 23           # pixels per GPU pass (~235 MB of float textures): larger frames go in strips
N_RANDOM = 1 << 20              # random validation samples (fixed seed)

VS = """#version 400
in vec2 in_pos;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
"""
FS = """#version 400
uniform sampler2D c2c_src;
out vec4 c2c_out;
%s
void main() {
    vec4 c = texelFetch(c2c_src, ivec2(gl_FragCoord.xy), 0);
    c2c_out = OCIOMain(c);
}
"""


def disabled() -> bool:
    return os.environ.get(KILL_SWITCH, "").strip().lower() in ("0", "off", "false", "no")


# ---- the input domain the gate validates (and the runtime routes by) ------------------------------------------------
# A domain is ("lin", lo, hi) in scene-linear values or ("code", lo, hi) in code values.

def _lin_from_code(c):
    """ACEScct's curve: how scene-linear validation samples are spaced (dense in the shadows)."""
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.155251141552511, (c - 0.0729055341958355) / 10.5402377416545,
                    2.0 ** (c * 17.52 - 9.72)).astype(np.float32)


def _code_from_lin(x):
    x = np.asarray(x, np.float64)
    return np.where(x <= 0.0078125, x * 10.5402377416545 + 0.0729055341958355,
                    (np.log2(np.maximum(x, 1e-30)) + 9.72) / 17.52).astype(np.float32)


def input_domain(encoding: str, to_input=None) -> tuple[str, float, float]:
    """The domain for an input of this OCIO encoding. A log space covers the code values of LIN_DOMAIN in it
    (`to_input`: an OCIO CPU processor scene_linear -> that space), so every log curve gets the same scene range."""
    if encoding == "scene-linear":
        return ("lin",) + LIN_DOMAIN
    if encoding in ("sdr-video", "hdr-video"):
        return ("code",) + VIDEO_DOMAIN
    if to_input is not None:
        try:
            v = np.array([[LIN_DOMAIN[0]] * 3, [LIN_DOMAIN[1]] * 3], np.float32)
            to_input.applyRGB(v)
            lo, hi = float(v[0].max()), float(v[1].min())    # greys: equal channels; else the narrower side
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                return ("code", lo, hi)
        except Exception:  # noqa: BLE001 - the generic range
            pass
    return ("code",) + CODE_DOMAIN


def out_of_domain(rgb: np.ndarray, domain) -> np.ndarray:
    """Flat indices of pixels with any channel outside the domain (or not finite)."""
    _, lo, hi = domain
    flat = rgb.reshape(-1, 3)
    if not flat.size:
        return np.zeros(0, np.intp)
    mn, mx = float(flat.min()), float(flat.max())        # a NaN anywhere propagates and fails this test
    if lo <= mn and mx <= hi:
        return np.zeros(0, np.intp)
    v = flat.reshape(-1)
    nz = np.flatnonzero(~((v >= lo) & (v <= hi)))       # NaN fails both comparisons
    px = nz // 3                                         # sorted: drop the repeats of a pixel
    keep = np.ones(px.size, bool)
    np.not_equal(px[1:], px[:-1], out=keep[1:])
    return px[keep]


def validation_set(domain, seed: int = 0) -> tuple[np.ndarray, int]:
    """[rows, 512, 3] float32 + the real sample count, all inside the domain: a 41^3 grid (saturated, out of gamut),
    2^20 random samples, the cube's faces and near-faces (where inverse tone scales are steepest), a neutral ramp,
    and - for any domain - value-uniform samples (bright, arbitrary channel ratios) and log-magnitude samples (tiny
    values of either sign)."""
    kind, lo, hi = domain
    rng = np.random.default_rng(seed)
    plo, phi = (float(_code_from_lin(lo)), float(_code_from_lin(hi))) if kind == "lin" else (lo, hi)
    g = np.linspace(plo, phi, 41, dtype=np.float32)
    cube = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    rnd = rng.uniform(plo, phi, (N_RANDOM, 3)).astype(np.float32)
    f = np.linspace(plo, phi, 129, dtype=np.float32)
    ff = np.stack(np.meshgrid(f, f, indexing="ij"), -1).reshape(-1, 2)
    span = phi - plo
    levels = (plo, plo + 1e-3 * span, phi - 1e-3 * span, phi - 1e-4 * span, phi)
    faces = [np.insert(ff, ch, np.float32(v), axis=1) for ch in range(3) for v in levels]
    ramp = np.repeat(np.linspace(plo, phi, 8192, dtype=np.float32)[:, None], 3, 1)
    p = np.concatenate([cube, rnd, *faces, ramp])
    v = _lin_from_code(p) if kind == "lin" else p
    k = N_RANDOM // 4
    uni = rng.uniform(lo, hi, (k, 3))
    top = np.log10(max(abs(lo), abs(hi), 1e-6))
    small = 10.0 ** rng.uniform(-6.0, top, (k, 3)) * np.where(rng.random((k, 3)) < 0.8, 1.0, -1.0)
    v = np.clip(np.concatenate([v, uni, small]), lo, hi).astype(np.float32)
    n = len(v)
    v = np.concatenate([v, np.zeros(((-n) % 512, 3), np.float32)])
    return v.reshape(-1, 512, 3), n


def compare(cpu: np.ndarray, gpu: np.ndarray, n: int, out_encoding: str, display_out: bool,
            inputs: np.ndarray | None = None) -> dict[str, Any]:
    """The worst case of GPU vs CPU over n samples. Float32 rounding of a colour transform scales with the pixel's
    largest channel, so relative errors are taken against that (a near-zero channel beside a bright one is noise)."""
    a, b = cpu.reshape(-1, 3)[:n], gpu.reshape(-1, 3)[:n]
    fa, fb = np.isfinite(a).all(1), np.isfinite(b).all(1)
    both = np.flatnonzero(fa & fb)
    diff = np.abs(a[both] - b[both]).max(1)
    mag = np.abs(a[both]).max(1)
    inr = mag <= 1.0
    if display_out:                                     # relative only above display white (un-tone-mapped views)
        rel = diff[~inr] / mag[~inr]
    else:
        rel = diff / np.maximum(mag, REL_FLOOR)
    out = {"samples": int(n), "nonfinite_mismatch": int((fa != fb).sum()),
           "max_abs": float(diff.max()) if diff.size else 0.0,
           "max_abs_in_range": float(diff[inr].max()) if inr.any() else 0.0,
           "p999_abs": float(np.percentile(diff, 99.9)) if diff.size else 0.0,
           "max_rel": float(rel.max()) if rel.size else 0.0}
    if display_out:
        ok = out["max_abs_in_range"] <= TOL_DISPLAY_ABS and out["max_rel"] <= TOL_REL
        out["tolerance"] = f"display: max |d| <= {TOL_DISPLAY_ABS:g} in 0-1, relative <= {TOL_REL:g} above 1"
    elif out_encoding == "scene-linear":
        ok = out["max_rel"] <= TOL_REL
        out["tolerance"] = (f"scene-linear: max relative <= {TOL_REL:g} of the pixel's largest channel "
                            f"(absolute {TOL_REL * REL_FLOOR:g} below {REL_FLOOR:g})")
    else:
        ok = out["max_abs"] <= TOL_LOG_ABS
        out["tolerance"] = f"code values: max |d| <= {TOL_LOG_ABS:g}"
    out["ok"] = bool(ok and out["nonfinite_mismatch"] == 0)
    if diff.size:
        i = int(both[int(np.argmax(diff))])
        out["worst"] = {"cpu": a[i].tolist(), "gpu": b[i].tolist()}
        if inputs is not None:
            out["worst"]["in"] = inputs.reshape(-1, 3)[i].tolist()
    return out


# ---- the worker that owns the OpenGL context ----------------------------------------------------------------------

class _Job:
    __slots__ = ("box", "done")

    def __init__(self):
        self.box: dict = {}
        self.done = threading.Event()

    def result(self):
        self.done.wait()
        if "error" in self.box:
            raise self.box["error"]
        return self.box.get("value")


class _Worker:
    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self._t = threading.Thread(target=self._loop, name="c2c_cm_gpu", daemon=True)
        self._t.start()

    def _loop(self):
        while True:
            fn, args, job = self._q.get()
            try:
                job.box["value"] = fn(*args)
            except BaseException as exc:  # noqa: BLE001 - handed back to the caller
                job.box["error"] = exc
            job.done.set()

    def submit(self, fn: Callable, *args) -> _Job:
        job = _Job()
        self._q.put((fn, args, job))
        return job

    def call(self, fn: Callable, *args):
        return self.submit(fn, *args).result()


_WORKER: _Worker | None = None
_LOCK = threading.Lock()
_CTX: dict[str, Any] = {"ctx": None, "reason": None, "checked": False, "info": None}
_PROGRAMS: dict = {}
_GATES: dict = {}               # transform key -> the validation-set gate (once per transform per session)
_DECISIONS: dict = {}           # (key, mode, frame size) -> the decision
_REVOKED: dict = {}             # transform key -> why the GPU failed mid-clip (the CPU from then on)


def _worker() -> _Worker:
    global _WORKER
    with _LOCK:
        if _WORKER is None:
            _WORKER = _Worker()
    return _WORKER


def _open_context():                                    # runs in the worker
    if _CTX["checked"]:
        return _CTX["ctx"]
    _CTX["checked"] = True
    try:
        import moderngl  # type: ignore[import-not-found]
    except ImportError:
        _CTX["reason"] = "moderngl is not installed (pip install moderngl) - exact CPU"
        return None
    errors = []
    for kw in ({}, {"backend": "egl"}):                 # Windows / X11, then headless Linux
        try:
            ctx = moderngl.create_standalone_context(require=400, **kw)
            _CTX["ctx"] = ctx
            _CTX["info"] = f"{ctx.info.get('GL_RENDERER')} (OpenGL {ctx.info.get('GL_VERSION')})"
            return ctx
        except Exception as exc:  # noqa: BLE001 - the next backend, then CPU
            errors.append(str(exc).splitlines()[0][:120] if str(exc) else type(exc).__name__)
    _CTX["reason"] = "no OpenGL 4.0 context (" + "; ".join(errors) + ") - exact CPU"
    return None


def available() -> tuple[bool, str]:
    """(usable, what / why). Never raises."""
    if disabled():
        return False, f"{KILL_SWITCH}=off - exact CPU"
    try:
        ctx = _worker().call(_open_context)
    except Exception as exc:  # noqa: BLE001
        return False, f"the GPU worker failed ({exc}) - exact CPU"
    return (ctx is not None), (_CTX["info"] if ctx is not None else _CTX["reason"])


class _Program:
    """One OCIO processor as a program + its LUT textures (created and used in the worker only)."""

    def __init__(self, ctx, proc):
        import moderngl  # type: ignore[import-not-found]
        import PyOpenColorIO as ocio  # type: ignore[import-not-found]
        self.mgl, self.ctx = moderngl, ctx
        desc = ocio.GpuShaderDesc.CreateShaderDesc(language=ocio.GPU_LANGUAGE_GLSL_4_0, functionName="OCIOMain")
        desc.setAllowTexture1D(False)                    # moderngl has no 1D textures: OCIO writes 2D ones
        proc.getDefaultGPUProcessor().extractGpuShaderInfo(desc)
        if list(desc.getUniforms()):
            raise RuntimeError("the transform has dynamic OCIO uniforms (not supported on the GPU path)")
        self.prog = ctx.program(vertex_shader=VS, fragment_shader=FS % desc.getShaderText())
        self.luts = []
        unit = 1
        for t in desc.getTextures():
            comps = 1 if t.channel == ocio.GpuShaderDesc.TEXTURE_RED_CHANNEL else 3
            tex = ctx.texture((t.width, t.height), comps, np.asarray(t.getValues(), np.float32).tobytes(), dtype="f4")
            f = moderngl.NEAREST if t.interpolation == ocio.INTERP_NEAREST else moderngl.LINEAR
            tex.filter = (f, f)
            tex.repeat_x = tex.repeat_y = False
            self.luts.append((unit, tex))
            if t.samplerName in self.prog:
                self.prog[t.samplerName].value = unit
            unit += 1
        for t in desc.get3DTextures():
            e = t.edgeLen
            tex = ctx.texture3d((e, e, e), 3, np.asarray(t.getValues(), np.float32).tobytes(), dtype="f4")
            f = moderngl.NEAREST if t.interpolation == ocio.INTERP_NEAREST else moderngl.LINEAR
            tex.filter = (f, f)
            self.luts.append((unit, tex))
            if t.samplerName in self.prog:
                self.prog[t.samplerName].value = unit
            unit += 1
        if "c2c_src" in self.prog:
            self.prog["c2c_src"].value = 0
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32)
        self.vao = ctx.vertex_array(self.prog, [(ctx.buffer(quad.tobytes()), "2f", "in_pos")])
        self.max_size = int(ctx.info.get("GL_MAX_TEXTURE_SIZE") or 16384)
        self.size = None
        self.src = self.dst = self.fbo = None

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        """float32 [H, W, 3] (C-contiguous) -> a new float32 [H, W, 3], in strips of at most TILE_PIXELS."""
        h, w = rgb.shape[:2]
        if w > self.max_size:
            raise RuntimeError(f"the frame is {w} px wide; this GPU's texture limit is {self.max_size}")
        rows = max(1, min(h, TILE_PIXELS // max(w, 1), self.max_size))
        if self.size is None or self.size[0] != w or self.size[1] < rows:
            for obj in (self.fbo, self.dst, self.src):
                if obj is not None:
                    obj.release()
            self.src = self.ctx.texture((w, rows), 3, dtype="f4")
            self.src.filter = (self.mgl.NEAREST, self.mgl.NEAREST)
            self.dst = self.ctx.texture((w, rows), 4, dtype="f4")
            self.fbo = self.ctx.framebuffer(color_attachments=[self.dst])
            self.size = (w, rows)
        out = np.empty((h, w, 3), np.float32)
        self.src.use(0)
        for unit, tex in self.luts:
            tex.use(unit)
        self.fbo.use()
        for r0 in range(0, h, rows):
            k = min(rows, h - r0)
            self.src.write(rgb[r0:r0 + k], viewport=(0, 0, w, k))       # no bytes copy: the array's own buffer
            self.ctx.viewport = (0, 0, w, k)
            self.vao.render(self.mgl.TRIANGLE_STRIP)
            self.fbo.read_into(out[r0:r0 + k], viewport=(0, 0, w, k), components=3, dtype="f4")
        return out


def _program(key, proc):                                # runs in the worker
    p = _PROGRAMS.get(key)
    if p is None:
        ctx = _open_context()
        if ctx is None:
            raise RuntimeError(_CTX["reason"] or "no OpenGL context")
        p = _PROGRAMS[key] = _Program(ctx, proc)
    return p


def _gpu_apply(key, proc, rgb):                         # runs in the worker
    return _program(key, proc).apply(rgb)


def _release_programs():                                # runs in the worker
    for p in _PROGRAMS.values():
        for obj in (p.fbo, p.dst, p.src, p.vao, p.prog, *(t for _, t in p.luts)):
            try:
                if obj is not None:
                    obj.release()
            except Exception:  # noqa: BLE001 - best effort
                pass
    _PROGRAMS.clear()


# ---- the decision, once per transform (and frame size) ------------------------------------------------------------

def _median_ms(fn, n: int) -> float:
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(1000.0 * (time.perf_counter() - t0))
    return float(np.median(ts))


def decide(key: Any, proc, cpu_run: Callable[[np.ndarray], None], *, mode: str, domain, out_encoding: str,
           display_out: bool, sample: np.ndarray | None = None) -> dict[str, Any]:
    """{use, method, gate, frame_gate, speed, domain} for this transform on this machine (cached per frame size).
    OCIO GPU: the validation gate + the first frame's gate. Auto: those, then the GPU must be measured faster end to
    end (upload, shader, readback, domain check, CPU pixels) on the first frame - no frame, no GPU."""
    mode = mode if mode in MODES else "Auto"
    shape = tuple(int(s) for s in sample.shape[:2]) if sample is not None and sample.size else None
    ck = (key, mode, shape)
    if ck in _DECISIONS:
        return _DECISIONS[ck]
    res: dict[str, Any] = {"use": False, "mode": mode, "gate": None, "frame_gate": None, "speed": None,
                           "domain": list(domain)}

    def done(method: str) -> dict[str, Any]:
        res["method"] = method
        _DECISIONS[ck] = res
        return res

    if mode == "Exact CPU":
        return done("exact CPU (processing = Exact CPU)")
    if key in _REVOKED:
        return done(f"exact CPU: {_REVOKED[key]}")
    ok, info = available()
    if not ok:
        return done(f"exact CPU: {info}")
    if mode == "Auto" and shape is None:
        return done("exact CPU: Auto needs a frame to measure the speed on")
    try:
        gate = _GATES.get(key)
        if gate is None:
            vals, n = validation_set(domain)
            ref = vals.copy()
            cpu_run(ref)
            got = _worker().call(_gpu_apply, key, proc, vals)
            gate = _GATES[key] = compare(ref, got, n, out_encoding, display_out, vals)
        res["gate"] = gate
        if not gate["ok"]:
            return done(f"exact CPU: the GPU result is outside the tolerance on this machine (max "
                        f"{gate['max_abs']:.2g} over {gate['samples']} samples; {gate['tolerance']})")
        sp = ""
        if shape is not None:
            frame = np.ascontiguousarray(sample[..., :3], np.float32)
            keep = np.ones(frame.shape[0] * frame.shape[1], bool)
            keep[out_of_domain(frame, domain)] = False
            ref = frame.copy()
            cpu_run(ref)
            got = apply_frame(key, proc, frame, cpu_run, domain)      # also the warm-up at this size
            fg = compare(ref.reshape(-1, 3)[keep], got.reshape(-1, 3)[keep], int(keep.sum()), out_encoding,
                         display_out)
            fg["routed_to_cpu"] = int((~keep).sum())
            res["frame_gate"] = fg
            if not fg["ok"]:
                return done(f"exact CPU: on this clip's first frame the GPU result is outside the tolerance (max "
                            f"{fg['max_abs']:.2g}; {fg['tolerance']})")
            if mode == "Auto":
                g = _median_ms(lambda: apply_frame(key, proc, frame, cpu_run, domain), 5)   # laptop GPUs spike
                ts = []
                for _ in range(3):                      # the copy back to the input is not timed
                    np.copyto(ref, frame)
                    t0 = time.perf_counter()
                    cpu_run(ref)
                    ts.append(1000.0 * (time.perf_counter() - t0))
                c = float(np.median(ts))
                res["speed"] = {"gpu_ms": round(g, 1), "cpu_ms": round(c, 1), "frame": [shape[1], shape[0]],
                                "end_to_end": "upload + shader + readback + domain check + CPU pixels"}
                if g * SPEEDUP_NEEDED > c:
                    return done(f"exact CPU: the GPU was not faster here ({res['speed']['gpu_ms']} vs "
                                f"{res['speed']['cpu_ms']} ms per {shape[1]}x{shape[0]} frame)")
                sp = f", {res['speed']['gpu_ms']} vs {res['speed']['cpu_ms']} ms per frame"
    except Exception as exc:  # noqa: BLE001 - compile / driver errors: CPU
        return done(f"exact CPU: the GPU program failed ({(str(exc) or type(exc).__name__).splitlines()[0][:160]})")
    res["use"] = True
    return done(f"OCIO GPU on {_CTX['info']} (validated against the CPU: max {gate['max_abs']:.2g} over "
                f"{gate['samples']} samples + the first frame, {gate['tolerance']}{sp}; pixels outside "
                f"{domain[1]:g}..{domain[2]:g} on the CPU)")


def apply_frame(key: Any, proc, rgb: np.ndarray, cpu_run: Callable[[np.ndarray], None], domain) -> np.ndarray:
    """One frame through the GPU (the domain check overlaps it); pixels outside the domain recomputed on the CPU."""
    rgb = np.ascontiguousarray(rgb, np.float32)
    job = _worker().submit(_gpu_apply, key, proc, rgb)
    idx = out_of_domain(rgb, domain)
    sub = None
    if idx.size:
        sub = np.ascontiguousarray(rgb.reshape(-1, 3)[idx])
        cpu_run(sub)
    out = job.result()
    if sub is not None:
        out.reshape(-1, 3)[idx] = sub
    return out


def apply_chunk(key: Any, proc, chunk: np.ndarray, cpu_run: Callable[[np.ndarray], None], domain) -> None:
    """The RGB of a float32 [N, H, W, C] chunk in place (alpha untouched). Frame i+1's GPU pass overlaps frame i's CPU
    work; at most two frames in flight. If the GPU fails mid-chunk, that frame and the rest go to the CPU and the
    transform stays on the CPU for the session."""
    n = chunk.shape[0]
    w = _worker()

    def rgb_of(i):                                      # (C-contiguous RGB, is it a view into the chunk)
        x = chunk[i, ..., :3]
        return (x, True) if x.flags.c_contiguous else (np.ascontiguousarray(x), False)

    pending = None
    for i in range(n + 1):
        nxt = None
        if i < n:
            x = rgb_of(i)[0]
            nxt = (i, x, w.submit(_gpu_apply, key, proc, x))
        if pending is not None:
            j, x, job = pending
            idx = out_of_domain(x, domain)
            sub = None
            if idx.size:
                sub = np.ascontiguousarray(x.reshape(-1, 3)[idx])
                cpu_run(sub)
            try:
                out = job.result()
            except Exception as exc:  # noqa: BLE001 - the CPU from here on
                if nxt is not None:
                    try:
                        nxt[2].result()                 # it reads the chunk: let it finish first
                    except Exception:  # noqa: BLE001
                        pass
                why = f"the GPU failed mid-clip ({(str(exc) or type(exc).__name__).splitlines()[0][:160]})"
                _REVOKED[key] = why
                for d in list(_DECISIONS):
                    if d[0] == key:
                        del _DECISIONS[d]
                log.warning("[c2c_video] colour: %s - exact CPU for the rest", why)
                for r in range(j, n):
                    y, view = rgb_of(r)
                    cpu_run(y)
                    if not view:
                        chunk[r, ..., :3] = y
                return
            if sub is not None:
                out.reshape(-1, 3)[idx] = sub
            chunk[j, ..., :3] = out
        pending = nxt


def reset() -> None:
    """Forget the gates and decisions (a setting or the config changed) and free the compiled programs."""
    _DECISIONS.clear()
    _GATES.clear()
    _REVOKED.clear()
    if _WORKER is not None and _CTX["ctx"] is not None:
        try:
            _WORKER.call(_release_programs)
        except Exception:  # noqa: BLE001 - best effort
            pass
