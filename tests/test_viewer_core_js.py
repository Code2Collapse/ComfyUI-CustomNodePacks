"""L7.114: the Matte viewer's pure JavaScript (js/c2c_viewer/core.js, transport.js) run under Node.

core: hit maps, the overlap rules (parts inside the subject; an add may not swallow another subject), the picks
JSON the node reads (same schema as nodes/mask_matting/_picks.py), the stepper, QC levels and tooltips.
transport: helper prompts cut from an API prompt, execution ids inside subgraphs, and the runner against a fake
ComfyUI api (websocket events before / after the POST, the /history backstop, the waiting state, refusals).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
CORE = PACK / "js" / "c2c_viewer" / "core.js"
TRANSPORT = PACK / "js" / "c2c_viewer" / "transport.js"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not on PATH")

# a 10 x 10 plate proxy: a "person" box (cols 1-5, rows 1-8), a "bag" inside it (cols 2-3, rows 5-6) and a second
# person (cols 7-9, rows 2-8)
FIXTURE = """
const W = 10, H = 10;
const rect = (x0, y0, x1, y1) => { const m = new Uint8Array(W * H); for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) m[y * W + x] = 1; return m; };
const obj = (id, label, m) => ({ id, label, w: W, h: H, mask: m, count: C.countOf(m), box: [0, 0, 1, 1] });
const person = obj(0, "person", rect(1, 1, 5, 8));
const bag = obj(1, "bag", rect(2, 5, 3, 6));
const other = obj(2, "person", rect(7, 2, 9, 8));
const objects = [person, bag, other];
const at = (x, y) => [(x + 0.5) / W, (y + 0.5) / H];
"""


def _js(body, module=CORE, name="C"):
    script = (f"import * as {name} from {json.dumps(module.as_uri())};\n" + FIXTURE
              + f"\nconst __out = await (async () => {{ {body} }})();\nconsole.log(JSON.stringify(__out));")
    p = subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True, text=True,
                       encoding="utf-8", timeout=60)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.strip().splitlines()[-1])


def _tjs(body):
    script = (f"import * as C from {json.dumps(CORE.as_uri())};\n"
              f"import * as T from {json.dumps(TRANSPORT.as_uri())};\n" + FIXTURE + FAKE_API
              + f"\nconst __out = await (async () => {{ {body} }})();\nconsole.log(JSON.stringify(__out));")
    p = subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True, text=True,
                       encoding="utf-8", timeout=60)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_the_modules_import_nothing_from_comfyui():
    for f in (CORE, TRANSPORT):
        src = f.read_text(encoding="utf-8")
        assert "scripts/" not in src and "document." not in src and "window." not in src, f


# ── hit maps ─────────────────────────────────────────────────────────────────────────────────────────────────────

def test_mask_from_rgba_reads_the_alpha_channel_or_the_brightness():
    got = _js("""
        const rgba = new Uint8ClampedArray(4 * 4);
        rgba.set([255,255,255,0, 255,255,255,200, 255,255,255,40, 255,255,255,255]);
        const grey = new Uint8ClampedArray(4 * 2);
        grey.set([10,10,10,255, 240,240,240,255]);
        return [Array.from(C.maskFromRGBA(rgba, 4, 1)), Array.from(C.maskFromRGBA(grey, 2, 1))];""")
    assert got == [[0, 1, 0, 1], [0, 1]]


def test_hit_test_picks_the_smallest_object_under_the_point():
    got = _js("""return [C.hitTest(objects, ...at(2, 5))?.label, C.hitTest(objects, ...at(4, 2))?.label,
                         C.hitTest(objects, ...at(8, 4))?.id, C.hitTest(objects, ...at(6, 0))];""")
    assert got == ["bag", "person", 2, None]


def test_containment():
    assert _js("return [C.containment(person.mask, bag.mask), C.containment(bag.mask, person.mask), "
               "C.containment(person.mask, other.mask), C.containment(person.mask, new Uint8Array(3))];") \
        == [1, 4 / 40, 0, 0]


def test_the_interior_point_is_deep_inside():
    p = _js("return C.interiorPoint(person.mask, W, H);")
    x, y = p["x"] * 10 - 0.5, p["y"] * 10 - 0.5
    assert 2 <= x <= 4 and 3 <= y <= 6
    assert _js("return C.interiorPoint(new Uint8Array(100), W, H);") is None


# ── the rules ────────────────────────────────────────────────────────────────────────────────────────────────────

def test_click_makes_the_subject_parts_inside_it_and_everything_else_excluded():
    got = _js("""const r = C.clickSubject(objects, ...at(4, 2), 7);
                 return [r.sel.frame, r.sel.subject.id, r.sel.parts, r.sel.excluded.map((e) => e.id), r.message];""")
    frame, subject, parts, excluded, message = got
    assert (frame, subject, parts, excluded) == (7, 0, [1], [2])     # the bag is inside the person: a part
    assert message.startswith("Subject: person #1.")


def test_a_click_on_nothing_is_the_point_itself():
    got = _js("const r = C.clickSubject(objects, 0.05, 0.05, 0); return [r.sel.subject, r.sel.excluded.length, r.message];")
    assert got[0] == {"x": 0.05, "y": 0.05, "id": None} and got[1] == 3 and "the point itself" in got[2]


def test_an_add_that_swallows_another_subject_is_refused_and_says_why():
    got = _js("""
        const s = C.clickSubject(objects, ...at(4, 2), 0).sel;
        const onOther = C.addPart(s, objects, ...at(8, 4));
        const region = new Uint8Array(100); for (let i = 0; i < 100; i++) region[i] = (i % 10) >= 5 ? 1 : 0;
        const swallow = C.addPart(s, objects, ...at(6, 1), region);       // the part's region covers the other person
        const small = new Uint8Array(100); small[0 * 10 + 6] = 1;
        const fine = C.addPart(s, objects, ...at(6, 0), small);
        const noSubject = C.addPart(C.emptySelection(0), objects, 0.5, 0.5);
        return [onOther.ok, onOther.reason, swallow.ok, swallow.reason, fine.ok, fine.sel.add.length, s.add.length,
                noSubject.reason];""")
    assert got[0] is False and "person #3, another subject" in got[1] and "without Shift" in got[1]
    assert got[2] is False and "100% of person #3" in got[3]
    assert got[4] is True and got[5] == 1 and got[6] == 0              # the original selection is untouched
    assert "Click the subject first" in got[7]


def test_ctrl_click_removes_the_nearest_marker():
    got = _js("""
        let s = C.clickSubject(objects, ...at(4, 2), 0).sel;
        s = C.addPart(s, objects, 0.6, 0.05).sel;
        s = C.cutPart(s, 0.3, 0.3).sel;
        const a = C.removeNearest(s, 0.61, 0.06, 1);
        const far = C.removeNearest(a.sel, 0.0, 0.99, 1);
        const subj = C.removeNearest(a.sel, ...at(4, 2), 1);
        return [a.removed, a.sel.add.length, a.sel.cut.length, far.removed, subj.removed, subj.sel.subject, subj.sel.cut.length];""")
    assert got == ["add", 0, 1, None, "subject", None, 0]


def test_the_selection_preview_adds_and_cuts_regions():
    got = _js("""
        let s = C.clickSubject(objects, ...at(4, 2), 0).sel;
        const addR = new Uint8Array(100); addR[0] = 1;
        const cutR = new Uint8Array(100); cutR[1 * 10 + 1] = 1;
        s = C.addPart(s, objects, 0.05, 0.05, addR).sel;
        s = C.cutPart(s, 0.15, 0.15, cutR).sel;
        const m = C.selectionMask(objects, s, W, H);
        return [C.countOf(m), m[0], m[11], m[12], m[8 * 10 + 5], m[4 * 10 + 8]];""")
    assert got == [40 + 1 - 1, 1, 0, 1, 1, 0]


def test_picks_json_round_trip_matches_the_python_schema():
    got = _js("""
        let s = C.clickSubject(objects, ...at(4, 2), 3).sel;
        s = C.addPart(s, objects, 0.6, 0.05).sel;
        s = C.cutPart(s, 0.3, 0.3).sel;
        const text = C.toPicks(s, { text: "person", model: "sam3.1/x.safetensors", plate: { frames: 20, w: 10, h: 10 } });
        const back = C.fromPicks(text);
        return [JSON.parse(text), back.sel.subject.x, back.text, C.toPicks(C.emptySelection(0)), C.fromPicks("{bad"),
                C.fromPicks(JSON.stringify({ v: 2, subject: { x: 0.5, y: 0.5 } }))];""")
    p = got[0]
    assert p["v"] == 1 and p["frame"] == 3 and p["text"] == "person" and p["model"] == "sam3.1/x.safetensors"
    assert set(p["subject"]) == {"x", "y", "box", "label"} and p["subject"]["label"] == "person"
    assert [set(e) for e in p["excluded"]] == [{"x", "y", "box", "label"}]
    assert p["add"] == [{"x": 0.6, "y": 0.05}] and p["cut"] == [{"x": 0.3, "y": 0.3}]
    assert p["plate"] == {"frames": 20, "w": 10, "h": 10} and "range" not in p
    assert got[1] == p["subject"]["x"] and got[2] == "person"
    assert got[3] == "" and got[4] is None and got[5] is None
    # the Python side reads exactly this
    from nodes.mask_matting import _picks
    sel = _picks.parse(json.dumps(p))
    assert sel.frame == 3 and len(sel.excluded) == 1 and len(sel.add) == 1 and len(sel.cut) == 1


# ── stepper, ranges, frames ──────────────────────────────────────────────────────────────────────────────────────

def test_the_stepper_shows_one_next_action():
    got = _js("""return [
        C.stepper({ plate: false }), C.stepper({ plate: true, subject: false }),
        C.stepper({ plate: true, subject: true }), C.stepper({ plate: true, subject: true, preview: true }),
        C.stepper({ plate: true, subject: true, busy: true }), C.stepper({ plate: true, pickStep: false })];""")
    assert [g["action"]["id"] for g in got] == ["load", "find", "preview", "full", "wait", "preview"]
    assert [s["state"] for s in got[1]["steps"]] == ["done", "current", "todo"]
    assert [s["id"] for s in got[5]["steps"]] == ["plate", "render"]     # no pick step: absent, not disabled
    assert got[2]["action"]["label"] == "Preview 17 frames"


def test_the_preview_range_starts_at_the_pick_and_fits_the_plate():
    assert _js("return [C.previewRange(10, 100), C.previewRange(95, 100), C.previewRange(3, 8), C.previewRange(0, 100, 49)];") \
        == [{"start": 10, "count": 17}, {"start": 83, "count": 17}, {"start": 0, "count": 8}, {"start": 0, "count": 49}]


def test_frames_map_to_the_nearest_proxy_and_clicks_to_the_plate():
    got = _js("""
        const fr = [0, 5, 10, 15, 19].map((i) => ({ index: i }));
        return [C.nearestFrame(fr, 7).index, C.nearestFrame(fr, 8).index, C.nearestFrame(fr, 19).index, C.frameLabel(12, 120),
                C.viewToPlate({ left: 0, top: 0, width: 200, height: 200 }, 1920, 1080, 100, 100),
                C.viewToPlate({ left: 0, top: 0, width: 200, height: 200 }, 1920, 1080, 100, 10).inside];""")
    assert got[:4] == [5, 10, 19, "frame 12 (13 / 120)"]
    assert got[4] == {"x": 0.5, "y": 0.5, "inside": True} and got[5] is False


def test_qc_levels_and_tooltip_use_the_server_thresholds():
    from nodes.c2c_viewer import qc
    t = json.dumps(qc.THRESHOLDS)
    got = _js(f"""const T = {t};
        return [C.qcLevel(0.99, T.coverage), C.qcLevel(0.93, T.coverage), C.qcLevel(0.5, T.coverage),
                C.qcLevel(0.01, T.flicker), C.qcLevel(0.03, T.flicker), C.qcLevel(null, T.flicker),
                C.qcTooltip("flicker", {{ value: 0.031, worst: {{ value: 0.05, frame: 9 }} }}, T.flicker, 640)];""")
    assert got[:6] == ["green", "amber", "red", "green", "amber", "n/a"]
    assert "Green ≤ 0.025, amber ≤ 0.040" in got[6] and "frame 9" in got[6] and "synthetic clips" in got[6]
    for value, name in ((0.99, "coverage"), (0.93, "coverage"), (0.03, "flicker"), (0.2, "edge_detail")):
        assert _js(f"return C.qcLevel({value}, {t}.{name});") == qc.level(name, value)


# ── transport ────────────────────────────────────────────────────────────────────────────────────────────────────

PROMPT = {
    "3": {"class_type": "LoadVideo", "inputs": {"file": "a.mp4"}},
    "4": {"class_type": "Resize", "inputs": {"image": ["3", 0], "w": 512}},
    "9": {"class_type": "SAMModelLoader", "inputs": {"name": "x"}},
    "7:12": {"class_type": "MaskOpsMEC", "inputs": {"image": ["4", 0], "sam_model": ["9", 0], "picks": ""}},
    "20": {"class_type": "PreviewImage", "inputs": {"images": ["7:12", 2]}},
}


def test_helper_prompts_take_only_what_the_step_needs():
    got = _tjs(f"""const P = {json.dumps(PROMPT)};
        const plate = T.helperPrompt(P, "7:12", "plate");
        const det = T.helperPrompt(P, "7:12", "detect", {{ frame: 4, model: "m", text: "person", points: [{{ x: 0.5, y: 0.5, label: 1 }}] }});
        const ren = T.helperPrompt(P, "7:12", "render", {{ picks: "{{\\"v\\":1}}", start: 4 }});
        return [Object.keys(plate.prompt).sort(), plate.prompt[plate.helperId], det.prompt[det.helperId].inputs,
                Object.keys(ren.prompt).sort(), ren.prompt["7:12"].inputs.picks, ren.prompt[ren.helperId].inputs, P["7:12"].inputs.picks];""")
    assert got[0] == ["3", "4", "c2c_viewer_helper"]                     # the plate chain only: no SAM loader
    assert got[1] == {"class_type": "C2CViewerPlateProxy", "inputs": {"image": ["4", 0], "max_side": 960, "max_frames": 240}}
    assert got[2]["frame"] == 4 and got[2]["text"] == "person" and json.loads(got[2]["points"]) == [{"x": 0.5, "y": 0.5, "label": 1}]
    assert got[3] == ["3", "4", "7:12", "9", "c2c_viewer_helper"]        # everything the Matte node reads, no preview
    assert got[4] == '{"v":1}' and got[6] == ""                          # the copy gets the picks, the graph is untouched
    assert got[5] == {"image": ["4", 0], "alpha": ["7:12", 1], "coarse": ["7:12", 0], "start": 4, "max_side": 960,
                      "max_frames": 120}


def test_helper_prompts_refuse_in_words():
    got = _tjs(f"""const P = {json.dumps(PROMPT)};
        const msg = (f) => {{ try {{ f(); return null; }} catch (e) {{ return e.message; }} }};
        return [msg(() => T.helperPrompt(P, "99", "plate")),
                msg(() => T.helperPrompt({{ "1": {{ class_type: "MaskOpsMEC", inputs: {{}} }} }}, "1", "plate"))];""")
    assert "muted or bypassed" in got[0] and "Connect a plate" in got[1]


def test_execution_ids_inside_subgraphs():
    got = _tjs("""
        const root = { _nodes: [] };
        const inner2 = { _nodes: [] };
        const inner1 = { _nodes: [{ id: 3, subgraph: inner2 }] };
        root._nodes.push({ id: 7, subgraph: inner1 }, { id: 1 });
        const top = { id: 1, graph: root }, mid = { id: 12, graph: inner1 }, deep = { id: 5, graph: inner2 };
        const P = { "7:12": { class_type: "MaskOpsMEC" }, "8": { class_type: "Other" } };
        return [T.executionIdOf(root, top), T.executionIdOf(root, mid), T.executionIdOf(root, deep),
                T.findNodeId(P, "MaskOpsMEC", null, 12), T.findNodeId(P, "MaskOpsMEC", "7:12", 12)];""")
    assert got == ["1", "7:12", "7:3:5", "7:12", "7:12"]


FAKE_API = """
class FakeApi {
  constructor(script) { this.listeners = {}; this.calls = []; this.script = script; this.clientId = "cid"; }
  addEventListener(t, f) { (this.listeners[t] ||= []).push(f); }
  removeEventListener(t, f) { this.listeners[t] = (this.listeners[t] || []).filter((g) => g !== f); }
  emit(type, detail) { for (const f of [...(this.listeners[type] || [])]) f({ type, detail }); }
  listenerCount() { return Object.values(this.listeners).reduce((a, l) => a + l.length, 0); }
  async fetchApi(route, init) {
    this.calls.push([route, init?.body ? JSON.parse(init.body) : null]);
    const r = await this.script(route, init, this);
    return { ok: r.status === undefined ? true : r.status < 400, status: r.status ?? 200, json: async () => r.body };
  }
}
const payload = { kind: "plate", count: 3 };
const executed = (pid) => ({ prompt_id: pid, node: "c2c_viewer_helper", output: { c2c_viewer: [payload] } });
const fast = { pollMs: 5, vanishPolls: 3 };
"""


def test_runner_resolves_from_the_websocket_and_posts_to_the_front():
    got = _tjs("""
        const api = new FakeApi(async (route) => {
            if (route === "/prompt") { setTimeout(() => api.emit("executed", executed("p1")), 1); return { body: { prompt_id: "p1" } }; }
            return { body: {} };
        });
        const r = T.createRunner(api, fast);
        const out = await r.run({ a: 1 }, "c2c_viewer_helper");
        const post = api.calls[0];
        return [out, post[0], post[1].front, post[1].client_id, post[1].prompt, api.listenerCount(), r.busy];""")
    assert got == [{"kind": "plate", "count": 3}, "/prompt", True, "cid", {"a": 1}, 0, False]


def test_a_cached_helper_that_finishes_before_the_post_returns_is_not_lost():
    got = _tjs("""
        const api = new FakeApi(async (route, init, a) => {
            if (route === "/prompt") { a.emit("executed", executed("p2")); return { body: { prompt_id: "p2" } }; }
            return { body: {} };
        });
        return await T.createRunner(api, fast).run({}, "c2c_viewer_helper");""")
    assert got == {"kind": "plate", "count": 3}


def test_the_history_backstop_and_the_waiting_state():
    got = _tjs("""
        let polls = 0; const states = [];
        const api = new FakeApi(async (route) => {
            if (route === "/prompt") return { body: { prompt_id: "p3" } };
            if (route.startsWith("/history")) { polls += 1;
                return { body: polls < 3 ? {} : { p3: { outputs: { c2c_viewer_helper: { c2c_viewer: [payload] } }, status: {} } } }; }
            if (route === "/queue") return { body: { queue_running: [[1, "other"]], queue_pending: [[2, "p3"]] } };
            return { body: {} };
        });
        const out = await T.createRunner(api, fast).run({}, "c2c_viewer_helper", (s, t) => states.push([s, t]));
        return [out, states];""")
    assert got[0] == {"kind": "plate", "count": 3}
    assert ["waiting", "Waiting for ComfyUI: another job is running."] in got[1]


def test_errors_reach_the_user_as_sentences():
    got = _tjs("""
        const run = async (script) => { try { await T.createRunner(new FakeApi(script), fast).run({}, "c2c_viewer_helper"); return null; }
                                       catch (e) { return e.message; } };
        const refused = await run(async (route) => route === "/prompt"
            ? { status: 400, body: { error: { message: "Prompt outputs failed validation" },
                node_errors: { c2c_viewer_helper: { class_type: "C2CViewerDetect", errors: [{ message: "Value not in list", details: "model" }] } } } }
            : { body: {} });
        const failed = await run(async (route, init, a) => {
            if (route === "/prompt") { setTimeout(() => a.emit("execution_error", { prompt_id: "p4", node_type: "C2CViewerDetect", exception_message: "No SAM 3 / 3.1 checkpoint is installed" }), 1);
                                      return { body: { prompt_id: "p4" } }; }
            return { body: {} }; });
        const vanished = await run(async (route) => route === "/prompt" ? { body: { prompt_id: "p5" } }
            : { body: route === "/queue" ? { queue_running: [], queue_pending: [] } : {} });
        return [refused, failed, vanished];""")
    assert got[0] == "ComfyUI refused the viewer's job. Prompt outputs failed validation; C2CViewerDetect: Value not in list (model)"
    assert got[1] == "C2CViewerDetect failed: No SAM 3 / 3.1 checkpoint is installed"
    assert "disappeared" in got[2] and "restarted" in got[2]


def test_cancel_removes_the_job_and_stops_waiting():
    got = _tjs("""
        const api = new FakeApi(async (route) => route === "/prompt" ? { body: { prompt_id: "p6" } }
            : { body: route === "/queue" ? { queue_running: [[1, "x"]], queue_pending: [[2, "p6"]] } : {} });
        const r = T.createRunner(api, fast);
        const p = r.run({}, "c2c_viewer_helper").catch((e) => e.message);
        await new Promise((ok) => setTimeout(ok, 20));
        await r.cancel();
        return [await p, api.calls.filter((c) => c[0] === "/queue" && c[1]).map((c) => c[1]), r.busy];""")
    assert got[0] == "Stopped." and got[1] == [{"delete": ["p6"]}] and got[2] is False
