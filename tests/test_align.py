"""Tests for C2C Align: c2c_align_core.js geometry + c2c_align.js static guards."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
CORE = PACK / "js" / "c2c_align_core.js"
EXT = PACK / "js" / "c2c_align.js"


def _run_probe(tmp_path, name, uri: str, body: str):
    probe = tmp_path / f"{name}.mjs"
    probe.write_text(f'const M = await import("{uri}");\n' + body, encoding="utf-8")
    node = shutil.which("node")
    p = subprocess.run([node, str(probe)], capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr[:800]
    import json
    return json.loads(p.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_smart_align_row_vs_column(tmp_path):
    uri = CORE.resolve().as_uri()
    data = _run_probe(tmp_path, "axis", uri, """
const row = M.smartAlign([
  {id:"a",x:0,y:0,w:100,h:50},{id:"b",x:200,y:10,w:100,h:50}
], {minGap:10});
const col = M.smartAlign([
  {id:"a",x:0,y:0,w:100,h:50},{id:"b",x:10,y:200,w:100,h:50}
], {minGap:10});
process.stdout.write(JSON.stringify({row:row.axis,col:col.axis}));
""")
    assert data == {"row": "row", "col": "column"}


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_smart_align_idempotent(tmp_path):
    uri = CORE.resolve().as_uri()
    data = _run_probe(tmp_path, "idem", uri, """
const u = [
  {id:"a",x:0,y:0,w:80,h:40},{id:"b",x:120,y:5,w:80,h:40},{id:"c",x:260,y:0,w:80,h:40}
];
const r1 = M.smartAlign(u, {minGap:20});
const m1 = new Map(r1.moves.map(m=>[m.id,m]));
const u2 = u.map(x=>({...x, x:x.x+(m1.get(x.id)?.dx||0), y:x.y+(m1.get(x.id)?.dy||0)}));
const r2 = M.smartAlign(u2, {minGap:20});
process.stdout.write(JSON.stringify(r2.moves));
""")
    assert all(m["dx"] == 0 and m["dy"] == 0 for m in data)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_smart_align_min_gap_separates_overlap(tmp_path):
    uri = CORE.resolve().as_uri()
    data = _run_probe(tmp_path, "gap", uri, """
const u = [{id:"a",x:0,y:0,w:100,h:50},{id:"b",x:50,y:0,w:100,h:50}];
const r = M.smartAlign(u, {minGap:30});
const a = {...u[0], x:u[0].x+r.moves[0].dx, y:u[0].y+r.moves[0].dy};
const b = {...u[1], x:u[1].x+r.moves[1].dx, y:u[1].y+r.moves[1].dy};
const gap = b.x - (a.x + a.w);
process.stdout.write(JSON.stringify({gap, axis:r.axis}));
""")
    assert data["axis"] == "row"
    assert data["gap"] >= 30


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_smart_align_preserves_order(tmp_path):
    uri = CORE.resolve().as_uri()
    data = _run_probe(tmp_path, "order", uri, """
const u = [
  {id:"a",x:0,y:0,w:50,h:50},{id:"b",x:80,y:0,w:50,h:50},{id:"c",x:160,y:0,w:50,h:50}
];
const r = M.smartAlign(u, {minGap:10});
const placed = u.map((x,i)=>({id:x.id, x:x.x+r.moves[i].dx}));
placed.sort((a,b)=>a.x-b.x);
process.stdout.write(JSON.stringify(placed.map(p=>p.id)));
""")
    assert data == ["a", "b", "c"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_align_modes(tmp_path):
    uri = CORE.resolve().as_uri()
    data = _run_probe(tmp_path, "align", uri, """
const u = [
  {id:"a",x:10,y:20,w:100,h:50},
  {id:"b",x:200,y:80,w:80,h:40},
];
const modes = ["left","right","top","bottom","centerX","centerY"];
const out = {};
for (const m of modes) {
  const r = M.alignUnits(u, m);
  out[m] = r.moves;
}
process.stdout.write(JSON.stringify(out));
""")
    assert data["left"][0]["dx"] == 0 and data["left"][1]["dx"] == -190
    assert data["right"][1]["dx"] == 0 and data["right"][0]["dx"] == 170
    assert data["top"][0]["dy"] == 0 and data["top"][1]["dy"] == -60
    assert data["bottom"][1]["dy"] == 0 and data["bottom"][0]["dy"] == 50


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_distribute_fits_and_overflow(tmp_path):
    uri = CORE.resolve().as_uri()
    data = _run_probe(tmp_path, "dist", uri, """
const u = [
  {id:"a",x:0,y:0,w:50,h:50},
  {id:"b",x:100,y:0,w:50,h:50},
  {id:"c",x:400,y:0,w:50,h:50},
];
const fit = M.distribute(u, "x", {minGap:10});
const u2 = [{id:"a",x:0,y:0,w:50,h:50},{id:"b",x:55,y:0,w:50,h:50},{id:"c",x:120,y:0,w:50,h:50}];
const tight = M.distribute(u2, "x", {minGap:40});
const apply = (units, moves) => units.map((x,i)=>{
  const m = moves.find(m=>m.id===x.id);
  return {...x, x:x.x+(m?.dx||0)};
});
const placed = apply(u, fit.moves);
const gap = placed[1].x - (placed[0].x+placed[0].w);
const placedT = apply(u2, tight.moves);
const gapT = placedT[1].x - (placedT[0].x+placedT[0].w);
process.stdout.write(JSON.stringify({gap, gapT, outer0:fit.moves[0].dx, outer2:fit.moves[2].dx}));
""")
    assert data["outer0"] == 0 and data["outer2"] == 0
    assert data["gap"] >= 10
    assert data["gapT"] >= 40


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_equal_size(tmp_path):
    uri = CORE.resolve().as_uri()
    data = _run_probe(tmp_path, "eq", uri, """
const u = [{id:"a",x:0,y:0,w:50,h:30},{id:"b",x:0,y:0,w:120,h:80}];
const w = M.equalSize(u, "width");
const h = M.equalSize(u, "height");
process.stdout.write(JSON.stringify({w,h}));
""")
    assert data["w"] == [{"id": "a", "w": 120, "h": 30}, {"id": "b", "w": 120, "h": 80}]
    assert data["h"] == [{"id": "a", "w": 50, "h": 80}, {"id": "b", "w": 120, "h": 80}]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_snap_edge_and_centre(tmp_path):
    uri = CORE.resolve().as_uri()
    data = _run_probe(tmp_path, "snap", uri, """
const moving = {x:102,y:50,w:80,h:40};
const targets = [{id:"t",x:0,y:0,w:100,h:100}];
const edge = M.snapMove(moving, targets, {threshold:5});
const moving2 = {x:12,y:50,w:80,h:40};
const centre = M.snapMove(moving2, targets, {threshold:5});
process.stdout.write(JSON.stringify({edge, centre}));
""")
    assert data["edge"]["dx"] == -2
    assert abs(data["centre"]["dx"] + 2) < 0.01


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_snap_spacing_and_threshold(tmp_path):
    uri = CORE.resolve().as_uri()
    data = _run_probe(tmp_path, "snap2", uri, """
const targets = [
  {id:"l",x:0,y:0,w:100,h:80},
  {id:"r",x:260,y:0,w:100,h:80},
];
// equal gaps put left at 155; at x=150 that is 5 away and nearer than any edge
const eq = M.snapMove({x:150,y:10,w:50,h:40}, targets, {threshold:40});
// at x=120 the left edge is 20 from l's right edge, nearer than equal spacing (35)
const edge = M.snapMove({x:120,y:10,w:50,h:40}, targets, {threshold:40});
const far = M.snapMove({x:200,y:10,w:50,h:40}, targets, {threshold:2});
process.stdout.write(JSON.stringify({eq, edge, far}));
""")
    assert any(g["kind"] == "spacing" for g in data["eq"]["guides"])
    assert abs(data["eq"]["dx"] - 5) < 0.01
    assert abs(data["edge"]["dx"] - (-20)) < 0.01, "the closest candidate must win"
    assert data["far"]["dx"] == 0 and data["far"]["dy"] == 0


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_snap_closest_wins_per_axis(tmp_path):
    uri = CORE.resolve().as_uri()
    data = _run_probe(tmp_path, "snap3", uri, """
const moving = {x:103,y:203,w:50,h:50};
const targets = [{id:"t",x:0,y:0,w:100,h:100}];
const r = M.snapMove(moving, targets, {threshold:10});
process.stdout.write(JSON.stringify(r));
""")
    assert abs(data["dx"]) <= 10 and abs(data["dy"]) <= 10
    assert data["dx"] != 0 or data["dy"] != 0


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_node_rect_title_offset(tmp_path):
    uri = CORE.resolve().as_uri()
    data = _run_probe(tmp_path, "nrect", uri, """
const NO_TITLE = 2;
const TITLE_H = 30;
const node = { pos: [100, 80], size: [200, 120], flags: {} };
const r = M.nodeRect(node, { noTitle: NO_TITLE, titleHeight: TITLE_H });
const pos = M.nodePosFromRect(r, M.nodeTitleHeight(node, { noTitle: NO_TITLE, titleHeight: TITLE_H }));
process.stdout.write(JSON.stringify({ r, pos }));
""")
    assert data["r"] == {"x": 100, "y": 50, "w": 200, "h": 150}
    assert data["pos"] == [100, 80]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_node_rect_no_title(tmp_path):
    uri = CORE.resolve().as_uri()
    data = _run_probe(tmp_path, "notitle", uri, """
const NO_TITLE = 2;
const node = { pos: [10, 20], size: [80, 60], title_mode: 2 };
const r = M.nodeRect(node, { noTitle: NO_TITLE, titleHeight: 30 });
process.stdout.write(JSON.stringify(r));
""")
    assert data == {"x": 10, "y": 20, "w": 80, "h": 60}


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_node_rect_collapsed(tmp_path):
    uri = CORE.resolve().as_uri()
    data = _run_probe(tmp_path, "collapsed", uri, """
const r = M.nodeRect(
  { pos: [5, 40], size: [200, 100], flags: { collapsed: true }, _collapsed_width: 64 },
  { noTitle: 1, titleHeight: 30 },
);
process.stdout.write(JSON.stringify(r));
""")
    assert data == {"x": 5, "y": 10, "w": 64, "h": 30}


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_group_rect(tmp_path):
    uri = CORE.resolve().as_uri()
    data = _run_probe(tmp_path, "grect", uri, """
const r = M.groupRect({ pos: [12, 34], size: [300, 180] });
process.stdout.write(JSON.stringify(r));
""")
    assert data == {"x": 12, "y": 34, "w": 300, "h": 180}


def test_c2c_align_static_guards():
    text = EXT.read_text(encoding="utf-8")
    assert 'combo: { key: "a", alt: true }' in text
    assert 'commandId: CMD.SMART' in text or 'commandId: "C2C.Align.Smart"' in text
    for arrow, cmd in [
        ("ArrowLeft", "LEFT"),
        ("ArrowRight", "RIGHT"),
        ("ArrowUp", "TOP"),
        ("ArrowDown", "BOTTOM"),
    ]:
        assert f'key: "{arrow}", alt: true, shift: true' in text
    assert 'key: "x", alt: true, shift: true' in text
    assert 'key: "y", alt: true, shift: true' in text
    assert 'key: "h", alt: true, shift: true' in text
    assert 'key: "v", alt: true, shift: true' in text
    assert "LGraphCanvas.prototype.processMouseMove" not in text
    assert 'id: unitKey(item)' in text or 'unitKey(item)' in text
    assert "`g:${item.id}`" in text or '"g:${item.id}"' in text
    assert "units: null" in text
    assert not re.search(r"setInterval\s*\(", text)
    assert not re.search(r"requestAnimationFrame\s*\([^)]*tick", text, re.I)
    for m in re.finditer(r"(strokeStyle|fillStyle)\s*=\s*([^;]+)", text):
        assert "var(" not in m.group(2)


def test_item_types_are_duck_typed_not_array_checked():
    """node.pos / node.size are typed-array views in this frontend, not Arrays:
    `Array.isArray(item.pos)` rejected every node, so every command silently
    did nothing (found live 2026-09-28). Class names are not a contract either."""
    import re
    text = EXT.read_text(encoding="utf-8")
    assert not re.search(r"Array\.isArray\(\s*item\.(pos|size)\s*\)", text)
    assert "constructor?.name ===" not in text and "constructor.name ===" not in text
    assert "recomputeInsideNodes" in text[text.index("function isGroup"):text.index("function isReroute")]
