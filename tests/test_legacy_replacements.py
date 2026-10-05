"""CPU tests for legacy workflow compatibility (issue #11, ledger L2.02)."""
from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch

PACK_ROOT = Path(__file__).resolve().parents[1]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

for mod_name in ("folder_paths", "comfy", "comfy.utils", "server"):
    if mod_name not in sys.modules:
        stub = types.ModuleType(mod_name)
        stub.__path__ = []
        sys.modules[mod_name] = stub

TABLE_PATH = PACK_ROOT / "nodes" / "_legacy_replacements.json"
FIXTURE_PATH = PACK_ROOT / "tests" / "fixtures_legacy_signatures.json"
EXAMPLE_DIR = PACK_ROOT / "example_workflows"

NEW_NODE_MODULES = {
    "MaskEditMEC": "nodes.mask_edit_mec",
    "SplineMaskMEC": "nodes.spline_mask_mec",
    "MaskTrackerMEC": "nodes.mask_tracker_mec",
    "ProPainterMEC": "nodes.propainter_unified",
    "MaskOpsMEC": "nodes.mask_matting.node",
    "VideoStabilizerMEC": "nodes.video_stabilizer_mec",
}

EXTERNAL_ALLOWLIST = {
    "VHS_LoadVideo": "ComfyUI-VideoHelperSuite",
}

CORE_FALLBACK_ALLOWLIST = frozenset({
    "LoadImage",
    "PreviewImage",
    "SaveImage",
    "MaskToImage",
})


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _input_type_name(spec: Any) -> str:
    if isinstance(spec, (list, tuple)) and spec:
        # a combo's first element is its option list; the fixture records it as COMBO
        return "COMBO" if isinstance(spec[0], (list, tuple)) else str(spec[0])
    return str(spec)


def _combo_options(spec: Any) -> list[str] | None:
    if not isinstance(spec, (list, tuple)) or not spec:
        return None
    if isinstance(spec[0], (list, tuple)):
        return [str(x) for x in spec[0]]
    return None


def _flatten_input_types(cls: type) -> dict[str, tuple[str, str]]:
    raw = cls.INPUT_TYPES()
    out: dict[str, tuple[str, str]] = {}
    for section in ("required", "optional"):
        for name, spec in (raw.get(section) or {}).items():
            out[name] = (section, _input_type_name(spec))
    return out


def _resolve_new_class(new_node_id: str) -> type:
    module_name = NEW_NODE_MODULES.get(new_node_id)
    if module_name is None:
        pytest.fail(f"unknown new_node_id {new_node_id!r} in test map")
    mod = importlib.import_module(module_name)
    cls = getattr(mod, new_node_id, None)
    if cls is None:
        pytest.fail(f"{new_node_id} not found in {module_name}")
    return cls


def _expected_widget_ids(fixture_inputs: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for inp in fixture_inputs:
        if inp.get("widget"):
            ids.append(inp["name"])
            if inp.get("control"):
                ids.append("control_after_generate")
    return ids


def _fixture_inputs_by_name(fixture: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {inp["name"]: inp for inp in fixture.get("inputs") or []}


_LIVE_CACHE: dict[str, Any] = {}


def _live() -> dict[str, Any]:
    """One fresh interpreter with the REAL ComfyUI core (this test process has stubbed comfy
    modules): the pack's registered ids, core's built-in ids, and the legacy table registered into
    core's real NodeReplaceManager. Run once per session."""
    if not _LIVE_CACHE:
        import os
        import subprocess

        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.run([sys.executable, str(PACK_ROOT / "tests" / "_load_summary.py"), "--ids", "--replacements"],
                              cwd=str(PACK_ROOT), capture_output=True, text=True, encoding="utf-8",
                              timeout=600, env=env)
        assert proc.returncode == 0, f"_load_summary.py failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        for line in proc.stdout.splitlines():
            for key in ("C2C_IDS_JSON=", "C2C_REPLACEMENTS_JSON="):
                if line.startswith(key):
                    _LIVE_CACHE[key] = json.loads(line[len(key):])
    return _LIVE_CACHE


def _core_node_ids() -> set[str]:
    return set(_live()["C2C_IDS_JSON="]["core"])


def _pack_node_ids() -> set[str]:
    return set(_live()["C2C_IDS_JSON="]["pack"])


def _collect_workflow_node_types(data: dict[str, Any]) -> list[str]:
    types_found: list[str] = []
    for node in data.get("nodes") or []:
        if isinstance(node, dict) and node.get("type"):
            types_found.append(node["type"])
    definitions = data.get("definitions") or {}
    for subgraph in (definitions.get("subgraphs") or {}).values():
        for node in (subgraph.get("nodes") or []):
            if isinstance(node, dict) and node.get("type"):
                types_found.append(node["type"])
    return types_found


@pytest.fixture(scope="module")
def replacement_table() -> list[dict[str, Any]]:
    return _load_json(TABLE_PATH)


@pytest.fixture(scope="module")
def legacy_fixtures() -> dict[str, Any]:
    return _load_json(FIXTURE_PATH)


class TestReplacementTable:
    def test_every_new_node_importable(self, replacement_table):
        seen = {row["new_node_id"] for row in replacement_table}
        assert seen == set(NEW_NODE_MODULES)
        for new_id in seen:
            _resolve_new_class(new_id)

    def test_input_mappings_valid(self, replacement_table, legacy_fixtures):
        for row in replacement_table:
            old_id = row["old_node_id"]
            new_cls = _resolve_new_class(row["new_node_id"])
            new_inputs = _flatten_input_types(new_cls)
            fixture = legacy_fixtures[old_id]
            set_value_new_ids = set()
            mapped_new_ids = set()
            for entry in row.get("input_mapping") or []:
                new_key = entry["new_id"]
                assert new_key in new_inputs, (
                    f"{old_id}: new_id {new_key!r} missing from {row['new_node_id']}.INPUT_TYPES"
                )
                if "set_value" in entry:
                    set_value_new_ids.add(new_key)
                    options = _combo_options(
                        (new_cls.INPUT_TYPES().get("required") or {}).get(new_key)
                        or (new_cls.INPUT_TYPES().get("optional") or {}).get(new_key)
                    )
                    if options is not None:
                        assert entry["set_value"] in options, (
                            f"{old_id}: set_value {entry['set_value']!r} not in combo {new_key}"
                        )
                else:
                    mapped_new_ids.add(new_key)
                    assert entry.get("old_id"), f"{old_id}: mapping without old_id or set_value"
            assert not (set_value_new_ids & mapped_new_ids), (
                f"{old_id}: new_id used for both map and set_value: "
                f"{set_value_new_ids & mapped_new_ids}"
            )

    def test_output_mappings_valid(self, replacement_table, legacy_fixtures):
        for row in replacement_table:
            old_id = row["old_node_id"]
            new_cls = _resolve_new_class(row["new_node_id"])
            old_out = legacy_fixtures[old_id]["outputs"]
            new_out = new_cls.RETURN_TYPES
            for mapping in row.get("output_mapping") or []:
                oi = mapping["old_idx"]
                ni = mapping["new_idx"]
                assert 0 <= oi < len(old_out), f"{old_id}: old_idx {oi} out of range"
                assert 0 <= ni < len(new_out), f"{old_id}: new_idx {ni} out of range"
                assert old_out[oi] == new_out[ni], (
                    f"{old_id}: type mismatch old[{oi}]={old_out[oi]} new[{ni}]={new_out[ni]}"
                )

    def test_old_widget_ids_match_fixture(self, replacement_table, legacy_fixtures):
        for row in replacement_table:
            old_id = row["old_node_id"]
            fixture = legacy_fixtures[old_id]
            expected = _expected_widget_ids(fixture["inputs"])
            assert row.get("old_widget_ids") == expected, (
                f"{old_id}: old_widget_ids mismatch\n"
                f"  expected: {expected}\n"
                f"  actual:   {row.get('old_widget_ids')}"
            )
            mapped_old = {
                e["old_id"] for e in (row.get("input_mapping") or []) if "old_id" in e
            }
            dropped = set(row.get("dropped") or [])
            for inp in fixture["inputs"]:
                name = inp["name"]
                if name in mapped_old or name in dropped:
                    continue
                pytest.fail(
                    f"{old_id}: fixture input {name!r} not in input_mapping or dropped"
                )


class _StubNodeReplace:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _patch_node_replace(monkeypatch):
    fake_io = types.ModuleType("comfy_api.latest.io")
    fake_io.NodeReplace = _StubNodeReplace
    fake_latest = types.ModuleType("comfy_api.latest")
    fake_latest.io = fake_io
    fake_api = types.ModuleType("comfy_api")
    fake_api.latest = fake_latest
    monkeypatch.setitem(sys.modules, "comfy_api", fake_api)
    monkeypatch.setitem(sys.modules, "comfy_api.latest", fake_latest)
    monkeypatch.setitem(sys.modules, "comfy_api.latest.io", fake_io)


class TestRegister:
    def test_register_records_every_row(self, replacement_table, monkeypatch):
        _patch_node_replace(monkeypatch)
        from nodes._legacy_replacements import register

        calls: list[Any] = []

        class _Manager:
            def register(self, node_replace):
                calls.append(node_replace)

        class _Server:
            node_replace_manager = _Manager()

        count = register(_Server())
        assert count == len(replacement_table) == 17
        assert len(calls) == 17
        for row, call in zip(replacement_table, calls):
            assert call.old_node_id == row["old_node_id"]
            assert call.new_node_id == row["new_node_id"]
            assert call.old_widget_ids == (row.get("old_widget_ids") or None)
            assert call.input_mapping == (row.get("input_mapping") or None)
            assert call.output_mapping == (row.get("output_mapping") or None)

    def test_register_without_manager_returns_zero(self, monkeypatch):
        _patch_node_replace(monkeypatch)
        from nodes._legacy_replacements import register

        class _Server:
            pass

        assert register(_Server()) == 0

    def test_register_into_real_core_manager(self, replacement_table):
        """Real io.NodeReplace + real NodeReplaceManager, in a fresh interpreter with real core
        (this process has stubbed comfy modules, so it cannot import app.node_replace_manager)."""
        live = _live()["C2C_REPLACEMENTS_JSON="]
        assert live["registered"] == len(replacement_table) == 17
        for row in replacement_table:
            [entry] = live["table"][row["old_node_id"]]
            assert entry["new_node_id"] == row["new_node_id"]
            assert entry["old_widget_ids"] == (row.get("old_widget_ids") or None)
            assert entry["input_mapping"] == (row.get("input_mapping") or None)
            assert entry["output_mapping"] == (row.get("output_mapping") or None)


class TestLegacyCompat:
    @pytest.fixture(scope="class")
    def legacy_classes(self):
        from nodes.legacy_compat import NODE_CLASS_MAPPINGS

        return NODE_CLASS_MAPPINGS

    def test_deprecated_flag(self, legacy_classes):
        for node_id, cls in legacy_classes.items():
            assert getattr(cls, "DEPRECATED", False) is True, node_id

    def test_input_types_match_fixture(self, legacy_classes, legacy_fixtures):
        legacy_ids = {
            "MaskPreviewOverlay",
            "MaskCompositeAdvanced",
            "MaskMath",
            "BBoxFromMask",
            "BBoxPad",
            "BBoxCrop",
            "BBoxToMask",
        }
        for node_id in legacy_ids:
            cls = legacy_classes[node_id]
            fixture = legacy_fixtures[node_id]
            live = _flatten_input_types(cls)
            for inp in fixture["inputs"]:
                assert inp["name"] in live, f"{node_id}: missing input {inp['name']}"
                section, typ = live[inp["name"]]
                assert section == inp["section"], f"{node_id}.{inp['name']} section"
                assert typ == inp["type"], f"{node_id}.{inp['name']} type"

    def test_mask_math_invert(self, legacy_classes):
        cls = legacy_classes["MaskMath"]()
        mask = torch.tensor([[1.0, 0.0], [0.5, 1.0]])
        out = cls.compute(mask, "invert", 0.0, 1.0)[0]
        assert torch.allclose(out, 1.0 - mask)

    def test_mask_composite_ops(self, legacy_classes):
        cls = legacy_classes["MaskCompositeAdvanced"]()
        a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        b = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        union = cls.composite(a, b, "union", 0.5, False, False, 0.0)[0]
        assert torch.allclose(union, torch.ones_like(a))
        intersect = cls.composite(a, b, "intersect", 0.5, False, False, 0.0)[0]
        assert torch.allclose(intersect, torch.zeros_like(a))
        subtract = cls.composite(a, b, "subtract", 0.5, False, False, 0.0)[0]
        assert torch.allclose(subtract, a)

    def test_bbox_from_mask_rect(self, legacy_classes):
        cls = legacy_classes["BBoxFromMask"]()
        mask = torch.zeros(4, 6)
        mask[1:3, 1:4] = 1.0
        bbox, x, y, w, h, _ = cls.extract(mask, 0, 0, 0, 0.5)
        assert bbox == [1, 1, 3, 2]
        assert (x, y, w, h) == (1, 1, 3, 2)

    def test_bbox_to_mask_roundtrip(self, legacy_classes):
        from_cls = legacy_classes["BBoxFromMask"]()
        to_cls = legacy_classes["BBoxToMask"]()
        mask = torch.zeros(4, 6)
        mask[1:3, 1:4] = 1.0
        bbox, *_ = from_cls.extract(mask, 0, 0, 0, 0.5)
        out = to_cls.convert(bbox, 6, 4)[0][0]
        assert torch.allclose(out[1:3, 1:4], torch.ones(2, 3))
        assert out.sum() == 6.0

    def test_bbox_pad_clamps(self, legacy_classes):
        cls = legacy_classes["BBoxPad"]()
        bbox = [1, 1, 2, 2]
        out, _ = cls.pad(bbox, 10, 10, 10, 10, 6, 4)
        assert out == [0, 0, 6, 4]

    def test_bbox_crop(self, legacy_classes):
        cls = legacy_classes["BBoxCrop"]()
        image = torch.randn(1, 4, 6, 3)
        mask = torch.zeros(4, 6)
        mask[1:3, 1:4] = 1.0
        cropped_img, cropped_mask, out_bbox = cls.crop(image, [1, 1, 3, 2], mask)
        assert cropped_img.shape == (1, 2, 3, 3)
        assert cropped_mask.shape == (1, 2, 3)
        assert out_bbox == [1, 1, 3, 2]

    def test_mask_preview_overlay_shape(self, legacy_classes):
        cls = legacy_classes["MaskPreviewOverlay"]()
        image = torch.rand(1, 4, 6, 3)
        mask = torch.zeros(4, 6)
        mask[1:3, 1:4] = 1.0
        out = cls.preview(
            image, mask, "overlay", 1.0, 0.0, 0.0, 0.4, 0, False, 0.0, 1.0, 0.0
        )[0]
        assert out.shape == image.shape


class TestExampleWorkflows:
    def test_all_node_types_allowed(self):
        table = _load_json(TABLE_PATH)
        replacement_old = {row["old_node_id"] for row in table}
        from nodes.legacy_compat import NODE_CLASS_MAPPINGS as legacy_map

        # frontend-only node types: the litegraph/frontend creates them, the backend never sees them
        frontend_virtual = {"Note", "MarkdownNote", "Reroute", "PrimitiveNode"}
        allowed = (
            _pack_node_ids()
            | replacement_old
            | set(legacy_map)
            | _core_node_ids()
            | set(EXTERNAL_ALLOWLIST)
            | frontend_virtual
        )
        offenders: dict[str, list[str]] = {}
        for path in sorted(EXAMPLE_DIR.glob("*.json")):
            data = _load_json(path)
            bad = sorted({t for t in _collect_workflow_node_types(data) if t not in allowed})
            if bad:
                offenders[path.name] = bad
        assert not offenders, f"example workflow node types not allowed: {offenders}"


class TestResavedExamples:
    """The examples as re-saved from the live graph (L2.02): current node ids only, links consistent with both
    endpoints, a Templates thumbnail each, and all of them listed in the README."""

    @staticmethod
    def _examples() -> list[Path]:
        return sorted(EXAMPLE_DIR.glob("*.json"))

    def test_examples_use_current_node_ids_only(self):
        replaced = {row["old_node_id"] for row in _load_json(TABLE_PATH)}
        frontend_virtual = {"Note", "MarkdownNote", "Reroute", "PrimitiveNode"}
        current = _pack_node_ids() | _core_node_ids() | frontend_virtual
        offenders: dict[str, list[str]] = {}
        for path in self._examples():
            used = set(_collect_workflow_node_types(_load_json(path)))
            bad = sorted(t for t in used if t not in current or t in replaced or t in EXTERNAL_ALLOWLIST)
            if bad:
                offenders[path.name] = bad
        assert not offenders, f"examples must use current, in-pack node ids: {offenders}"

    def test_links_are_consistent_with_their_endpoints(self):
        for path in self._examples():
            wf = _load_json(path)
            nodes = {n["id"]: n for n in wf["nodes"]}
            seen: set[int] = set()
            for lid, o_id, o_slot, t_id, t_slot, _typ in wf["links"]:
                seen.add(lid)
                assert o_id in nodes and t_id in nodes, f"{path.name}: link {lid} references a missing node"
                assert lid in (nodes[o_id]["outputs"][o_slot].get("links") or []), \
                    f"{path.name}: link {lid} is not listed on its origin output"
                assert nodes[t_id]["inputs"][t_slot].get("link") == lid, \
                    f"{path.name}: link {lid} is not referenced by its target input"
            for n in wf["nodes"]:
                for inp in n.get("inputs") or []:
                    if inp.get("link") is not None:
                        assert inp["link"] in seen, f"{path.name}: node {n['id']}.{inp['name']} -> unknown link"
                for out in n.get("outputs") or []:
                    for lid in out.get("links") or []:
                        assert lid in seen, f"{path.name}: node {n['id']}.{out['name']} -> unknown link {lid}"

    def test_every_example_has_a_template_thumbnail(self):
        from PIL import Image

        for path in self._examples():
            jpg = path.with_suffix(".jpg")
            assert jpg.is_file(), f"{jpg.name} missing (ComfyUI's Templates browser loads <name>.jpg)"
            with Image.open(jpg) as im:
                assert im.format == "JPEG" and im.width >= 320 and im.height >= 200, (jpg.name, im.size)

    def test_readme_lists_every_example(self):
        readme = (EXAMPLE_DIR / "README.md").read_text(encoding="utf-8")
        missing = [p.name for p in self._examples() if f"`{p.name}`" not in readme]
        assert not missing, f"example_workflows/README.md does not list {missing}"


class TestMigrationDoc:
    def test_committed_doc_matches_generator(self):
        from tests._migration_doc import generate_migration_markdown

        doc_path = PACK_ROOT / "docs" / "MIGRATION.md"
        if not doc_path.is_file():
            pytest.skip("docs/MIGRATION.md not written yet — run tests/_migration_doc.py --write")
        assert doc_path.read_text(encoding="utf-8") == generate_migration_markdown()
