"""L7.99 slice 3 (ORDERS 11.4: at most 10 visible controls on a node): Inpaint Crop Pro's rarely used inputs are
flagged advanced + c2c_section; the inputs a switch reveals stay out of sections (js/c2c_mode_widgets.js shows them
with their switch); order, names and defaults are unchanged so saved workflows keep their values.
Live, both renderers: docs/evidence/L7.99/compact_ui_live.json (9 shown by default)."""
from pathlib import Path

import pytest

# The widget order saved workflows depend on (positional widgets_values) - must never change
ORDER = ['image', 'downscale_algorithm', 'upscale_algorithm', 'preresize', 'preresize_mode', 'preresize_min_width',
         'preresize_min_height', 'preresize_max_width', 'preresize_max_height', 'mask_fill_holes', 'mask_expand_pixels',
         'mask_invert', 'mask_blend_pixels', 'mask_hipass_filter', 'extend_for_outpainting', 'extend_up_factor',
         'extend_down_factor', 'extend_left_factor', 'extend_right_factor', 'context_from_mask_extend_factor',
         'auto_context_factor', 'aspect_preset', 'output_resize_to_target_size', 'output_target_width',
         'output_target_height', 'output_padding', 'device_mode', 'wan_align_multiple', 'wan_temporal_smooth_frames',
         'wan_stable_crop', 'wan_mask_polarity', 'inpaint_mask_mode', 'stitch_blend_mode', 'blend_radius',
         'video_stable_temporal_sigma', 'video_stable_dilate_px', 'video_stable_blur_sigma', 'fill_masked_area',
         'mask', 'optional_context_mask', 'roto_quality', 'preview_frame']
GATED = {"preresize_mode", "preresize_min_width", "preresize_min_height", "preresize_max_width", "preresize_max_height",
         "extend_up_factor", "extend_down_factor", "extend_left_factor", "extend_right_factor", "output_target_width",
         "output_target_height", "video_stable_temporal_sigma", "video_stable_dilate_px", "video_stable_blur_sigma"}
SOCKETS = {"image", "mask", "optional_context_mask"}
# L7.65 P08: the former Smart Image Crop / Inpaint Mask Prepare controls, appended after preview_frame; each group is
# shown by its crop_mode (js/c2c_mode_widgets.js), so it counts as switch-gated here
SMART = ["smart_resolution_mode", "smart_max_resolution", "smart_min_resolution", "smart_manual_width",
         "smart_manual_height", "smart_mask_grow_pixels", "smart_patch_mask_holes", "smart_no_mask_mode",
         "smart_force_divisibility"]
PREPARE = ["prepare_fill_holes", "prepare_remove_small_regions", "prepare_min_region_area", "prepare_grow_pixels",
           "prepare_inpaint_edge_mode", "prepare_stitch_edge_mode", "prepare_stitch_feather_radius",
           "prepare_temporal_smooth", "prepare_temporal_sigma"]
APPENDED = ["crop_mode", *SMART, *PREPARE]
GATED = GATED | set(SMART) | set(PREPARE)


@pytest.fixture(scope="module")
def spec():
    return pytest.importorskip("nodes.inpaint_suite").InpaintCropProMEC.INPUT_TYPES()


def _opts(spec):
    for g in ("required", "optional"):
        for n, s in spec[g].items():
            yield n, (s[1] if len(s) > 1 and isinstance(s[1], dict) else {})


def test_order_and_names_unchanged(spec):
    # L7.65 P08: "image" moved from required to the first optional input and the mode inputs were appended. What saved
    # workflows depend on is unchanged: the widgets in ORDER's order (positional widgets_values) with the new ones
    # after them, and the sockets in the same order (the front end adds sockets first, so they keep their slots).
    names = [n for g in ("required", "optional") for n in spec[g]]
    assert [n for n in names if n not in SOCKETS] == [n for n in ORDER if n not in SOCKETS] + APPENDED
    assert [n for n in names if n in SOCKETS] == [n for n in ORDER if n in SOCKETS] == ["image", "mask",
                                                                                         "optional_context_mask"]


def test_at_most_ten_controls_by_default(spec):
    shown = [n for n, o in _opts(spec) if n not in SOCKETS and not o.get("advanced") and n not in GATED]
    # the switch-gated ones visible by default: output_target_width / height (output_resize_to_target_size is on)
    shown += ["output_target_width", "output_target_height"]
    assert len(shown) <= 10, shown
    assert {"context_from_mask_extend_factor", "aspect_preset", "mask_expand_pixels", "mask_blend_pixels",
            "wan_stable_crop", "stitch_blend_mode", "preview_frame"} <= set(shown)


def test_sections_and_advanced_go_together_and_gated_inputs_stay_out(spec):
    for n, o in _opts(spec):
        assert bool(o.get("advanced")) == bool(o.get("c2c_section")), n
        if n in GATED:
            assert not o.get("c2c_section"), f"{n} is shown by its switch, not by a section"


def test_one_visibility_owner():
    js = Path(__file__).resolve().parents[1] / "js"
    assert not (js / "inpaint_crop_pro.js").exists()
    src = (js / "c2c_mode_widgets.js").read_text(encoding="utf-8")
    assert "InpaintCropProMEC: {" in src                                  # the node's one visibility owner
