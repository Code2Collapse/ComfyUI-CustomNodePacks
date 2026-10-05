# Node migration guide (legacy → unified)

Fourteen node ids were merged into unified nodes. ComfyUI core's NodeReplaceManager (and this pack's server registration) maps saved workflows and API prompts from the old id to the new one.

## Merged nodes (replacement table)

| Old id | New id | Mode / pinned | Renamed inputs | Dropped | Output remap | Note |
|--------|--------|---------------|----------------|---------|--------------|------|
| MaskTransformXY | MaskEditMEC | mode='transform' | — | — | 0→0 | — |
| MaskDrawFrame | MaskEditMEC | mode='draw_advanced' | — | — | 0→0 | — |
| DrawShapeMEC | MaskEditMEC | mode='draw_shape' | points_json→points_json_shape | coords_json | 0→0 | coords_json (per-frame positions from a Points editor) has no MaskEditMEC input; it is dropped |
| PointsMaskEditor | MaskEditMEC | mode='points_bbox' | — | — | 0→0, 1→1, 2→2, 3→3, 4→4, 5→5, 6→6, 7→7 | — |
| BBoxSmooth | MaskEditMEC | mode='bbox_smooth' | method→smoothing_method | — | 0→6, 1→7 | — |
| SplineMaskEditorMEC | SplineMaskMEC | mode='edit' | — | — | 0→0, 1→1, 2→2, 3→3, 4→4 | — |
| SplineMaskTrackerMEC | SplineMaskMEC | mode='track' | keyframes_json→spline_data | — | 0→0, 1→3 | — |
| SplinePathFlowMaskMEC | SplineMaskMEC | mode='flow_path' | — | — | 0→0 | — |
| MotionMaskTrackerMEC | MaskTrackerMEC | mode='motion' | images→video | — | 0→0, 1→2, 2→3 | — |
| MaskPropagateVideo | MaskTrackerMEC | mode='propagate' | images→video, mode→propagate_mode, flow_threshold→prop_flow_threshold | — | 0→0, 1→1 | — |
| TemporalAnchorMEC | MaskTrackerMEC | mode='anchor' | anchor_masks→mask, images→video | — | 0→0, 1→2, 2→3 | — |
| TemporalConsistencyCheckerMEC | MaskTrackerMEC | mode='consistency_check' | image→video | — | 0→1, 1→0, 2→2, 3→3 | — |
| ProPainterTemporalMEC | ProPainterMEC | mode='temporal' | — | — | 0→0, 1→4 | — |
| MaskMattingMEC | MaskOpsMEC | pipeline='cascade (legacy)' | — | — | 0→0, 1→1, 2→2, 3→3, 4→4, 5→5, 6→6, 7→7, 8→8, 9→9, 10→10, 11→11, 12→12 | pipeline is pinned to 'cascade (legacy)' so a migrated graph keeps the old matting path |

## Deprecated nodes kept as-is (7)

These classes have no unified successor. They are registered with `DEPRECATED = True` (hidden from search; old workflows still load).

| Node id | Display name | Description |
|---------|--------------|-------------|
| MaskPreviewOverlay | Mask Preview Overlay (MEC) (legacy) | Deprecated — kept so old workflows load. Closest current node: MaskOpsMEC preview output. |
| MaskCompositeAdvanced | Mask Composite Advanced (MEC) (legacy) | Deprecated — kept so old workflows load. Closest current node: core MaskComposite (fewer operations). |
| MaskMath | Mask Math (MEC) (legacy) | Deprecated — kept so old workflows load. No unified successor. |
| BBoxFromMask | BBox From Mask (MEC) (legacy) | Deprecated — kept so old workflows load. No unified successor. |
| BBoxToMask | BBox To Mask (MEC) (legacy) | Deprecated — kept so old workflows load. No unified successor. |
| BBoxPad | BBox Pad (MEC) (legacy) | Deprecated — kept so old workflows load. No unified successor. |
| BBoxCrop | BBox Crop (MEC) (legacy) | Deprecated — kept so old workflows load. Closest current node: SmartImageCropMEC. |

## Core limitation

Known core limitation (0.36.0): ``apply_replacements`` indexes ``node_struct["inputs"][old_id]`` directly, so an API prompt that omits an optional input of a legacy node raises KeyError (HTTP 500) instead of the 400 "node not found" it would get without a replacement - it failed either way.
