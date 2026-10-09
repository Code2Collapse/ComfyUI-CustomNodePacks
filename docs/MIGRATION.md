# Node migration guide (legacy → unified)

23 node ids have a unified successor. ComfyUI core's NodeReplaceManager (and this pack's server registration) maps saved workflows and API prompts from the old id to the new one. Core only rewrites an id that is no longer registered, so rows for deprecated nodes that still load change nothing until those nodes are removed.

When a workflow loads, the pack migrates every node whose id is gone automatically (setting: C2C › Workflows › Migrate old C2C nodes on load; off = ComfyUI offers the replacement instead). Links and values move to the successor; one notice lists what was updated, what changes on the successor, and anything that could not be carried over - nothing is dropped silently. A successor from another pack (core, NukeMax) has to be installed; otherwise the node is left as it was and the notice says so.

## Merged nodes (replacement table)

| Old id | New id | Mode / pinned | Renamed inputs | Translated values | Dropped | Output remap | Note |
|--------|--------|---------------|----------------|-------------------|---------|--------------|------|
| MaskTransformXY | MaskEditMEC | mode='transform' | — | — | — | 0→0 | — |
| MaskDrawFrame | MaskEditMEC | mode='draw_advanced' | — | — | — | 0→0 | — |
| DrawShapeMEC | MaskEditMEC | mode='draw_shape' | points_json→points_json_shape | — | coords_json | 0→0 | coords_json (per-frame positions from a Points editor) has no MaskEditMEC input; it is dropped |
| PointsMaskEditor | MaskEditMEC | mode='points_bbox' | — | — | — | 0→0, 1→1, 2→2, 3→3, 4→4, 5→5, 6→6, 7→7 | — |
| BBoxSmooth | MaskEditMEC | mode='bbox_smooth' | method→smoothing_method | — | — | 0→6, 1→7 | — |
| SplineMaskEditorMEC | SplineMaskMEC | mode='edit' | — | — | — | 0→0, 1→1, 2→2, 3→3, 4→4 | — |
| SplineMaskTrackerMEC | SplineMaskMEC | mode='track' | keyframes_json→spline_data | — | — | 0→0, 1→3 | — |
| SplinePathFlowMaskMEC | SplineMaskMEC | mode='flow_path' | — | — | — | 0→0 | — |
| MotionMaskTrackerMEC | MaskTrackerMEC | mode='motion' | images→video | — | — | 0→0, 1→2, 2→3 | — |
| MaskPropagateVideo | MaskTrackerMEC | mode='propagate' | images→video, mode→propagate_mode, flow_threshold→prop_flow_threshold | — | — | 0→0, 1→1 | — |
| TemporalAnchorMEC | MaskTrackerMEC | mode='anchor' | anchor_masks→mask, images→video | — | — | 0→0, 1→2, 2→3 | — |
| TemporalConsistencyCheckerMEC | MaskTrackerMEC | mode='consistency_check' | image→video | — | — | 0→1, 1→0, 2→2, 3→3 | — |
| ProPainterTemporalMEC | ProPainterMEC | mode='temporal' | — | — | — | 0→0, 1→4 | — |
| MaskMattingMEC | MaskOpsMEC | pipeline='cascade (legacy)' | — | — | — | 0→0, 1→1, 2→2, 3→3, 4→4, 5→5, 6→6, 7→7, 8→8, 9→9, 10→10, 11→11, 12→12 | pipeline is pinned to 'cascade (legacy)' so a migrated graph keeps the old matting path |
| VideoStabilizerClassicMEC | VideoStabilizerMEC | method='classic', preset='manual' | — | — | — | 0→0, 1→1, 2→2 | Deprecated shim (removal announced in video_stabilizer_mec.py); mirrors its own delegation: method=classic, preset=manual. |
| VideoStabilizerFlowMEC | VideoStabilizerMEC | method='raft_flow', preset='manual' | — | — | — | 0→0, 1→1, 2→2 | Deprecated shim; mirrors its own delegation: method=raft_flow, preset=manual. |
| VideoStabilizerAutoMEC | VideoStabilizerMEC | — | force_backend→method | — | — | 0→0, 1→1, 2→2 | Deprecated shim; force_backend becomes method. Its 'flow' arrives verbatim and VideoStabilizerMEC accepts it as an alias of raft_flow. |
| MECBuilderSampler | KSampler | — | — | — | cfg_mode, cfg_finish, cfg_pivot, self_correction, resolution_preset, custom_width, custom_height, vae | 0→0 | L7.65 P10 (owner-approved removal): Builder Sampler duplicated core KSampler. What changes: KSampler runs one constant CFG (the old CFG curve, 2-step polish pass, size preset and preview image are gone). If latent_image is empty, connect an Empty Latent Image - the old node made one itself. |
| MECToneRefiner | NukeMax_Grade | — | — | — | neural_corrector, corrector_tone, corrector_color, highlight_protection, shadow_lift, enable_upscale, upscale_factor, ai_enable_dof, ai_dof_strength, ai_dof_focus_depth, auto_upscale, latent, vae, upscale_model, depth_map | 0→0 | L7.65 P11 (owner-approved removal): an automatic levels + gray-world grade and a fake blur; no setting maps. What changes: Grade starts neutral, so the picture changes: Tone Refiner's automatic levels and colour balance, upscale and depth-of-field have no Grade equivalent. Set Grade by eye; the refined_latent output is gone. |
| C2CColorSpaceConvert | ColorSpaceConvertMEC | — | source_space→src_space, target_space→dst_space | source_space→src_space (sRGB→srgb, Linear→linear, Log C3→logc3); target_space→dst_space (sRGB→srgb, Linear→linear, Log C3→logc3) | — | 0→0 | L7.65 P14 (owner-approved removal): NukeMax Color Space Convert (gained logc3 for this). The CNP Log C3 toe had scrambled constants (near-black encoded to white). What changes: Same conversion; linear and Log C3 results are no longer clipped at 1.0, and the Log C3 toe now follows ARRI's curve (the old one turned near-black pixels white). |
| C2CACESTonemap | NukeMax_HDRToneMap | preset='None (Custom)', operator='filmic_aces', white_point=1.0, highlight_compression=0.0, shadow_lift=0.0 | — | exposure→exposure (log2: multiplier to stops, clamped to -5..5); output_colorspace→gamma (sRGB (gamma)→2.2, Linear→1.0, ACES AP1→1.0) | source_space | 0→0 | L7.65 P14 (owner-approved removal): NukeMax HDR Tone Map. exposure multiplier -> stops (log2). Measured (docs/evidence/L7.65/wave1_equivalence.json): same Narkowicz curve (linear out: 0.0 difference), sRGB out max 0.034, contrast 1.2 / saturation 0.8 max 0.41. What changes: HDR Tone Map expects scene-linear input: if the image is sRGB or Log C3 (the old node converted it itself), put Color Space Convert (srgb or logc3 -> linear) in front. The ACES curve is the same; the sRGB output now uses gamma 2.2 (up to 3% brighter or darker), and contrast and saturation act on the tone-mapped image, so if you changed them from 1.0, set them again by eye. |
| PromptRelayEncodeKijaiC2C | PromptRelayEncodeC2C | backend='kijai' | model→wan_model, t5→wan_t5 | — | — | 0→2, 1→3 | L7.65 P31 (owner-approved removal): both call nodes.prompt_relay._nodes._encode_kijai with the same arguments. |
| PromptRelayEncodeSmartC2C | PromptRelayEncodeC2C | backend='smart' | — | — | — | 0→0, 1→1 | L7.65 P31 (owner-approved removal): both call nodes.prompt_relay._nodes._encode_smart with the same arguments. |

## Removed nodes with no successor (2)

A loading workflow drops these nodes and their links, and the notice names each one.

| Node id | Where the information is now | Note |
|---------|------------------------------|------|
| InsightStatusMEC | per-node time and memory show as the Insight badges under each node. | L7.65 P26 (owner-approved removal): printed whether the Insight executor wrap is installed; the Insight overlay (js/nukenodemax/insight_overlay.js) shows per-node time and memory itself. |
| IntegrityStatusMEC | the integrity report is in the Diagnostics sidebar, Integrity tab. | L7.65 P26 (owner-approved removal): printed the latest integrity scan; the Diagnostics sidebar's Integrity tab shows it. |

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
