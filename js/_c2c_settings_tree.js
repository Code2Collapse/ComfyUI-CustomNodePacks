/**
 * _c2c_settings_tree.js - where every C2C setting sits in Settings > C2C, and the plain name it shows (L9.306,
 * owner 2026-11-23: "the settings of C2C are outdated and confusing").
 *
 * Measured before (live settings store, owner-stack testbed): 199 C2C settings in ~70 sub-groups, most of them holding
 * one or two rows, named after internal modules ("Vme", "Int", "WsLogger", "MemGuard", "StatsPill"), with brand
 * prefixes repeated inside names ("C2C > Auto-Checkpoint > Enabled") and a feature's two switches ("X: enabled" and
 * "Overlays > X button") filed in different groups.
 *
 * Now: a small fixed tree of task groups; within a group the rows of one feature sit together ("Undo panel: ...").
 * Only category, name and sortOrder change - setting IDS AND SAVED VALUES ARE UNTOUCHED, so nothing resets.
 * A setting missing from this table keeps the old automatic grouping (nothing ever disappears); the local test
 * tests/test_settings_tree.mjs fails when a registered C2C setting is not placed here.
 *
 * Side-effect-free ES module (pure data + functions), imported by _c2c_settings_grouping.js.
 */

/** [group, [[settingId, name], ...]] in display order. A null name keeps the setting's own name. */
export const TREE = [
  ["Appearance", [
    ["c2c.theme.variant", "Theme"],
    ["c2c.theme.reducedMotion", "Reduce motion"],
    ["c2c.brand.nodeColors", "Night node style"],
    ["c2c.i18n.locale", "Language"],
    ["mec.noodle.style", "Wires: style"],
    ["mec.noodle.shape", "Wires: pipe shape"],
    ["c2c.floatingPorts.enabled", "Wires: floating ports (Nuke-style, wires end at the node edge)"],
    ["c2c.nodeIdentity.hideTitles", "Node titles: hide"],
    ["c2c.nodeIdentity.hoverReveal", "Node titles: show name and type on hover"],
    ["c2c.sections.enabled", "Control sections inside nodes"],
    ["c2c.ui.layout", "Quick-action buttons: layout"],
    ["c2c.overlays.miniRow", "Quick-action buttons: compact launcher row"],
  ]],
  ["Celebrations", [
    ["c2c.overlays.completionFx", "Finish effect: show"],
    ["c2c.completion_fx.confetti", "Finish effect: play when a run succeeds"],
    ["c2c.completion_fx.style", "Finish effect: style"],
    ["c2c.completion_fx.intensity", "Finish effect: intensity"],
    ["c2c.completion_fx.min_runtime_ms", "Finish effect: skip runs shorter than (ms)"],
    ["mec.noodle.completion_fx", "Finish effect: on the wires"],
    ["c2c.completion_fx.chime", "Chime: play"],
    ["c2c.completion_fx.chime_style", "Chime: style"],
    ["c2c.celebrations.enabled", "Milestones: celebrate milestones"],
    ["c2c.celebrations.confetti", "Milestones: confetti"],
    ["c2c.celebrations.sound", "Milestones: sound"],
    ["c2c.celebrations.easterEggs", "Milestones: easter eggs"],
  ]],
  ["Canvas & wires", [
    ["c2c.align.guides", "Align: smart guides while dragging"],
    ["c2c.align.snapPx", "Align: snap distance (screen px)"],
    ["c2c.align.minGap", "Align: minimum gap (graph units)"],
    ["mec.autolayout.enabled", "Auto-layout: Tidy in the canvas menu"],
    ["mec.autolayout.col_gap", "Auto-layout: horizontal gap (px)"],
    ["mec.autolayout.row_gap", "Auto-layout: vertical gap (px)"],
    ["c2c.linkSnap.matchingSlot", "Wire drop snaps to the matching slot"],
    ["mec.compatibility_hints.enabled", "Highlight matching slots while dragging a wire"],
    ["c2c.connectedHighlight.enabled", "Trace connections on hover / select"],
    ["c2c.autoConnector.enabled", "Auto-connect: wire up newly added nodes"],
    ["c2c.autoconnect.enabled", "Auto-connect: learn from my connections"],
    ["c2c.slotDoubleClick", "Double-click a slot"],
    ["mec.wire_labels.enabled", "Wire labels (double-click a wire to name it)"],
    ["c2c.graph.repairMistypedLinks", "Repair mistyped wires"],
    ["c2c.linkedValues.enabled", "Linked values: on"],
    ["c2c.linkedValues.propagate_master_only", "Linked values: only the master value spreads"],
    ["C2C.BypassManager.ApplyToNew", "Bypass manager: new nodes follow their type"],
    ["c2c.focusMode.enabled", "Focus mode"],
    ["mec.isolate.enabled", "Isolate subgraph: menu entry"],
    ["mec.isolate.dim", "Isolate subgraph: dim the rest (0-1)"],
    ["c2c.subgraph.auto_promote_widgets", "Subgraphs: show inner widgets on the subgraph node"],
    ["c2c.subgraph.hide_widget_sockets", "Subgraphs: hide socket dots on widget rows"],
    ["c2c.subgraph.smart_title", "Subgraphs: smart title"],
    ["c2c.coreFix.subgraphAutogrowLinks", "Subgraphs: Convert to Subgraph keeps links on their inputs"],
    ["c2c.coreFix.clearStaleWidgetValues", "Clear workflow forgets old widget values"],
  ]],
  ["Editing & shortcuts", [
    ["c2c.commandPalette.enabled", "Command palette (Ctrl+K)"],
    ["c2c.workflowFind.enabled", "Find in this workflow (Ctrl+F)"],
    ["c2c.bookmarks.enabled", "Node bookmarks (Alt+B, Ctrl+Shift+1..9)"],
    ["c2c.bulkEdit.enabled", null],
    ["c2c.expressionFields.enabled", "Expressions in number fields (=expr)"],
    ["c2c.modelSwap.enabled", "Quick-swap models (Ctrl+M)"],
    ["c2c.modelSwap.lorasOnly", "Quick-swap models: LoRA loaders only"],
    ["mec.clipboard.autoCopy", "Copy node metadata with Ctrl+C"],
    ["c2c.reset.enabled", "Ctrl+R resets nodes to their defaults"],
    ["mec.seed_sweep.enabled", "Seed sweep (right-click menu)"],
    ["mec.batch_runner.enabled", "Batch runner (right-click menu)"],
    ["mec.batch_runner.max_runs", "Batch runner: most runs per batch"],
    ["c2c.group_presets.enabled", "Group presets: on"],
    ["c2c.overlays.groupPresetsBtn", "Group presets: show button"],
    ["c2c.ab_split.enabled", "A/B split: on"],
    ["c2c.overlays.abSplitBtn", "A/B split: show button"],
    ["mec.sticky_notes.enabled", "Sticky notes (right-click the canvas)"],
    ["mec.token_counter.enabled", "Token counter on prompt boxes"],
  ]],
  ["Workflows & files", [
    ["c2c.migrate.auto", "Update old C2C nodes when a workflow loads"],
    ["c2c.workflowDrop.enabled", "Load a workflow when a .json file is dropped"],
    ["c2c.safeDrop.enabled", "Safe image drop: on"],
    ["c2c.safeDrop.mode", "Safe image drop: mode"],
    ["c2c.safeDrop.confirmWorkflow", "Safe image drop: ask before loading a workflow found in an image"],
    ["c2c.metaInspect.enabled", "Show image metadata when a PNG is dropped"],
    ["c2c.autocheckpoint.enabled", "Auto-checkpoint: on"],
    ["c2c.autocheckpoint.ringSize", "Auto-checkpoint: snapshots kept"],
    ["c2c.autocheckpoint.showButton", "Auto-checkpoint: show button"],
    ["c2c.undo_panel.enabled", "Undo panel: keep graph snapshots"],
    ["c2c.undo_panel.max", "Undo panel: snapshots kept"],
    ["c2c.overlays.undoBtn", "Undo panel: show button"],
    ["c2c.reset.settingsButton", "Reset every C2C setting to its default"],
  ]],
  ["Performance & memory", [
    ["c2c.perf.mode", "Performance mode"],
    ["c2c.autobatch.mode", "Image batching"],
    ["c2c.memGuard.enabled", "Memory guard: on"],
    ["c2c.memGuard.warnPct", "Memory guard: trim at browser memory (%)"],
    ["c2c.browser.headroomGb", "Keep GPU memory free for the browser (GB)"],
    ["c2c.gpu.spillNotice", "Tell me when GPU memory spills into system RAM"],
    ["c2c.distributed.enabled", "Multi-GPU: on"],
    ["c2c.distributed.idleUnloadS", "Multi-GPU: unload idle GPU workers after (s)"],
  ]],
  ["Run monitoring", [
    ["mec.system_hud.enabled", "System HUD: on"],
    ["c2c.overlays.systemHud", "System HUD: show"],
    ["mec.system_hud.vram", "System HUD: include GPU memory"],
    ["mec.system_hud.cost", "System HUD: include AI cost"],
    ["c2c.statsPill.enabled", "Stats pill: on"],
    ["c2c.statsPill.placement", "Stats pill: placement"],
    ["c2c.statusStrip.enabled", "Status strip, bottom right (older)"],
    ["mec.progress_hud.enabled", "Progress HUD: per-node overlay"],
    ["mec.progress_hud.eta", "Progress HUD: time left and rate"],
    ["c2c.insight.enabled", "Per-node time and memory: on"],
    ["mec.insight_overlay.always_on", "Per-node time and memory: always show badges"],
    ["c2c.int.pollMs", "INT badge: refresh every (ms)"],
    ["c2c.int.windowSeconds", "INT badge: runtime window (s)"],
    ["c2c.flamegraph.enabled", "Flame graph: on"],
    ["c2c.flamegraph.auto_show", "Flame graph: open after each run"],
    ["c2c.overlays.flamegraphBtn", "Flame graph: show button"],
    ["mec.complexity_hud.enabled", "Complexity HUD: on"],
    ["c2c.overlays.complexityHud", "Complexity HUD: show"],
    ["mec.cost_estimator.enabled", "Cost estimator: on"],
    ["c2c.overlays.costBtn", "Cost estimator: show button"],
    ["c2c.tensor_inspector.enabled", "Tensor inspector"],
    ["c2c.wsLogger.enabled", "WebSocket logger: on"],
    ["c2c.wsLogger.buffer_size", "WebSocket logger: messages kept"],
    ["c2c.omnibar.visible", "OmniBar: show"],
    ["c2c.omnibar.position", "OmniBar: position"],
    ["c2c.omnibar.density", "OmniBar: density"],
  ]],
  ["Errors & help", [
    ["c2c.overlays.errorToast", "Error pop-ups: show"],
    ["c2c.error_toast.enabled", "Error pop-ups: rewrite in plain English"],
    ["c2c.ai.errorTranslator.enabled", "Error panel: large plain-English panel"],
    ["c2c.ai.errorTranslator.autoAskAI", "Error panel: ask AI for a deeper explanation"],
    ["c2c.doctor.showButton", "Workflow doctor: show button"],
    ["c2c.issues.button.show", "Workflow issues: show button"],
    ["c2c.issues.button.size", "Workflow issues: button size"],
    ["c2c.graphHealth.enabled", "Graph health overlay (dead ends, loops, loose wires)"],
    ["c2c.whats_wired.enabled", "What's wired: workflow legend"],
    ["c2c.overlays.whatsWired", "What's wired: show chip"],
    ["c2c.workflow_wizard.enabled", "Workflow wizard: on"],
    ["c2c.overlays.wizardBtn", "Workflow wizard: show button"],
    ["mec.node_explain.surface", "Node explainer: where it shows"],
    ["mec.node_explain.dwell_ms", "Node explainer: hover delay (ms)"],
    ["mec.node_explain.backend", "Node explainer: AI backend"],
    ["mec.node_explain.gguf_quant", "Node explainer: GGUF quantisation"],
    ["c2c.inspector.ai_widget_blurb", "Node explainer: AI descriptions of widgets"],
    ["c2c.depcheck.enabled", "Warn about pip conflicts after a Manager install"],
    ["c2c.registry.showOnBoot", "Show the component status notice at start-up"],
  ]],
  ["AI", [
    ["c2c.ai.paused", "Pause all AI calls"],
    ["c2c.ai.explainer.enabled", "'Explain with AI' in the node menu"],
    ["c2c.ai.hudVisible", "AI chip in the OmniBar"],
    ["c2c.overlays.aiStatusBar", "AI status pills, top right"],
    ["mec.surprise_me.enabled", "Surprise me: on"],
    ["c2c.overlays.surpriseBtn", "Surprise me: show button"],
    ["mec.surprise_me.profile", "Surprise me: default profile"],
    ["mec.surprise_me.dry_run", "Surprise me: try without queueing"],
    ["mec.surprise_me.spice_prompts", "Surprise me: AI style tags (chaos profile only)"],
  ]],
  ["Previews & video", [
    ["c2c.preview.method", "Sampler preview: method"],
    ["c2c.preview.noForce", "Sampler preview: do not force it on"],
    ["c2c.preview.videoFrame", "Video preview: frame shown"],
    ["c2c.video.autoDecoder", "Load video: automatic decoder may use"],
    ["c2c.video.uploadMaxMB", "Load video: largest upload (MB)"],
    ["c2c.video.sourceProxy", "Load video: write a source proxy"],
    ["c2c.framePlayer.proxyAbove", "Frame player: one proxy video above (frames)"],
    ["c2c.video.saveFfmpeg", "Save video: encode with FFmpeg"],
    ["c2c.saveVideo.naming", "Save video: file naming"],
    ["mec.frame_overlay.enabled", "Frame scrubber on multi-image previews: on"],
    ["c2c.overlays.frameOverlay", "Frame scrubber on multi-image previews: show"],
    ["mec.frame_overlay.fps", "Frame scrubber on multi-image previews: playback FPS"],
    ["c2c.vme.previewMaxPx", "Video mask editor: frame size (long side, 0 = original)"],
    ["c2c.vme.sampledFrames", "Video mask editor: sampled frames"],
    ["c2c.vme.cacheFrames", "Video mask editor: decoded frames kept"],
    ["c2c.vme.playFps", "Video mask editor: playback FPS"],
    ["c2c.matte.viewer", "Matte: viewer"],
    ["c2c.matte.livePreview", "Matte: live preview while running"],
    ["c2c.matte.sam31Refine", "Matte: SAM 3.1 refine pass"],
    ["c2c.inpaintCrop.livePreview", "Inpaint crop: live preview"],
    ["c2c.seedvr2.preview", "SeedVR2: preview"],
    ["mec.colorspace_badges.enabled", "Colour-space badges on image outputs"],
  ]],
  ["Colour management", [
    ["c2c.video.cm.processing", "Processing"],
    ["c2c.video.cm.builtinConfig", "Built-in OCIO config"],
    ["c2c.video.cm.ocioConfig", "OCIO config file (used when $OCIO is not set)"],
    ["c2c.video.cm.view", "Display view (empty = the config's default)"],
  ]],
  ["Models & downloads", [
    ["c2c.models.neverDownload", "Never download models"],
    ["c2c.models.fallbacks", "Model-free fallbacks"],
    ["c2c.lora_scrubber.enabled", "LoRA scrubber"],
  ]],
  ["Vault", [
    ["c2c.vault.scope", "Unlock applies to"],
    ["c2c.vault.ttlMinutes", "Unlock lasts (minutes)"],
  ]],
  ["OmniScale", [
    ["OmniScale.ApiKey", null], ["c2c.omniscale.check", null], ["c2c.omniscale.httpTimeoutS", null],
    ["c2c.omniscale.restWorkers", null], ["c2c.omniscale.restMaxRetries", null], ["c2c.omniscale.restPollS", null],
    ["c2c.omniscale.restDeadlineS", null], ["c2c.omniscale.maxDownloadGB", null], ["c2c.omniscale.maxImagesGB", null],
    ["c2c.omniscale.maxWaveFrames", null], ["c2c.omniscale.retouchMaxEdge", null],
  ]],
  ["LTX-Enhanced", [["c2c.ltxe.enabled", null], ["c2c.ltxe.hdr.enabled", null], ["c2c.ltxe.tiling.enabled", null]]],
  ["NukeMax", [["c2c.nukemax.uploadMaxMB.exr", null], ["c2c.nukemax.uploadMaxMB.media", null]]],
  ["MiniMax H3", [["c2c.minimax.autoContinue", null]]],
  ["Wan experiments", [["c2c.wne.proxyCacheMB", null]]],
];

/** Hidden bookkeeping settings: never shown, placed so the completeness test can tell them apart. */
export const HIDDEN = new Set(["c2c.ai.firstRunCompleted", "c2c.issues.button.state"]);

/**
 * One feature, two switches for the SAME on-screen button: the feature's own "show button" setting and the generic
 * Overlays switch. The Overlays row is folded into the feature's own row: a hidden-by-Overlays button turns the
 * feature's own switch off once, then the Overlays row is no longer shown. [overlayId, featureSettingId]
 */
export const MERGED_OVERLAYS = [
  ["c2c.overlays.autoCheckpointBtn", "c2c.autocheckpoint.showButton"],
  ["c2c.overlays.doctorBtn", "c2c.doctor.showButton"],
];

const GROUP_STEP = 1000;
const _index = (() => {
  const m = new Map();
  TREE.forEach(([group, rows], gi) => {
    rows.forEach(([id, name], ri) => {
      m.set(id, { group, name, sortOrder: (TREE.length - gi) * GROUP_STEP - ri });
    });
  });
  return m;
})();

/** {group, name|null, sortOrder} for a placed setting id, or null. */
export function placeSetting(id) {
  return _index.get(id) || null;
}

/** Every placed id (for tests). */
export function placedIds() {
  return [..._index.keys()];
}

/** Group names in display order. */
export function groups() {
  return TREE.map(([g]) => g);
}
