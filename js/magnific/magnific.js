// Magnific sign-in commands for the ComfyUI frontend. The heavy lifting
// (device grant, tokens) happens server-side in magnific/routes.py — this file
// only opens the verification URL, reports progress, and makes the login
// discoverable (startup toast + a Sign in button on the Save To node).
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { C } from "../_c2c_theme.js";
import { remapMagnificWidgetValues } from "./widget_remap.js";

const POLL_INTERVAL_MS = 2000;

// First characters of SIGN_IN_SENTINEL in magnific/nodes.py — the signal that
// the node definitions were built without a session.
const SIGN_IN_OPTION_PREFIX = "— sign in";

const SIGN_IN_BUTTON = "Sign in to Magnific";

const toast = (severity, summary, detail) => {
  const manager = app.extensionManager;
  if (manager?.toast?.add) {
    manager.toast.add({ severity, summary, detail, life: 8000 });
  } else {
    // Older frontends without the toast manager still get feedback.
    console.log(`[Magnific] ${summary}${detail ? ` — ${detail}` : ""}`); // console-ok fallback UI
  }
};

// Per-process token that proves a request comes from this ComfyUI's own web UI
// (TH-651). Fetched from the token bootstrap route — reachable only from a
// loopback, same-origin caller, so a rebound or cross-origin page never obtains
// it — then sent on every subsequent route call. The account routes reject any
// request without it. Cached, but the token is per-process: a backend restart
// mints a new one, so apiFetch drops the cache and refetches on a 403.
const UI_TOKEN_HEADER = "X-Magnific-Token";
let uiTokenPromise = null;

const loadUiToken = async () => {
  const response = await api.fetchApi("/magnific/csrf-token");
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const { token } = await response.json();
  if (!token) throw new Error("missing token");
  return token;
};

const uiToken = () => {
  if (!uiTokenPromise) {
    uiTokenPromise = loadUiToken().catch((error) => {
      uiTokenPromise = null; // a failed fetch must not wedge every later call
      throw error;
    });
  }
  return uiTokenPromise;
};

// api.fetchApi with the UI token header merged in — use for every Magnific route.
// A 403 means the cached token is stale (the backend restarted and reissued
// UI_TOKEN while the page kept running, a common reconnect-without-reload); drop
// it and retry once with a fresh one so the UI recovers without a hard refresh.
const apiFetch = async (route, options = {}) => {
  const send = async () => {
    const token = await uiToken();
    return api.fetchApi(route, {
      ...options,
      headers: { ...(options.headers ?? {}), [UI_TOKEN_HEADER]: token },
    });
  };
  const response = await send();
  if (response.status !== 403) return response;
  uiTokenPromise = null;
  return send();
};

const fetchJson = async (route, options) => {
  const response = await apiFetch(route, options);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
};

// --- Sign-in button on Magnific Save To nodes -------------------------------

// The catalog combo that carries the sentinel: `project` on Save To / Creation
// Picker, `asset` on Library Reference.
const CATALOG_WIDGETS = new Set(["project", "asset"]);

const needsSignInButton = (node) => {
  const catalog = node.widgets?.find((widget) => CATALOG_WIDGETS.has(widget.name));
  const first = catalog?.options?.values?.[0];
  return typeof first === "string" && first.startsWith(SIGN_IN_OPTION_PREFIX);
};

const syncSignInButton = (node) => {
  const index = node.widgets?.findIndex((widget) => widget.name === SIGN_IN_BUTTON) ?? -1;
  const needed = needsSignInButton(node);
  if (needed && index < 0) {
    // serialize:false — a dynamic widget must never shift widgets_values
    // indexes of saved workflows.
    node.addWidget("button", SIGN_IN_BUTTON, null, () => signIn(), { serialize: false });
  } else if (!needed && index >= 0) {
    node.widgets.splice(index, 1);
    node.setSize?.(node.computeSize());
  }
  node.setDirtyCanvas?.(true, true);
};

// --- Per-project folder narrowing on Save To ---------------------------------

const PROJECT_ROOT = "(project root)"; // mirrors PROJECT_ROOT in magnific/nodes.py

// project label → its folder paths ("Folder" / "Folder / Sub"), from the
// server's flattened catalog. null until loaded; reloaded on every defs refresh.
let folderGroups = null;

const loadFolderGroups = async () => {
  try {
    const data = await fetchJson("/magnific/folders");
    const groups = {};
    for (const item of data.choices ?? []) {
      const [project, ...rest] = item.label.split(" / ");
      groups[project] ??= [];
      if (rest.length) groups[project].push(rest.join(" / "));
    }
    folderGroups = groups;
  } catch {
    folderGroups = null; // keep the full (unnarrowed) combo — still valid
  }
};

const syncFolderOptions = (node) => {
  const project = node.widgets?.find((widget) => widget.name === "project");
  const folder = node.widgets?.find((widget) => widget.name === "folder");
  if (!project || !folder || !folderGroups) return;
  const values = [PROJECT_ROOT, ...(folderGroups[project.value] ?? [])];
  folder.options.values = values;
  if (!values.includes(folder.value)) folder.value = PROJECT_ROOT;
  node.setDirtyCanvas?.(true, true);
};

const attachFolderNarrowing = (node) => {
  const project = node.widgets?.find((widget) => widget.name === "project");
  if (!project) return;
  const callback = project.callback;
  project.callback = function (...args) {
    const result = callback?.apply(this, args);
    syncFolderOptions(node);
    return result;
  };
  if (folderGroups) syncFolderOptions(node);
  else loadFolderGroups().then(() => syncFolderOptions(node));
};

const syncAllSaveToNodes = () => {
  for (const node of app.graph?._nodes ?? []) {
    if (node.comfyClass !== "MagnificSaveTo" && node.comfyClass !== "MagnificCreationPicker") continue;
    syncSignInButton(node);
    syncFolderOptions(node);
  }
};

// --- Library asset combo on Library Reference ---------------------------------

// Combo entries ({label, type}) per catalog scope, from the server's catalog;
// a scope is null until loaded (or when its fetch failed). The Python side
// lists every label of both catalogs in INPUT_TYPES, so a value picked here
// always passes queue-time combo validation. Mirrors LIBRARY_CATALOGS /
// LIBRARY_TYPE_FILTERS in magnific/api.py.
const LIBRARY_SCOPE_BY_CATALOG = { "My Library": "mine", Magnific: "public" };
const LIBRARY_TYPE_BY_FILTER = { All: null, Characters: "character", Styles: "style", Elements: "product", Locations: "locations" };
const libraryChoices = { mine: null, public: null };

const loadLibraryChoices = async (scope, force = false) => {
  try {
    const data = await fetchJson(`/magnific/library?scope=${scope}${force ? "&force=1" : ""}`);
    libraryChoices[scope] = (data.choices ?? []).map((item) => ({ label: item.label, type: item.type }));
  } catch {
    libraryChoices[scope] = null; // keep whatever the node definitions listed
  }
};

// Each catalog is its own request so a failing public fetch leaves the
// personal list intact (and vice versa).
const loadAllLibraryChoices = (force = false) => Promise.all(Object.keys(libraryChoices).map((scope) => loadLibraryChoices(scope, force)));

const libraryScopeOf = (node) => LIBRARY_SCOPE_BY_CATALOG[node.widgets?.find((widget) => widget.name === "catalog")?.value] ?? "mine";

const syncLibraryOptions = (node) => {
  const asset = node.widgets?.find((widget) => widget.name === "asset");
  const choices = libraryChoices[libraryScopeOf(node)];
  if (!asset || !choices) return;
  const refType = LIBRARY_TYPE_BY_FILTER[node.widgets?.find((widget) => widget.name === "type")?.value] ?? null;
  const labels = choices.filter((item) => refType === null || item.type === refType).map((item) => item.label);
  // An empty narrowing keeps the catalog's full list rather than an empty
  // combo, which LiteGraph renders as a dead widget.
  asset.options.values = labels.length ? labels : choices.map((item) => item.label);
  if (!asset.options.values.includes(asset.value)) asset.value = asset.options.values[0];
  node.setDirtyCanvas?.(true, true);
};

const LIBRARY_REFRESH_BUTTON = "Refresh library";

const attachLibraryRefresh = (node) => {
  // Switching catalog or type re-narrows the asset combo in place, the way
  // model narrows resolution on Generate Image.
  for (const name of ["catalog", "type"]) {
    const widget = node.widgets?.find((entry) => entry.name === name);
    if (!widget) continue;
    const callback = widget.callback;
    widget.callback = function (...args) {
      const result = callback?.apply(this, args);
      syncLibraryOptions(node);
      return result;
    };
  }
  // serialize:false — same rule as the sign-in button: a dynamic widget must
  // never shift widgets_values indexes of saved workflows.
  node.addWidget(
    "button",
    LIBRARY_REFRESH_BUTTON,
    null,
    async () => {
      await loadAllLibraryChoices(true);
      const choices = libraryChoices[libraryScopeOf(node)];
      if (choices) {
        syncLibraryOptions(node);
        toast("success", "Magnific Library refreshed", `${choices.length} asset${choices.length === 1 ? "" : "s"} available.`);
      } else {
        toast("error", "Could not refresh the Magnific Library", "Sign in first (Magnific menu), or press R to retry.");
      }
    },
    { serialize: false },
  );
  if (libraryChoices.mine || libraryChoices.public) syncLibraryOptions(node);
  else loadAllLibraryChoices().then(() => syncLibraryOptions(node));
};

const syncAllLibraryNodes = () => {
  for (const node of app.graph?._nodes ?? []) {
    if (node.comfyClass !== "MagnificLibraryReference") continue;
    syncSignInButton(node);
    syncLibraryOptions(node);
  }
};

// --- Thumbnail galleries on the visual picker nodes ---------------------------

// Magnific's premium crown (editor-plugins/src/assets/icons/crown.svg), gold on
// a translucent pill exactly like the panel's MediaTile / the Magnific web.
const CROWN_SVG =
  '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M3 6.5l4.2 3.2L12 3.5l4.8 6.2L21 6.5l-1.6 11.2a1.5 1.5 0 0 1-1.49 1.3H6.09A1.5 1.5 0 0 1 4.6 17.7L3 6.5Z" fill="#FFC93D"/></svg>';

// A DOM-widget grid: a fetch button fills it with thumbnails; clicking one
// writes its key into the node's `selected` STRING widget (which is what
// serializes and what Python resolves at execution).
const attachGallery = (node, { buttonLabel, fetchItems, itemKey, itemThumb, itemTitle, itemBadge, resetOn = [] }) => {
  const selectedWidget = node.widgets?.find((entry) => entry.name === "selected");
  // Changing the browse context invalidates the pick: a selection made in one
  // folder/type must not silently execute under different widget values.
  for (const name of resetOn) {
    const widget = node.widgets?.find((entry) => entry.name === name);
    if (!widget) continue;
    const callback = widget.callback;
    widget.callback = function (...args) {
      const result = callback?.apply(this, args);
      if (selectedWidget?.value) {
        selectedWidget.value = "";
        grid.textContent = "Selection cleared — browse again";
        node.setDirtyCanvas?.(true, true);
      }
      return result;
    };
  }
  const grid = document.createElement("div");
  grid.style.cssText =
    "display:flex;flex-wrap:wrap;gap:4px;align-content:flex-start;overflow-y:auto;height:220px;padding:4px;font-size:11px;color:var(--c2c-sub);";
  const render = (items) => {
    grid.replaceChildren();
    if (!items.length) {
      grid.textContent = "No results";
      return;
    }
    for (const item of items) {
      const cell = document.createElement("div");
      cell.style.cssText = "position:relative;width:70px;height:70px;cursor:pointer;";
      const img = document.createElement("img");
      img.src = itemThumb(item);
      img.title = itemTitle(item) ?? "";
      img.loading = "lazy";
      img.style.cssText =
        "width:100%;height:100%;object-fit:cover;border-radius:4px;border:2px solid transparent;box-sizing:border-box;";
      cell.appendChild(img);
      const badge = itemBadge?.(item);
      if (badge) {
        const mark = document.createElement("span");
        mark.innerHTML = badge; // static, trusted markup (CROWN_SVG)
        mark.style.cssText =
          "position:absolute;top:2px;right:2px;display:flex;align-items:center;height:16px;padding:0 3px;border-radius:4px;background:rgba(23,23,23,.7);pointer-events:none;";
        cell.appendChild(mark);
      }
      const key = itemKey(item);
      if (selectedWidget?.value === key) img.style.borderColor = "var(--c2c-dangerHot)";
      cell.onclick = () => {
        if (selectedWidget) selectedWidget.value = key;
        for (const thumb of grid.querySelectorAll("img")) thumb.style.borderColor = "transparent";
        img.style.borderColor = "var(--c2c-dangerHot)";
        node.setDirtyCanvas?.(true, true);
      };
      grid.appendChild(cell);
    }
  };
  node.addWidget(
    "button",
    buttonLabel,
    null,
    async () => {
      grid.textContent = "Loading…";
      try {
        render(await fetchItems(node));
      } catch (error) {
        grid.textContent = String(error);
      }
    },
    { serialize: false },
  );
  node.addDOMWidget("magnific_gallery", "div", grid, { serialize: false });
  node.setSize?.([Math.max(node.size?.[0] ?? 0, 340), Math.max(node.size?.[1] ?? 0, 460)]);
};

const widgetValue = (node, name) => node.widgets?.find((entry) => entry.name === name)?.value;

const postJson = async (route, body) => {
  const response = await apiFetch(route, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error === "not_signed_in" ? "Sign in first (Magnific menu)" : (payload.error ?? `HTTP ${response.status}`));
  return payload;
};

const attachStockGallery = (node) =>
  attachGallery(node, {
    buttonLabel: "Search",
    fetchItems: async () => {
      const payload = await postJson("/magnific/stock-search", {
        query: widgetValue(node, "query"),
        filters: {
          content_type: widgetValue(node, "content_type"),
          license: widgetValue(node, "license"),
          orientation: widgetValue(node, "orientation"),
          ai_generated: widgetValue(node, "ai_generated"),
        },
      });
      return payload.items;
    },
    itemKey: (item) => `${item.id}|${item.type}`,
    itemThumb: (item) => item.previewUrl,
    itemTitle: (item) => `${item.title ?? item.id}${item.premium ? " (premium)" : ""}`,
    itemBadge: (item) => (item.premium ? CROWN_SVG : null),
    resetOn: ["query", "content_type", "license", "orientation", "ai_generated"],
  });

// Placeholder tile for creations without a thumbnail (typically audio).
// Built at call time so C.* tracks the active theme (data-URI cannot use var()).
function noThumbSvg() {
  return "data:image/svg+xml," +
    encodeURIComponent(
      `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 70 70"><rect width="70" height="70" rx="4" fill="${C.surface0}"/><path d="M44 20v20.5a6.5 6.5 0 1 1-3-5.5V26l-12 3v16.5a6.5 6.5 0 1 1-3-5.5V24l18-4Z" fill="${C.overlay0}"/></svg>`,
    );
}

const attachCreationGallery = (node) =>
  attachGallery(node, {
    buttonLabel: "Browse",
    fetchItems: async () => {
      const payload = await postJson("/magnific/folder-creations", {
        project: widgetValue(node, "project"),
        folder: widgetValue(node, "folder"),
        content_type: widgetValue(node, "content_type"),
      });
      return payload.items;
    },
    itemKey: (item) => item.identifier,
    itemThumb: (item) => item.thumbnailUrl || noThumbSvg(),
    itemTitle: (item) => item.prompt || item.tool || item.identifier,
    resetOn: ["project", "folder", "content_type"],
  });

// --- Per-model capability narrowing on Generate Video ------------------------

// slug → {durations, resolutions, aspectRatios}; null until loaded.
let videoModels = null;

const loadVideoModels = async () => {
  try {
    const data = await fetchJson("/magnific/video-models");
    videoModels = Object.fromEntries((data.models ?? []).map((model) => [model.slug, model]));
  } catch {
    videoModels = null; // keep the full (unnarrowed) combos — still valid
  }
};

const narrowCombo = (node, name, values, preferred) => {
  const widget = node.widgets?.find((entry) => entry.name === name);
  if (!widget || !values?.length) return;
  widget.options.values = values;
  if (!values.includes(widget.value)) widget.value = preferred && values.includes(preferred) ? preferred : values[0];
};

const syncVideoOptions = (node) => {
  const model = node.widgets?.find((entry) => entry.name === "model");
  if (!model || !videoModels) return;
  // 'auto' (or an unknown slug) keeps the full unions captured at attach time.
  const info = videoModels[model.value];
  const full = node.__magnificFullOptions ?? {};
  narrowCombo(node, "duration", info?.durations?.length ? info.durations.map(String) : full.duration, "5");
  narrowCombo(node, "resolution", info?.resolutions?.length ? ["auto", ...info.resolutions] : full.resolution, "auto");
  const ratios = info?.aspectRatios?.filter((ratio) => ratio !== "auto");
  narrowCombo(node, "aspect_ratio", ratios?.length ? ["auto", ...ratios] : full.aspect_ratio, "auto");
  node.setDirtyCanvas?.(true, true);
};

const attachVideoNarrowing = (node) => {
  const model = node.widgets?.find((entry) => entry.name === "model");
  if (!model) return;
  node.__magnificFullOptions = Object.fromEntries(
    ["duration", "resolution", "aspect_ratio"].map((name) => [
      name,
      [...(node.widgets?.find((entry) => entry.name === name)?.options?.values ?? [])],
    ]),
  );
  const callback = model.callback;
  model.callback = function (...args) {
    const result = callback?.apply(this, args);
    syncVideoOptions(node);
    return result;
  };
  if (videoModels) syncVideoOptions(node);
  else loadVideoModels().then(() => syncVideoOptions(node));
};

const syncAllVideoNodes = () => {
  for (const node of app.graph?._nodes ?? []) {
    if (node.comfyClass === "MagnificGenerateVideo") syncVideoOptions(node);
  }
};

// --- Per-model resolution narrowing on Generate Image -------------------------

// slug → {resolutions}; null until loaded.
let imageModels = null;

const loadImageModels = async () => {
  try {
    const data = await fetchJson("/magnific/image-models");
    imageModels = Object.fromEntries((data.models ?? []).map((model) => [model.slug, model]));
  } catch {
    imageModels = null; // keep the full (unnarrowed) combo — still valid
  }
};

const syncImageOptions = (node) => {
  const model = node.widgets?.find((entry) => entry.name === "model");
  if (!model || !imageModels) return;
  const info = imageModels[model.value];
  // A catalogued model that declares no tiers takes no resolution at all — the
  // server drops the field — so 'auto' alone is the honest choice list ('auto'
  // mode is one of those models). An unknown slug keeps the full union.
  const values = info ? ["auto", ...(info.resolutions ?? [])] : node.__magnificFullOptions?.resolution;
  narrowCombo(node, "resolution", values, "auto");
  node.setDirtyCanvas?.(true, true);
};

const attachImageNarrowing = (node) => {
  const model = node.widgets?.find((entry) => entry.name === "model");
  if (!model) return;
  node.__magnificFullOptions = {
    resolution: [...(node.widgets?.find((entry) => entry.name === "resolution")?.options?.values ?? [])],
  };
  const callback = model.callback;
  model.callback = function (...args) {
    const result = callback?.apply(this, args);
    syncImageOptions(node);
    return result;
  };
  if (imageModels) syncImageOptions(node);
  else loadImageModels().then(() => syncImageOptions(node));
};

const syncAllImageNodes = () => {
  for (const node of app.graph?._nodes ?? []) {
    if (node.comfyClass === "MagnificGenerateImage") syncImageOptions(node);
  }
};

// --- Legacy combo values → AI Suite labels ------------------------------------

// Pre-label workflows serialized raw API slugs. Python keeps those slugs as
// valid combo entries so queue-time validation passes even without this file;
// here we rewrite a loaded legacy value to its label and hide the legacy tail
// from the dropdown. Mirrors UPSCALE_*_BY_LABEL / MUSIC_MODEL_LABELS in
// magnific/api.py.
const LEGACY_COMBO_VALUES = {
  MagnificUpscaleImage: {
    engine: {
      automatic: "Automatic",
      magnific_illusio: "Illusio",
      magnific_sharpy: "Sharpy",
      magnific_sparkle: "Sparkle",
    },
    mode: {
      default: "Automatic",
      ultra: "Magnific v1 (high HDR)",
      "ultra-sublime": "Magnific v2 (sublime)",
      "ultra-photo": "Magnific v2 (photo)",
      "ultra-denoiser": "Magnific v2 (photo denoiser)",
    },
  },
  MagnificGenerateMusic: {
    model: {
      "google-lyria": "Google Lyria",
      "google-lyria-3": "Lyria 3 Short",
      "google-lyria-3-pro": "Lyria 3 Long",
      "elevenlabs-music-generation": "ElevenLabs Music",
      "elevenlabs-music-generation-v2": "ElevenLabs Music v2",
    },
  },
};

const migrateLegacyComboValues = (node) => {
  const widgetMaps = LEGACY_COMBO_VALUES[node.comfyClass];
  if (!widgetMaps) return;
  for (const [name, labelByLegacy] of Object.entries(widgetMaps)) {
    const widget = node.widgets?.find((entry) => entry.name === name);
    if (!widget) continue;
    if (Object.hasOwn(labelByLegacy, widget.value)) widget.value = labelByLegacy[widget.value];
    const values = widget.options?.values;
    if (Array.isArray(values)) {
      widget.options.values = values.filter((value) => !Object.hasOwn(labelByLegacy, value));
    }
  }
  node.setDirtyCanvas?.(true, true);
};

const syncAllLegacyCombos = () => {
  for (const node of app.graph?._nodes ?? []) migrateLegacyComboValues(node);
};

// --- Credit-cost estimate on generation nodes --------------------------------

const COST_NODES = new Set([
  "MagnificGenerateImage",
  "MagnificUpscaleImage",
  "MagnificGenerateVideo",
  "MagnificUpscaleVideo",
  "MagnificGenerateMusic",
  "MagnificVoiceover",
  "MagnificSkinEnhancer",
  "MagnificRemoveBackground",
]);
const COST_WIDGET = "estimated_cost";
// Widgets that never change the price — skipping them avoids a refetch per
// queued run (seed randomizes after every queue).
const COST_NEUTRAL_WIDGETS = new Set([COST_WIDGET, "seed", "prompt"]);

const formatCost = (estimate) => {
  if (typeof estimate?.credits !== "number") return "—";
  const approx = estimate.certainty && estimate.certainty !== "exact";
  let label = `${approx ? "≈ " : ""}${estimate.credits} credit${estimate.credits === 1 ? "" : "s"}`;
  if (estimate.range) label += ` (${estimate.range.min}–${estimate.range.max} by source size)`;
  if (estimate.note) label += ` ${estimate.note}`;
  return label;
};

const updateCost = async (node) => {
  const widget = node.widgets?.find((entry) => entry.name === COST_WIDGET);
  if (!widget) return;
  const inputs = {};
  for (const entry of node.widgets ?? []) {
    if (entry.name !== COST_WIDGET) inputs[entry.name] = entry.value;
  }
  try {
    const response = await apiFetch("/magnific/cost", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ node: node.comfyClass, inputs }),
    });
    const payload = await response.json().catch(() => ({}));
    if (response.ok) {
      widget.value = formatCost(payload);
    } else if (response.status === 401) {
      widget.value = "sign in to estimate";
    } else {
      // e.g. "The selected scale is invalid." — the config would fail the
      // real run too, so surfacing it here is the earliest possible warning.
      widget.value = payload.error ? String(payload.error).slice(0, 70) : "—";
    }
  } catch {
    widget.value = "—";
  }
  node.setDirtyCanvas?.(true, true);
};

const attachCostEstimate = (node) => {
  const widget = node.addWidget("text", COST_WIDGET, "…", () => {}, { serialize: false });
  widget.disabled = true;
  let timer = null;
  const schedule = () => {
    clearTimeout(timer);
    timer = setTimeout(() => updateCost(node), 600);
  };
  for (const entry of node.widgets ?? []) {
    if (COST_NEUTRAL_WIDGETS.has(entry.name)) continue;
    const callback = entry.callback;
    entry.callback = function (...args) {
      schedule();
      return callback?.apply(this, args);
    };
  }
  schedule();
};

// Re-fetch node definitions (what the R shortcut does) so the combos pick up
// the post-login projects/models without a manual step. Returns false on
// frontends without the API so the caller falls back to instructing the user.
const refreshNodeDefs = async () => {
  if (typeof app.refreshComboInNodes !== "function") return false;
  await app.refreshComboInNodes();
  await Promise.all([loadFolderGroups(), loadVideoModels(), loadImageModels(), loadAllLibraryChoices(true)]);
  syncAllSaveToNodes();
  syncAllVideoNodes();
  syncAllImageNodes();
  syncAllLibraryNodes();
  // The refreshed defs restore the full choice lists, legacy tail included.
  syncAllLegacyCombos();
  return true;
};

// --- Device-grant flow -------------------------------------------------------

let polling = false;

const waitForApproval = async () => {
  try {
    for (;;) {
      await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
      const status = await fetchJson("/magnific/signin/status");
      if (status.state === "ok") {
        const refreshed = await refreshNodeDefs().catch(() => false);
        toast(
          "success",
          "Signed in to Magnific",
          refreshed
            ? "Your projects and models are loaded."
            : "Press R (Refresh) so the nodes load your projects and models.",
        );
        return;
      }
      if (status.state === "error") {
        toast("error", "Magnific sign-in failed", status.error);
        return;
      }
      if (status.state === "idle") return; // signed out mid-flow
    }
  } catch (error) {
    toast("error", "Magnific sign-in failed", String(error));
  }
};

const signIn = async () => {
  // The guard arms synchronously on the first click — a double click during
  // the awaits below must not start a second device-grant flow.
  if (polling) {
    toast("info", "Magnific sign-in already in progress", "Approve it in the tab that was opened.");
    return;
  }
  polling = true;
  try {
    const current = await fetchJson("/magnific/status");
    if (current.signed_in) {
      toast("info", "Already signed in to Magnific");
      return;
    }
    const start = await fetchJson("/magnific/signin", { method: "POST" });
    if (start.error) {
      toast("error", "Magnific sign-in failed", start.error);
      return;
    }
    // Must be a user-gesture-adjacent open; the approval happens on magnific.com.
    window.open(start.verification_uri_complete, "_blank", "noopener");
    toast("info", "Approve the sign-in in the new tab", `Code: ${start.user_code}`);
    await waitForApproval();
  } catch (error) {
    toast("error", "Magnific sign-in failed", String(error));
  } finally {
    polling = false;
  }
};

const signOut = async () => {
  await fetchJson("/magnific/signout", { method: "POST" });
  await refreshNodeDefs().catch(() => false);
  toast("success", "Signed out of Magnific");
};

// --- Startup checks ----------------------------------------------------------

// Update banner + killswitch notice for zip installs, which never self-update.
// The server checks the distribution manifest (fail-open); here we only relay.
const checkForUpdate = async () => {
  try {
    const update = await fetchJson("/magnific/update");
    if (update.blocked) {
      toast(
        "error",
        "Magnific plugin disabled",
        `Version ${update.installed} was disabled — download ${update.latest ?? "the latest version"} from magnific.com/plugins.`,
      );
      return;
    }
    if (update.update_available) {
      toast(
        "info",
        `Magnific ${update.latest} is available`,
        "Download it from magnific.com/plugins (or update via ComfyUI Manager).",
      );
    }
  } catch {
    // Never bother the user because an update check failed.
  }
};

const checkSignedIn = async () => {
  try {
    const status = await fetchJson("/magnific/status");
    if (!status.signed_in) {
      toast(
        "info",
        "Sign in to Magnific",
        "Use the Magnific menu — or the button on a Magnific Save To node — to use the Magnific nodes.",
      );
    }
  } catch {
    // Status endpoint unreachable — the nodes surface their own errors.
  }
};

app.registerExtension({
  name: "magnific.auth",
  setup() {
    checkForUpdate();
    checkSignedIn();
  },
  beforeRegisterNodeDef(nodeType, nodeData) {
    const isSaveTo = nodeData.name === "MagnificSaveTo";
    const isVideo = nodeData.name === "MagnificGenerateVideo";
    const isImage = nodeData.name === "MagnificGenerateImage";
    const isStockPicker = nodeData.name === "MagnificStockPicker";
    const isCreationPicker = nodeData.name === "MagnificCreationPicker";
    const isLibrary = nodeData.name === "MagnificLibraryReference";
    const hasCost = COST_NODES.has(nodeData.name);
    if (!isSaveTo && !hasCost && !isStockPicker && !isCreationPicker && !isLibrary) return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      if (isSaveTo || isCreationPicker) {
        syncSignInButton(this);
        attachFolderNarrowing(this);
      }
      if (isLibrary) {
        syncSignInButton(this);
        attachLibraryRefresh(this);
      }
      if (isVideo) attachVideoNarrowing(this);
      if (isImage) attachImageNarrowing(this);
      if (isStockPicker) attachStockGallery(this);
      if (isCreationPicker) attachCreationGallery(this);
      if (hasCost) attachCostEstimate(this);
      migrateLegacyComboValues(this);
      return result;
    };
  },
  // Runs on the raw workflow JSON before any node is configured — the only
  // point where widgets_values can still be reordered to match the current
  // node definitions.
  beforeConfigureGraph(graphData) {
    remapMagnificWidgetValues(graphData, LiteGraph.registered_node_types);
  },
  // Runs after configure() has restored a saved workflow's widget values —
  // the point where a pre-label slug value can be rewritten to its label.
  loadedGraphNode(node) {
    migrateLegacyComboValues(node);
    // onNodeCreated narrowed `resolution` for the *default* model — the saved
    // one only lands here, and restoring it fires no widget callback.
    if (node.comfyClass === "MagnificGenerateImage") syncImageOptions(node);
    if (node.comfyClass === "MagnificLibraryReference") syncLibraryOptions(node);
  },
  commands: [
    // No "Magnific:" prefix: the commands live under the Magnific submenu, so
    // the prefix read as a repetition there (Slack: C0BQN1KSRDY/p1788162284466149).
    { id: "magnific.signin", label: "Sign in", function: signIn },
    { id: "magnific.signout", label: "Sign out", function: signOut },
  ],
  menuCommands: [{ path: ["Magnific"], commands: ["magnific.signin", "magnific.signout"] }],
});
