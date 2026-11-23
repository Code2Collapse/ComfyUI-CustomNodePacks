"""Magnific nodes: Save To (project/folder destination), Generate Image,
Upscale, and Stock Search.

Dynamic combos (projects/folders, model catalog) are resolved inside
INPUT_TYPES, which ComfyUI re-evaluates on every node-definition refresh — the
built-in Refresh action is the reload path after signing in, no custom JS
needed. A fetch failure degrades to a sentinel entry instead of breaking
/object_info for the whole node pack.
"""

import random
from typing import List, Optional

from . import api, config, references as reference_images, tensors, update_check
from . import metadata as creation_metadata
from .auth import NotSignedInError
from .mcp import McpError

try:                                                       # A28 layout declarations (display only)
    from .._layout_decl import with_layout
except ImportError:  # loaded as a plain file (tests): the same module by path, never an absolute import
    try:
        import importlib.util as _c2c_ilu
        import os as _c2c_os
        _c2c_spec = _c2c_ilu.spec_from_file_location("c2c_layout_by_path", _c2c_os.path.join(
            _c2c_os.path.dirname(_c2c_os.path.abspath(__file__)), '..', '_layout_decl.py'))
        _c2c_mod = _c2c_ilu.module_from_spec(_c2c_spec)
        _c2c_spec.loader.exec_module(_c2c_mod)
        with_layout = _c2c_mod.with_layout
    except Exception:  # pragma: no cover - layout keys are display only: never take the node down
        def with_layout(spec, layout):
            return spec

# A28 (docs/audit/A28/PARAM_CNP_B.md, Magnific): which controls show when. Display only - every input keeps its name,
# order, type and default and is still sent as before. The per-family slider rules follow the code comments and
# tooltips (UpscaleImage: "the server ignores the family that doesn't match"; SkinEnhancer "Faithful / Flexible
# version only"); they are NOT VERIFIED against the Magnific API.
_REFS = ["reference_image", "reference_image_2", "reference_image_3", "reference_image_4"]
_MAG_GENERATE_IMAGE = {
    "reference_type": {"c2c_show_if_linked": _REFS},
    "confirm_task_count": {"c2c_show_if": {"count": {"gt": 4}}},   # api.check_task_count: up to 4 run as they are
}
_MAG_UPSCALE_IMAGE = {
    "prompt": {"c2c_section": "Prompt"},
    **{k: {"c2c_show_if": {"precision": ["creative"]}} for k in ("creativity", "hdr", "resemblance", "fractality")},
    **{k: {"c2c_show_if": {"precision": ["precision"]}} for k in ("sharpness", "grain", "ultra_detail")},
}
_MAG_STOCK_SEARCH = {
    **{k: {"c2c_section": "Filters"} for k in ("content_type", "license", "orientation", "ai_generated")},
    "result_index": {"c2c_show_if": {"selection": ["index"]}},
    **{k: {"c2c_show_if": {"selection": ["random"]}} for k in ("range_from", "range_to", "seed")},
}
_MAG_UPSCALE_VIDEO = {
    "target_resolution": {"c2c_show_if": {"mode": ["magnific", "magnific_precision"]}},
    "fps_boost": {"c2c_show_if": {"mode": ["magnific", "magnific_precision"]}},
    "upscale_factor": {"c2c_show_if": {"mode": ["topaz"]}},
    "preset": {"c2c_show_if": {"mode": ["magnific"]}},
    **{k: {"c2c_show_if": {"mode": ["magnific"], "preset": ["custom"]}}
       for k in ("creativity", "sharpen", "smart_grain", "flavor", "premium_quality", "turbo")},
    "strength": {"c2c_show_if": {"mode": ["magnific_precision"]}},
}
# duration is sent for every model except the fixed-30 s ones (api.music_args: MUSIC_FIXED_DURATION), by label or slug
_MAG_GENERATE_MUSIC = {
    "duration_seconds": {"c2c_show_if": {"model": {"not": sorted(
        {*api.MUSIC_FIXED_DURATION, *(api.music_model_label(m) for m in api.MUSIC_FIXED_DURATION)})}}},
    "seed": {"c2c_section": "Re-run"},
}
_MAG_SKIN = {
    "skin_detail": {"c2c_show_if": {"version": ["faithful"]}},
    "optimized_for": {"c2c_show_if": {"version": ["flexible"]}},
}
_MAG_RETOUCH = {"prompt": {"c2c_show_if": {"mode": ["replace"]}}}
_MAG_PICKER_SELECTED = {"selected": {"c2c_hidden": True}}       # machine field: set by a thumbnail click

# Filed with the rest of this pack rather than as a top-level vendor menu.
# The node IDs are deliberately NOT renamed, so a workflow saved against the
# vendor's own pack still resolves here - which also means that installing
# that pack alongside this one registers the same fifteen IDs twice and
# ComfyUI keeps whichever loads last. Pick one.
CATEGORY = "C2C/Magnific"

FOLDER_TYPE = "MAGNIFIC_FOLDER"

# Placeholder combo entries for the Save To project dropdown when there is
# nothing real to list; resolve() maps each back to a specific, actionable error.
SIGN_IN_SENTINEL = "— sign in (Magnific menu) and press R to refresh —"
LOAD_FAILED_SENTINEL = "— couldn't load your projects — press R to retry —"
NO_PROJECTS_SENTINEL = "— no projects yet — create one at magnific.com, then press R —"
NO_VOICES_SENTINEL = "— no voices available — press R to retry —"
NO_ASSETS_SENTINEL = "— no characters, styles, elements or locations yet — create one at magnific.com, then press R —"
LIBRARY_LOAD_FAILED_SENTINEL = "— couldn't load your Library — press R to retry —"

PROJECT_ROOT = api.PROJECT_ROOT

# Shared tooltip for the optional folder socket. Omitting the folder omits
# folderReference on the MCP call, which the server resolves to the user's
# Personal project (AcceptsFolderDestination).
FOLDER_TOOLTIP = "Optional — from a Magnific Save To node. Not connected → your Personal project."

# A chain of Library Reference nodes travels as one list of catalog entries.
LIBRARY_REFS_TYPE = "MAGNIFIC_LIBRARY_REFS"
LIBRARY_TOOLTIP = (
    "Optional — from a Magnific Library Reference node. Attaches Library characters, styles, "
    "elements or locations; their @name tokens are added to the prompt when missing."
)

REFERENCE_IMAGE_TOOLTIP = (
    "Optional reference picture. A batch counts one reference per image. Use the other "
    "reference_image sockets for pictures of different sizes; reference_type applies to all of them."
)
EXTRA_REFERENCE_IMAGE_TOOLTIP = "Another reference picture (or batch), same reference_type as the first."

ASPECT_RATIOS = ["auto", "1:1", "16:9", "9:16", "2:3", "3:4", "4:5", "5:4", "3:2", "4:3", "21:9", "2:1", "1:2"]

# What stock_download can rasterise to a bitmap the graph can consume (vectors,
# PSDs and designs come back as JPG renders). Icons (SVG) and video have no
# IMAGE surface.
STOCK_CONTENT_TYPES = ["any", "photo", "vector", "illustration", "psd", "design"]
STOCK_IMAGE_TYPES = api.STOCK_IMAGE_TYPES


class MagnificSaveTo:
    """Picks the Magnific project (and optionally a folder inside it) new
    creations are stored in. Wiring its output into a generation node is
    optional — left unconnected, results go to the user's Personal project.
    web/magnific.js narrows the folder combo to the selected project's
    folders; the combo itself declares every folder path so any narrowed pick
    still passes ComfyUI's server-side combo validation."""

    @classmethod
    def INPUT_TYPES(cls):
        folder_paths: List[str] = []
        try:
            choices = api.folder_choices()
            projects = [c["label"] for c in choices if " / " not in c["label"]] or [NO_PROJECTS_SENTINEL]
            folder_paths = sorted({c["label"].split(" / ", 1)[1] for c in choices if " / " in c["label"]})
        except NotSignedInError:
            projects = [SIGN_IN_SENTINEL]
        except Exception:
            # Signed in but the listing failed (network, premium gate…) —
            # don't mislabel it as "sign in required".
            projects = [LOAD_FAILED_SENTINEL]
        return {
            "required": {
                "project": (projects, {"tooltip": "Your Magnific project. Press R after creating one at magnific.com to refresh the list."}),
                "folder": (
                    [PROJECT_ROOT] + folder_paths,
                    {
                        "default": PROJECT_ROOT,
                        "tooltip": "Folder inside the selected project; the list narrows to that project.",
                    },
                ),
            },
        }

    RETURN_TYPES = (FOLDER_TYPE,)
    RETURN_NAMES = ("folder",)
    OUTPUT_TOOLTIPS = ("Where new creations are stored - wire it to a generation node's folder input.",)
    FUNCTION = "resolve"
    CATEGORY = CATEGORY

    def resolve(self, project: str, folder: str = PROJECT_ROOT):
        if project == SIGN_IN_SENTINEL:
            raise NotSignedInError()
        if project == NO_PROJECTS_SENTINEL:
            raise McpError("Your Magnific library has no projects — create one at magnific.com, then press R to refresh")
        if project == LOAD_FAILED_SENTINEL:
            raise McpError("Could not load your projects — press R to refresh the node definitions and retry")
        label = project if folder in ("", PROJECT_ROOT) else f"{project} / {folder}"
        reference = api.folder_reference_for(label)
        if not reference:
            raise McpError(
                f"'{folder}' is not a folder of '{project}' — pick one from the folder dropdown, "
                "or press R to refresh if it was just created"
            )
        return (reference,)


class MagnificLibraryReference:
    """Picks one Library asset (character, style, element or location) to guide
    a generation, from the user's own catalog or Magnific's public one. Several
    assets chain through the optional `chain` socket, the same way Save To hands
    a folder along: the output is the whole chain."""

    @classmethod
    def INPUT_TYPES(cls):
        # The combo must list BOTH catalogs: ComfyUI validates the picked value
        # against this list at queue time, whichever catalog the JS narrowed to.
        try:
            assets = [choice["label"] for choice in api.library_choices("mine")]
        except NotSignedInError:
            assets = [SIGN_IN_SENTINEL]
        except Exception:
            assets = [LIBRARY_LOAD_FAILED_SENTINEL]
        else:
            # The public catalog is a bonus: its outage must not hide the personal one.
            try:
                assets += [choice["label"] for choice in api.library_choices("public")]
            except Exception:
                pass
            assets = assets or [NO_ASSETS_SENTINEL]
        return {
            "required": {
                "catalog": (
                    list(api.LIBRARY_CATALOGS),
                    {"default": api.DEFAULT_LIBRARY_CATALOG, "tooltip": "My Library: your own and shared assets. Magnific: the public catalog."},
                ),
                "type": (list(api.LIBRARY_TYPE_FILTERS), {"default": "All", "tooltip": "Narrows the asset list to one kind."}),
                "asset": (assets, {"tooltip": "@name (Type · Origin), grouped by type. Only ready assets are listed; press Refresh library or R."}),
            },
            "optional": {
                "chain": (LIBRARY_REFS_TYPE, {"tooltip": "Another Library Reference node, to attach several assets to one generation."}),
            },
        }

    RETURN_TYPES = (LIBRARY_REFS_TYPE,)
    RETURN_NAMES = ("library_references",)
    OUTPUT_TOOLTIPS = ("This asset plus everything chained into it - wire to a generation node or another Library Reference.",)
    FUNCTION = "resolve"
    CATEGORY = CATEGORY

    def resolve(self, catalog: str, type: str, asset: str, chain: Optional[List[dict]] = None):
        if asset == SIGN_IN_SENTINEL:
            raise NotSignedInError()
        if asset == NO_ASSETS_SENTINEL:
            raise McpError("Your Magnific Library has no characters, styles, elements or locations — create one at magnific.com, then press R to refresh")
        if asset == LIBRARY_LOAD_FAILED_SENTINEL:
            raise McpError("Could not load your Library — press R to refresh the node definitions and retry")
        scope = api.library_scope_for_catalog(catalog)
        other = "public" if scope == "mine" else "mine"
        # The label carries its origin, so a value from the other catalog (the
        # combo was switched after picking) is still honored before failing. A
        # miss on the cached catalog is re-checked against a fresh one: the
        # asset may simply have been renamed or shared since.
        entry = api.library_entry_for(asset, scope) or _quiet_library_entry(asset, other) or api.library_entry_for(asset, scope, force=True)
        if entry is None:
            raise McpError(
                f"'{asset}' is no longer available in the {catalog} catalog (deleted, renamed, unshared or "
                "still processing) — pick another asset from the dropdown, or press Refresh library"
            )
        entries = [ref for ref in (chain or []) if ref["id"] != entry["id"]] + [entry]
        if len(entries) > api.MAX_GENERATION_REFERENCES:
            raise McpError(f"A generation takes at most {api.MAX_GENERATION_REFERENCES} references — remove some Library Reference nodes from the chain")
        return (entries,)


def _quiet_library_entry(label: str, scope: str) -> Optional[dict]:
    # The fallback catalog must not turn its own outage into this node's error.
    try:
        return api.library_entry_for(label, scope)
    except Exception:
        return None


def _checked_library_refs(
    entries: Optional[List[dict]],
    model: str,
    models: List[dict],
    surface_types: tuple,
    surface: str,
) -> List[dict]:
    """The Library entries a generation may send, or a clear error. Gated
    client-side because the server's 422 arrives only after every reference
    image has already been uploaded. `auto` (or a model the catalog doesn't
    know) sends everything the surface takes and lets the server resolve."""
    if not entries:
        return []
    for entry in entries:
        label = api.library_type_label(entry["type"]).lower()
        if entry["type"] not in surface_types:
            raise McpError(
                f"{surface} does not take {label} references — disconnect '@{entry['name']}' from this node"
            )
    if model != "auto":
        info = next((m for m in models if m["slug"] == model), None)
        if info is not None:
            allowed = info.get("referenceTypes", [])
            for entry in entries:
                if entry["type"] not in allowed:
                    accepts = ", ".join(api.library_type_label(t).lower() for t in allowed if t in api.LIBRARY_REF_TYPES)
                    raise McpError(
                        f"Model '{model}' does not accept {api.library_type_label(entry['type']).lower()} references"
                        f" ({'it takes ' + accepts if accepts else 'it takes no Library references'}) — "
                        f"pick another model or set model to auto, or disconnect '@{entry['name']}'"
                    )
    return list(entries)


def _image_models_quiet() -> List[dict]:
    try:
        return api.image_models()
    except Exception:
        return []  # no catalog, no client-side gating — the server still validates


def _video_models_quiet() -> List[dict]:
    try:
        return api.video_models()
    except Exception:
        return []


class MagnificGenerateImage:
    """web/magnific.js narrows `resolution` to the selected model's tiers (via
    /magnific/image-models); the combo declares the union across models so any
    narrowed pick passes combo validation."""

    @classmethod
    def INPUT_TYPES(cls):
        resolutions: List[str] = []
        try:
            catalog = api.image_models()
            # The catalog itself carries an 'auto' slug — dedupe or the combo
            # shows it twice.
            models = ["auto"] + [m["slug"] for m in catalog if m["slug"] != "auto"]
            resolutions = sorted({r for m in catalog for r in m["resolutions"]})
        except Exception:
            models = ["auto"]  # 'auto' always works; the catalog loads on the next refresh
        return with_layout({
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "What to generate."}),
                "model": (models, {"default": "auto", "tooltip": "Image model; auto lets Magnific choose. The list comes from your account (press R to refresh)."}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "auto", "tooltip": "Shape of the image; auto = the model's default."}),
                "count": ("INT", {"default": 1, "min": 1, "max": 8, "tooltip": "How many images; each one is a separate paid generation (above 4 needs confirm_task_count)."}),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Sent to Magnific. Same seed and settings reproduce the result on models that honor it; a batch uses seed, seed+1, … per image.",
                    },
                ),
            },
            "optional": {
                # FIRST optional key on purpose: it was the first required
                # socket, and required→optional keeps the socket order (and
                # saved link slots) only while it stays ahead of the others.
                "folder": (FOLDER_TYPE, {"tooltip": FOLDER_TOOLTIP}),
                # The extra reference sockets sit right after the first one.
                # Link sockets can move: the frontend reattaches a saved link
                # to the socket with the same name, verified against a 0.6.0
                # workflow with `library_references` pushed three slots down.
                "reference_image": ("IMAGE", {"tooltip": REFERENCE_IMAGE_TOOLTIP}),
                "reference_image_2": ("IMAGE", {"tooltip": EXTRA_REFERENCE_IMAGE_TOOLTIP}),
                "reference_image_3": ("IMAGE", {"tooltip": EXTRA_REFERENCE_IMAGE_TOOLTIP}),
                "reference_image_4": ("IMAGE", {"tooltip": EXTRA_REFERENCE_IMAGE_TOOLTIP}),
                "reference_type": (["image", "style"], {"default": "image", "tooltip": "How the reference pictures are used: image = content to follow, style = look to copy."}),
                # Widget order still matters: widgets_values is positional in a
                # saved workflow. web/magnific.js rebuilds it by name on load,
                # but only for workflows whose frontend recorded the widget
                # names in `inputs`, so append new widgets rather than insert.
                "resolution": (
                    ["auto"] + resolutions,
                    {"default": "auto", "tooltip": "auto = model default; narrows to the selected model."},
                ),
                "library_references": (LIBRARY_REFS_TYPE, {"tooltip": LIBRARY_TOOLTIP}),
                # L9.191 (owner A17): appended last so saved workflows keep their widget layout
                "confirm_task_count": ("INT", {"default": 0, "min": 0, "max": 8, "tooltip": (
                    "Every image is a separate paid generation. Counts of up to 4 run as they are; for a bigger "
                    "count type the same number here to submit it.")}),
            },
        }, _MAG_GENERATE_IMAGE)

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "creation_identifiers", "metadata")
    OUTPUT_TOOLTIPS = ("The generated images as one batch.", "The creation id of each image, one per line.", "Model, prompt, seed, tool and url of the result as JSON - split it with the Metadata (unpack) node.")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(
        self,
        prompt: str,
        model: str,
        aspect_ratio: str,
        count: int,
        seed: int,
        folder: Optional[str] = None,
        reference_image=None,
        reference_type: str = "image",
        resolution: str = "auto",
        library_references: Optional[List[dict]] = None,
        reference_image_2=None,
        reference_image_3=None,
        reference_image_4=None,
        confirm_task_count: int = 0,
    ):
        update_check.assert_not_blocked()
        if not prompt.strip():
            raise McpError("Prompt is empty")
        # L9.191 (A17): a count above the free limit needs the typed confirmation; refused before anything is sent
        api.check_task_count("Generate Image", count, confirm_task_count)
        # Checked before the reference upload so a rejected combination costs
        # nothing.
        library = _checked_library_refs(
            library_references, model, _image_models_quiet(), api.LIBRARY_REF_TYPES, "Image generation"
        )
        pictures = reference_images.frames(reference_image, reference_image_2, reference_image_3, reference_image_4)
        if len(library) + len(pictures) > api.MAX_GENERATION_REFERENCES:
            raise McpError(
                f"images_generate takes at most {api.MAX_GENERATION_REFERENCES} references and this node has "
                f"{len(pictures)} reference images plus {len(library)} Library references — disconnect some"
            )
        references: List[dict] = [
            {"type": reference_type, "identifier": _upload_image_tensor(picture)} for picture in pictures
        ]
        references.extend(api.library_image_references(library))
        refs = api.generate_image(
            prompt=api.with_library_mentions(prompt, library),
            mode=model,
            count=count,
            aspect_ratio=aspect_ratio,
            references=references or None,
            folder_reference=folder,
            resolution=resolution,
            seed=seed,
        )
        identifiers = [ref["identifier"] for ref in refs]
        entries = api.wait_for_creations(identifiers)
        urls = api.result_urls(entries)
        if len(urls) < len(identifiers):
            raise McpError(f"Only {len(urls)} of {len(identifiers)} generations returned an image URL")
        return (tensors.urls_to_batch(urls), "\n".join(identifiers), creation_metadata.for_identifiers(identifiers))


class MagnificUpscaleImage:
    @classmethod
    def INPUT_TYPES(cls):
        return with_layout({
            "required": {
                "image": ("IMAGE", {"tooltip": "The picture to upscale (uploaded to Magnific)."}),
                "scale": (["2x", "4x", "8x", "16x"], {"default": "2x", "tooltip": "Size multiplier."}),
                "precision": (["creative", "precision"], {"default": "creative", "tooltip": "creative = invents new detail (the -10..10 sliders apply); precision = stays faithful to the input (the 0..100 sliders apply)."}),
                "presets": (["subtle", "vivid", "wild"], {"default": "subtle", "tooltip": "How strong the added detail is."}),
                "engine": (api.UPSCALE_ENGINE_CHOICES, {"default": "Automatic", "tooltip": "Upscaling engine; Automatic lets Magnific choose."}),
                "mode": (api.UPSCALE_MODE_CHOICES, {"default": "Automatic", "tooltip": "Magnific model version; Automatic lets the server choose."}),
            },
            "optional": {
                # First optional key: preserves the socket order (image, folder)
                # this input had while required, so saved link slots don't shift.
                "folder": (FOLDER_TYPE, {"tooltip": FOLDER_TOOLTIP}),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional: describe the picture to guide the new detail."}),
                # Creative sliders (-10..10); the server ignores the family that
                # doesn't match the chosen precision.
                "creativity": ("INT", {"default": 0, "min": -10, "max": 10, "tooltip": "Creative mode: how much new detail is invented."}),
                "hdr": ("INT", {"default": 0, "min": -10, "max": 10, "tooltip": "Creative mode: local contrast / definition."}),
                "resemblance": ("INT", {"default": 0, "min": -10, "max": 10, "tooltip": "Creative mode: how closely the result keeps to the original."}),
                "fractality": ("INT", {"default": 0, "min": -10, "max": 10, "tooltip": "Creative mode: how much fine, repeating detail the prompt adds per pixel."}),
                # Precision sliders (0..100).
                "sharpness": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "Precision mode: sharpening."}),
                "grain": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "Precision mode: added film grain."}),
                "ultra_detail": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "Precision mode: extra fine detail."}),
            },
        }, _MAG_UPSCALE_IMAGE)

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "creation_identifier", "metadata")
    OUTPUT_TOOLTIPS = ("The upscaled picture.", "The Magnific creation id of the result (find it in your library, or pass it on).", "Model, prompt, seed, tool and url of the result as JSON - split it with the Metadata (unpack) node.")
    FUNCTION = "upscale"
    CATEGORY = CATEGORY

    def upscale(
        self,
        image,
        scale: str,
        precision: str,
        presets: str,
        engine: str,
        mode: str,
        folder: Optional[str] = None,
        prompt: str = "",
        creativity: int = 0,
        hdr: int = 0,
        resemblance: int = 0,
        fractality: int = 0,
        sharpness: int = 0,
        grain: int = 0,
        ultra_detail: int = 0,
    ):
        update_check.assert_not_blocked()
        identifier = _upload_image_tensor(image)
        ref = api.upscale_image(
            identifier,
            folder,
            {
                "scale": scale,
                "precision": precision,
                "presets": presets,
                "engine": api.upscale_engine_slug(engine),
                "mode": api.upscale_mode_slug(mode),
                "prompt": prompt.strip() or None,
                "creativity": creativity,
                "hdr": hdr,
                "resemblance": resemblance,
                "fractality": fractality,
                "sharpness": sharpness,
                "grain": grain,
                "ultraDetail": ultra_detail,
            },
        )
        entries = api.wait_for_creations([ref["identifier"]])
        urls = api.result_urls(entries)
        if not urls:
            raise McpError("Upscale finished but returned no image URL")
        return (tensors.urls_to_batch(urls[:1]), ref["identifier"], creation_metadata.for_identifiers([ref["identifier"]]))


def _target_index(selection: str, result_index: int, range_from: int, range_to: int, seed: int) -> int:
    """random mode picks uniformly in [range_from, range_to]; seeded so a fixed
    seed reproduces the pick and a randomized seed varies it per run."""
    if selection != "random":
        return result_index
    low, high = sorted((range_from, range_to))
    return random.Random(seed).randint(low, high)


class MagnificStockSearch:
    """Magnific stock search — returns one result as IMAGE, picked by index or
    at random within a range; the info output lists what else the query matched."""

    @classmethod
    def INPUT_TYPES(cls):
        return with_layout({
            "required": {
                "query": ("STRING", {"default": "", "tooltip": "What to search the Magnific stock library for."}),
                "content_type": (STOCK_CONTENT_TYPES, {"default": "photo", "tooltip": "Kind of stock item; vectors, PSDs and designs come back as JPG renders."}),
                "license": (["any", "free", "premium"], {"default": "any", "tooltip": "Only free items, only premium items, or both."}),
                "orientation": (["any", "portrait", "landscape", "square", "panoramic"], {"default": "any", "tooltip": "Shape filter."}),
                "ai_generated": (["any", "only", "excluded"], {"default": "any", "tooltip": "Include AI-generated items, show only them, or leave them out."}),
                "selection": (
                    ["index", "random"],
                    {"default": "index", "tooltip": "index = take result_index; random = pick between range_from and range_to."},
                ),
                "result_index": ("INT", {"default": 0, "min": 0, "max": 999, "tooltip": "Used when selection = index."}),
                "range_from": ("INT", {"default": 0, "min": 0, "max": 999, "tooltip": "Random range start (0 = first result)."}),
                "range_to": ("INT", {"default": 9, "min": 0, "max": 999, "tooltip": "Random range end, inclusive — first 10 results by default."}),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Drives the random pick: randomize for a different item per run, fix it to reproduce one.",
                    },
                ),
            },
            "optional": {
                "folder": (FOLDER_TYPE, {"tooltip": "Optional: also save the stock item to this Magnific project/folder."}),
            },
        }, _MAG_STOCK_SEARCH)

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    OUTPUT_TOOLTIPS = ("The chosen stock image.", "Every usable result found, the picked one marked >, and where it was saved.")
    FUNCTION = "search"
    CATEGORY = CATEGORY

    PER_PAGE = 20

    def search(
        self,
        query: str,
        content_type: str,
        license: str,
        orientation: str,
        ai_generated: str,
        selection: str,
        result_index: int,
        range_from: int,
        range_to: int,
        seed: int,
        folder: Optional[str] = None,
    ):
        update_check.assert_not_blocked()
        if not query.strip():
            raise McpError("Search query is empty")
        filters = {
            "content_type": None if content_type == "any" else content_type,
            "license": None if license == "any" else license,
            "orientation": None if orientation == "any" else orientation,
            "ai_generated": None if ai_generated == "any" else ai_generated,
        }
        target_index = _target_index(selection, result_index, range_from, range_to, seed)
        # The index counts image-capable hits globally, but server pages can
        # interleave non-image results (video/icon, especially with 'any'), so
        # walk pages accumulating usable items instead of assuming page/offset
        # alignment. Bounded so a sparse catalog can't turn into a crawl.
        max_pages = target_index // self.PER_PAGE + 4
        collected: List[dict] = []
        page = 1
        while True:
            response = api.stock_search(query, filters, page=page, per_page=self.PER_PAGE)
            raw_items = [item for item in response.get("items", []) if isinstance(item, dict)]
            collected.extend(item for item in raw_items if item.get("type") in STOCK_IMAGE_TYPES)
            pagination = response.get("pagination") if isinstance(response.get("pagination"), dict) else {}
            last_page = int(pagination.get("lastPage") or page)
            if len(collected) > target_index or not raw_items or page >= last_page or page >= max_pages:
                break
            page += 1
        if not collected:
            raise McpError(f"No usable stock image results for '{query}'")
        chosen = collected[min(target_index, len(collected) - 1)]

        download = api.stock_download(int(chosen["id"]), chosen.get("type"), folder)
        url = download.get("downloadUrl")
        if not url:
            raise McpError("stock_download returned no download URL")

        lines = [
            f"{'>' if item is chosen else ' '} [{index}] {item.get('title', '?')}"
            f" ({item.get('type')}{', premium' if item.get('premium') else ''})"
            for index, item in enumerate(collected)
        ]
        if selection == "random":
            low, high = sorted((range_from, range_to))
            lines.append(f"random pick [{target_index}] from range {low}–{high} (seed {seed})")
        if target_index >= len(collected):
            lines.append(f"only {len(collected)} image results found — returned the last one")
        if download.get("creationIdentifier"):
            lines.append(f"saved to library as {download['creationIdentifier']}")
        return (tensors.urls_to_batch([url]), "\n".join(lines))


def _await_single(ref: dict) -> str:
    """Wait for one queued creation and return its result URL."""
    entries = api.wait_for_creations([ref["identifier"]])
    urls = api.result_urls(entries)
    if not urls:
        raise McpError("Generation finished but returned no result URL")
    return urls[0]


def _upload_image_tensor(image) -> str:
    data, mime_type = tensors.image_upload_payload(image)
    return api.upload_media(data, mime_type)


def _upload_video(video) -> str:
    """A VIDEO input as a creation identifier. Shared by every node taking one,
    so the container normalization (any input container → MP4) can't diverge
    between them."""
    return api.upload_media(tensors.video_to_mp4_bytes(video), "video/mp4")


VIDEO_ASPECT_RATIO_FALLBACK = ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"]


class MagnificGenerateVideo:
    """web/magnific.js narrows duration/resolution/aspect_ratio to the selected
    model's capabilities (via /magnific/video-models); the combos declare the
    union across models so any narrowed pick passes combo validation."""

    @classmethod
    def INPUT_TYPES(cls):
        durations: List[int] = []
        resolutions: List[str] = []
        ratios: List[str] = []
        try:
            catalog = api.video_models()
            models = ["auto"] + [m["slug"] for m in catalog if m["slug"] != "auto"]
            durations = sorted({d for m in catalog for d in m["durations"]})
            resolutions = sorted({r for m in catalog for r in m["resolutions"]})
            ratios = sorted({r for m in catalog for r in m["aspectRatios"] if r != "auto"})
        except Exception:
            models = ["auto"]  # 'auto' always works; the catalog loads on the next refresh
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "What happens in the video."}),
                "model": (models, {"default": "auto", "tooltip": "Video model; auto lets Magnific choose. The list comes from your account (press R to refresh)."}),
                "duration": (
                    [str(d) for d in durations] or ["5", "10"],
                    {"default": "5" if 5 in durations or not durations else str(durations[0]), "tooltip": "Seconds; narrows to the selected model."},
                ),
                "resolution": (
                    ["auto"] + resolutions,
                    {"default": "auto", "tooltip": "auto = model default; narrows to the selected model."},
                ),
                "aspect_ratio": (["auto"] + (ratios or VIDEO_ASPECT_RATIO_FALLBACK), {"default": "auto", "tooltip": "Shape of the video; auto = the model's default (or the start frame's shape)."}),
                "sound_effects": ("BOOLEAN", {"default": False, "tooltip": "Only models with native audio honor it."}),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Sent to Magnific. Same seed and settings reproduce the result on models that honor it (e.g. Seedance).",
                    },
                ),
            },
            "optional": {
                # First optional key: it was the first required socket, so this
                # keeps the socket order and saved link slots intact.
                "folder": (FOLDER_TYPE, {"tooltip": FOLDER_TOOLTIP}),
                "start_frame": ("IMAGE", {"tooltip": "Image-to-video start keyframe."}),
                "end_frame": ("IMAGE", {"tooltip": "End keyframe (models that support it)."}),
                "reference_image": ("IMAGE", {"tooltip": "A picture of what should appear (models that take image references)."}),
                # Appended last: an input socket added above an existing one
                # would shift the saved link slots of existing workflows.
                "reference_video": (
                    "VIDEO",
                    {"tooltip": "Video reference (models that support it, e.g. Seedance 2.0). Not combinable with a start/end frame."},
                ),
                # Same rule: last, so existing saved link slots keep their index.
                "library_references": (
                    LIBRARY_REFS_TYPE,
                    {"tooltip": LIBRARY_TOOLTIP + " Video takes characters, elements and styles (no locations)."},
                ),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "creation_identifier", "metadata")
    OUTPUT_TOOLTIPS = ("The generated clip.", "The Magnific creation id of the result (find it in your library, or pass it on).", "Model, prompt, seed, tool and url of the result as JSON - split it with the Metadata (unpack) node.")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(
        self,
        prompt: str,
        model: str,
        duration: str,
        resolution: str,
        aspect_ratio: str,
        sound_effects: bool,
        seed: int,
        folder: Optional[str] = None,
        start_frame=None,
        end_frame=None,
        reference_image=None,
        reference_video=None,
        library_references: Optional[List[dict]] = None,
    ):
        update_check.assert_not_blocked()
        if not prompt.strip():
            raise McpError("Prompt is empty")
        library = _checked_library_refs(
            library_references, model, _video_models_quiet(), api.VIDEO_LIBRARY_TYPES, "Video generation"
        )
        # A video reference is prohibitedWith the start/end keyframes server-side,
        # so the pair 422s the whole request after both uploads have already been
        # paid for — refuse it before the first byte goes up.
        if reference_video is not None and (start_frame is not None or end_frame is not None):
            raise McpError(
                "A reference video cannot be combined with a start or end frame — the models that "
                "take a video reference drive the motion from that clip instead. Disconnect the "
                "start_frame/end_frame input, or remove the reference_video one."
            )
        references: List[dict] = []
        if reference_image is not None:
            references.append({"type": "image", "identifier": _upload_image_tensor(reference_image)})
        if reference_video is not None:
            references.append({"type": "video", "identifier": _upload_video(reference_video)})
        args = api.generate_video_args(
            prompt=api.with_library_mentions(prompt, library),
            slug=model,
            duration=int(duration),
            aspect_ratio=aspect_ratio,
            resolution=None if resolution == "auto" else resolution,
            sound_effects=sound_effects,
            start_identifier=_upload_image_tensor(start_frame) if start_frame is not None else None,
            end_identifier=_upload_image_tensor(end_frame) if end_frame is not None else None,
            references=references or None,
            folder_reference=folder,
            library_references=api.library_video_references(library) or None,
            seed=seed,
        )
        refs = api.generate_video(args)
        url = _await_single(refs[0])
        identifier = refs[0]["identifier"]
        video = tensors.video_bytes_to_comfy(api.download_bytes(url), f"magnific-video-{identifier}.mp4")
        return (video, identifier, creation_metadata.for_identifiers([identifier]))


class MagnificUpscaleVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return with_layout({
            "required": {
                "video": ("VIDEO", {"tooltip": "The clip to upscale (uploaded as MP4)."}),
                "mode": (["magnific", "magnific_precision", "topaz"], {"default": "magnific", "tooltip": "magnific = creative upscale (preset / custom sliders); magnific_precision = faithful (strength); topaz = scale by upscale_factor."}),
                # Magnific/Precision target an output width; Topaz scales by factor.
                "target_resolution": (["1280", "1920", "2560", "3840"], {"default": "1920", "tooltip": "Output width (Magnific/Precision modes)."}),
                "upscale_factor": (["2", "4"], {"default": "2", "tooltip": "Scale multiplier (Topaz mode)."}),
                "preset": (["realistic", "animationAnd3D", "artistic", "custom"], {"default": "realistic", "tooltip": "Creative bundle for the magnific mode; custom uses the sliders below."}),
                "fps_boost": ("BOOLEAN", {"default": False, "tooltip": "magnific / magnific_precision modes: ask the server to raise the frame rate as well (not sent in topaz mode)."}),
                "preview": ("BOOLEAN", {"default": False, "tooltip": "12-frame preview run instead of the full clip."}),
            },
            "optional": {
                # First optional key: preserves the socket order (video, folder)
                # this input had while required, so saved link slots don't shift.
                "folder": (FOLDER_TYPE, {"tooltip": FOLDER_TOOLTIP}),
                # Custom-preset creative fields (magnific mode) / strength (precision).
                "creativity": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "magnific mode, custom preset: how much new detail is invented."}),
                "sharpen": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "Sharpening (video: magnific mode, custom preset only)."}),
                "smart_grain": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "Added grain that follows the image (video: magnific mode, custom preset only)."}),
                "flavor": (["natural", "vivid"], {"default": "natural", "tooltip": "magnific mode, custom preset: colour rendering."}),
                "premium_quality": ("BOOLEAN", {"default": True, "tooltip": "magnific mode, custom preset: the slower, higher-quality pass."}),
                "turbo": ("BOOLEAN", {"default": False, "tooltip": "magnific mode, custom preset: faster processing."}),
                "strength": ("INT", {"default": 50, "min": 0, "max": 100, "tooltip": "Precision mode only."}),
            },
        }, _MAG_UPSCALE_VIDEO)

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "creation_identifier", "metadata")
    OUTPUT_TOOLTIPS = ("The upscaled clip.", "The Magnific creation id of the result (find it in your library, or pass it on).", "Model, prompt, seed, tool and url of the result as JSON - split it with the Metadata (unpack) node.")
    FUNCTION = "upscale"
    CATEGORY = CATEGORY

    # Fixed creative bundles, mirroring the panel's MAGNIFIC_PRESET_VALUES.
    PRESETS = {
        "realistic": {"creativity": 0, "premiumQuality": True, "sharpen": 0, "smartGrain": 0, "flavor": "natural", "turbo": False},
        "animationAnd3D": {"creativity": 0, "premiumQuality": True, "sharpen": 0, "smartGrain": 0, "flavor": "vivid", "turbo": False},
        "artistic": {"creativity": 50, "premiumQuality": True, "sharpen": 0, "smartGrain": 0, "flavor": "vivid", "turbo": False},
    }

    def upscale(
        self,
        video,
        mode: str,
        target_resolution: str,
        upscale_factor: str,
        preset: str,
        fps_boost: bool,
        preview: bool,
        folder: Optional[str] = None,
        creativity: int = 0,
        sharpen: int = 0,
        smart_grain: int = 0,
        flavor: str = "natural",
        premium_quality: bool = True,
        turbo: bool = False,
        strength: int = 50,
    ):
        update_check.assert_not_blocked()
        identifier = _upload_video(video)
        creative = self.PRESETS.get(preset) or {
            "creativity": creativity,
            "premiumQuality": premium_quality,
            "sharpen": sharpen,
            "smartGrain": smart_grain,
            "flavor": flavor,
            "turbo": turbo,
        }
        options = {
            "targetResolution": int(target_resolution),
            "upscaleFactor": int(upscale_factor),
            "fpsBoost": fps_boost,
            "preview": preview,
            "folderReference": folder,
            **(creative if mode == "magnific" else {}),
            **({"strength": strength} if mode == "magnific_precision" else {}),
        }
        ref = api.upscale_video(api.upscale_video_args(identifier, mode, options))
        url = _await_single(ref)
        video_out = tensors.video_bytes_to_comfy(api.download_bytes(url), f"magnific-upscale-{ref['identifier']}.mp4")
        return (video_out, ref["identifier"], creation_metadata.for_identifiers([ref["identifier"]]))


class MagnificGenerateMusic:
    @staticmethod
    def _models() -> List[str]:
        try:
            slugs = api.music_model_choices()
        except Exception:
            slugs = list(api.MUSIC_MODELS)  # schema enum loads on the next refresh
        labels = [api.music_model_label(slug) for slug in slugs]
        # Raw slugs stay valid (hidden) choices so pre-label saved workflows
        # pass ComfyUI's queue-time combo validation.
        return labels + [slug for slug in slugs if slug not in labels]

    @classmethod
    def INPUT_TYPES(cls):
        return with_layout({
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Describe the music: genre, mood, instruments, lyrics."}),
                "model": (cls._models(), {"default": api.music_model_label("google-lyria"), "tooltip": "Music model."}),
                "duration_seconds": ("INT", {"default": 30, "min": 10, "max": 300, "tooltip": "Only Lyria 3 Long and ElevenLabs honor it; the rest are fixed at 30s."}),
                "instrumental": ("BOOLEAN", {"default": False, "tooltip": "On: no vocals."}),
                # Not sent: audio_music_generate takes no seed, so this only
                # breaks ComfyUI's result cache to force a fresh run.
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Music generation takes no seed: changing it only forces a new run.",
                    },
                ),
            },
            "optional": {
                "folder": (FOLDER_TYPE, {"tooltip": FOLDER_TOOLTIP}),
            },
        }, _MAG_GENERATE_MUSIC)

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "creation_identifier", "metadata")
    OUTPUT_TOOLTIPS = ("The generated music.", "The Magnific creation id of the result (find it in your library, or pass it on).", "Model, prompt, seed, tool and url of the result as JSON - split it with the Metadata (unpack) node.")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, prompt: str, model: str, duration_seconds: int, instrumental: bool, seed: int, folder: Optional[str] = None):
        update_check.assert_not_blocked()
        if not prompt.strip():
            raise McpError("Prompt is empty")
        tensors.assert_audio_decodable()  # before the generation spends credits
        ref = api.generate_music(api.music_args(prompt, api.music_model_slug(model), duration_seconds, instrumental, folder))
        url = _await_single(ref)
        return (tensors.audio_bytes_to_comfy(api.download_bytes(url)), ref["identifier"], creation_metadata.for_identifiers([ref["identifier"]]))


class MagnificVoiceover:
    @classmethod
    def INPUT_TYPES(cls):
        try:
            # An empty catalog while signed in is NOT a sign-in problem.
            voices = [choice["label"] for choice in api.voice_choices()] or [NO_VOICES_SENTINEL]
        except NotSignedInError:
            voices = [SIGN_IN_SENTINEL]
        except Exception:
            voices = [LOAD_FAILED_SENTINEL]
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "tooltip": "The words to speak."}),
                "voice": (voices, {"tooltip": "Voice from your Magnific account (press R to refresh)."}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.7, "max": 1.2, "step": 0.05, "tooltip": "Speaking speed; 1 = normal."}),
            },
            "optional": {
                "folder": (FOLDER_TYPE, {"tooltip": FOLDER_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "creation_identifier", "metadata")
    OUTPUT_TOOLTIPS = ("The spoken audio.", "The Magnific creation id of the result (find it in your library, or pass it on).", "Model, prompt, seed, tool and url of the result as JSON - split it with the Metadata (unpack) node.")
    FUNCTION = "speak"
    CATEGORY = CATEGORY

    def speak(self, text: str, voice: str, speed: float, folder: Optional[str] = None):
        update_check.assert_not_blocked()
        if not text.strip():
            raise McpError("Text is empty")
        if voice == SIGN_IN_SENTINEL:
            raise NotSignedInError()
        if voice in (LOAD_FAILED_SENTINEL, NO_VOICES_SENTINEL):
            raise McpError("The voice catalog is unavailable or empty — press R to refresh and retry")
        voice_id = api.voice_id_for(voice)
        if voice_id is None:
            raise McpError(f"Unknown voice '{voice}' — press R to refresh the node definitions")
        # Last check before dispatch: everything above fails without spending
        # credits anyway, so auth/catalog errors keep priority over this one.
        tensors.assert_audio_decodable()
        ref = api.generate_tts(api.tts_args(text, voice_id, speed, folder))
        url = _await_single(ref)
        return (tensors.audio_bytes_to_comfy(api.download_bytes(url)), ref["identifier"], creation_metadata.for_identifiers([ref["identifier"]]))


class MagnificSkinEnhancer:
    @classmethod
    def INPUT_TYPES(cls):
        return with_layout({
            "required": {
                "image": ("IMAGE", {"tooltip": "A portrait to enhance (uploaded to Magnific)."}),
                "version": (["faithful", "creative", "flexible"], {"default": "faithful", "tooltip": "faithful = keeps the face as it is (skin_detail applies); creative = freer; flexible = aim set by optimized_for."}),
                "sharpen": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "Sharpening (video: magnific mode, custom preset only)."}),
                "smart_grain": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "Added grain that follows the image (video: magnific mode, custom preset only)."}),
                "skin_detail": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "Faithful version only."}),
                "optimized_for": (
                    ["default", "enhance_skin", "enhance_everything", "improve_lighting", "transform_to_real", "no_make_up"],
                    {"default": "default", "tooltip": "Flexible version only."},
                ),
            },
            "optional": {
                "folder": (FOLDER_TYPE, {"tooltip": FOLDER_TOOLTIP}),
            },
        }, _MAG_SKIN)

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "creation_identifier", "metadata")
    OUTPUT_TOOLTIPS = ("The enhanced picture.", "The Magnific creation id of the result (find it in your library, or pass it on).", "Model, prompt, seed, tool and url of the result as JSON - split it with the Metadata (unpack) node.")
    FUNCTION = "enhance"
    CATEGORY = CATEGORY

    def enhance(self, image, version: str, sharpen: int, smart_grain: int, skin_detail: int, optimized_for: str, folder: Optional[str] = None):
        update_check.assert_not_blocked()
        ref = api.skin_enhance(
            _upload_image_tensor(image),
            folder,
            {
                "version": version,
                "sharpen": sharpen,
                "smartGrain": smart_grain,
                "skinDetail": skin_detail,
                "optimizedFor": None if optimized_for == "default" else optimized_for,
            },
        )
        url = _await_single(ref)
        return (tensors.urls_to_batch([url]), ref["identifier"], creation_metadata.for_identifiers([ref["identifier"]]))


class MagnificRemoveBackground:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The picture to cut out (uploaded to Magnific)."}),
            },
            "optional": {
                "folder": (FOLDER_TYPE, {"tooltip": FOLDER_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "creation_identifier", "metadata")
    OUTPUT_TOOLTIPS = ("The cut-out picture (RGB).", "The subject's alpha: 1 = kept, 0 = removed background.", "The Magnific creation id of the result (find it in your library, or pass it on).", "Model, prompt, seed, tool and url of the result as JSON - split it with the Metadata (unpack) node.")
    FUNCTION = "remove"
    CATEGORY = CATEGORY

    def remove(self, image, folder: Optional[str] = None):
        update_check.assert_not_blocked()
        ref = api.remove_background(_upload_image_tensor(image), folder)
        url = _await_single(ref)
        rgb, mask = tensors.url_to_image_and_mask(url)
        return (rgb, mask, ref["identifier"], creation_metadata.for_identifiers([ref["identifier"]]))


class MagnificRetouch:
    """Masked region edit (inpainting): replace or erase only the MASK area.
    Any MASK source works — the built-in MaskEditor, a Load Image alpha, or
    the Remove Background node's cutout mask."""

    @classmethod
    def INPUT_TYPES(cls):
        return with_layout({
            "required": {
                "image": ("IMAGE", {"tooltip": "The picture to edit; scaled down to the retouch size limit when larger (the console says so)."}),
                "mask": ("MASK", {"tooltip": "1.0/white = area to change. Must be the same pixel size as the image."}),
                "mode": (["replace", "erase"], {"default": "replace", "tooltip": "replace = paint what the prompt says into the mask; erase = remove what is there."}),
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Replace mode only: what the masked area should become."}),
            },
            "optional": {
                "folder": (FOLDER_TYPE, {"tooltip": FOLDER_TOOLTIP}),
            },
        }, _MAG_RETOUCH)

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "creation_identifier", "metadata")
    OUTPUT_TOOLTIPS = ("The retouched picture (at the size it was sent).", "The Magnific creation id of the result (find it in your library, or pass it on).", "Model, prompt, seed, tool and url of the result as JSON - split it with the Metadata (unpack) node.")
    FUNCTION = "retouch"
    CATEGORY = CATEGORY

    def retouch(self, image, mask, mode: str, prompt: str, folder: Optional[str] = None):
        update_check.assert_not_blocked()
        frame = image[0] if image.dim() == 4 else image
        # tensors.mask_frame peels [B,1,H,W] / [B,H,W] down to [H,W] — checking
        # shape[-2:] on the raw tensor would let a 4D mask through to a cryptic
        # PIL failure instead of this message.
        selected = tensors.mask_frame(mask)
        # The server rejects a misaligned pair rather than guessing an
        # alignment — surface that before spending an upload.
        if tuple(selected.shape) != tuple(frame.shape[:2]):
            raise ValueError(
                f"Mask is {selected.shape[1]}x{selected.shape[0]} but the image is "
                f"{frame.shape[1]}x{frame.shape[0]} — connect a mask painted on this image"
            )
        if not tensors.mask_has_selection(mask):
            raise ValueError(
                "The mask selects nothing — paint the area to retouch first, at full strength: "
                "the mask is flattened to pure black and white, so anything below 50% counts as "
                "unselected"
            )
        if mode == "replace" and not prompt.strip():
            raise ValueError("Replace mode needs a prompt describing what the masked area should become")
        fitted_image, fitted_mask = tensors.fit_for_retouch(image, mask)
        if fitted_image is not image:
            fitted = fitted_image[0]
            # Announced rather than silent: the result comes back at the scaled
            # size, and a user comparing it to the input would otherwise wonder.
            print(  # console-ok
                f"[Magnific] Retouch source resized {frame.shape[1]}x{frame.shape[0]} -> "
                f"{fitted.shape[1]}x{fitted.shape[0]} (retouch takes at most "
                f"{config.RETOUCH_MAX_EDGE}px, on a multiple of 8)"
            )
            # Scaling can drop a thin selection below the binary threshold, and
            # an all-black mask is a paid failure server-side.
            if not tensors.mask_has_selection(fitted_mask):
                raise ValueError(
                    f"The masked area is too thin to survive scaling to "
                    f"{fitted.shape[1]}x{fitted.shape[0]} — paint a larger area, or scale the "
                    "image down yourself before this node"
                )
        ref = api.retouch(
            _upload_image_tensor(fitted_image),
            api.upload_media(tensors.mask_to_png_bytes(fitted_mask)),
            mode,
            prompt.strip() or None,
            folder,
        )
        url = _await_single(ref)
        return (tensors.urls_to_batch([url]), ref["identifier"], creation_metadata.for_identifiers([ref["identifier"]]))


class MagnificStockPicker:
    """Visual stock search: the Search button (web/magnific.js) fills a
    thumbnail grid; clicking one stores its id in `selected`, which execution
    downloads as the IMAGE output."""

    @classmethod
    def INPUT_TYPES(cls):
        return with_layout({
            "required": {
                "query": ("STRING", {"default": "", "tooltip": "What to search the Magnific stock library for."}),
                "content_type": (STOCK_CONTENT_TYPES, {"default": "photo", "tooltip": "Kind of stock item; vectors, PSDs and designs come back as JPG renders."}),
                "license": (["any", "free", "premium"], {"default": "any", "tooltip": "Only free items, only premium items, or both."}),
                "orientation": (["any", "portrait", "landscape", "square", "panoramic"], {"default": "any", "tooltip": "Shape filter."}),
                "ai_generated": (["any", "only", "excluded"], {"default": "any", "tooltip": "Include AI-generated items, show only them, or leave them out."}),
                "selected": ("STRING", {"default": "", "tooltip": "Set by clicking a thumbnail in the results grid."}),
            },
            "optional": {
                "folder": (FOLDER_TYPE, {"tooltip": "Optional: also save the stock item to this Magnific project/folder."}),
            },
        }, _MAG_PICKER_SELECTED)

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    OUTPUT_TOOLTIPS = ("The stock image you clicked.", "Its stock id and type, and where it was saved.")
    FUNCTION = "pick"
    CATEGORY = CATEGORY

    def pick(self, query, content_type, license, orientation, ai_generated, selected, folder=None):
        update_check.assert_not_blocked()
        if not selected.strip():
            raise McpError("Nothing selected — press Search on the node and click a thumbnail")
        item_id, _, item_type = selected.partition("|")
        if not item_id.isdigit():
            raise McpError(f"Invalid selection '{selected}' — click a thumbnail in the grid")
        download = api.stock_download(int(item_id), item_type or None, folder)
        url = download.get("downloadUrl")
        if not url:
            raise McpError("stock_download returned no download URL")
        info = f"stock item {item_id} ({item_type or 'photo'})"
        if download.get("creationIdentifier"):
            info += f" — saved to library as {download['creationIdentifier']}"
        return (tensors.urls_to_batch([url]), info)


class MagnificCreationPicker:
    """Visual browser of a Magnific project/folder: Browse fills a thumbnail
    grid of its image creations; the clicked one becomes the IMAGE output."""

    @classmethod
    def INPUT_TYPES(cls):
        # Same project/folder combos as Save To (the JS narrows the folder
        # combo per selected project via the shared helper), plus the media
        # type filter and the grid's selection slot.
        base = MagnificSaveTo.INPUT_TYPES()
        base["required"]["content_type"] = (
            ["image", "video", "audio"],
            {"default": "image", "tooltip": "Media type to browse; only the matching output is populated."},
        )
        base["required"]["selected"] = (
            "STRING",
            {"default": "", "tooltip": "Set by clicking a thumbnail in the grid."},
        )
        return with_layout(base, _MAG_PICKER_SELECTED)

    RETURN_TYPES = ("IMAGE", "VIDEO", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("image", "video", "audio", "creation_identifier", "metadata")
    OUTPUT_TOOLTIPS = ("The picked creation when content_type = image (empty otherwise).", "The picked creation when content_type = video.", "The picked creation when content_type = audio.", "Its creation id.", "Model, prompt, seed, tool and url of the result as JSON - split it with the Metadata (unpack) node.")
    FUNCTION = "pick"
    CATEGORY = CATEGORY

    def pick(self, project: str, folder: str, content_type: str, selected: str):
        update_check.assert_not_blocked()
        if project == SIGN_IN_SENTINEL:
            raise NotSignedInError()
        if project == NO_PROJECTS_SENTINEL:
            raise McpError("Your Magnific library has no projects — create one at magnific.com, then press R to refresh")
        if project == LOAD_FAILED_SENTINEL:
            raise McpError("Could not load your projects — press R to refresh the node definitions and retry")
        if not selected.strip():
            raise McpError("Nothing selected — press Browse on the node and click a thumbnail")
        identifier = selected.strip()
        media = api.creation_media(identifier)
        url = media.get("url") or media.get("previewUrl")
        if not url:
            raise McpError(f"The selected creation has no downloadable media (status: {media.get('status', '?')})")
        # Only the output matching content_type is populated; wiring another
        # one downstream fails there, which is the honest signal.
        info = creation_metadata.for_identifiers([identifier])
        if content_type == "video":
            return (None, tensors.video_bytes_to_comfy(api.download_bytes(url), f"magnific-pick-{identifier}.mp4"), None, identifier, info)
        if content_type == "audio":
            return (None, None, tensors.audio_bytes_to_comfy(api.download_bytes(url)), identifier, info)
        return (tensors.urls_to_batch([url]), None, None, identifier, info)


class MagnificMetadata:
    """Splits a `metadata` socket into plain sockets — the model for a Save
    Image filename prefix, the prompt for a text node, the seed to replay a
    result. The generation nodes emit one JSON string instead of a socket per
    field so a new field never shifts their outputs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "metadata": ("STRING", {"forceInput": True, "tooltip": "The metadata output of a Magnific generation or edit node."}),
                "index": ("INT", {"default": 0, "min": 0, "max": 63, "tooltip": "Which result of a batch to read; clamped to the last one."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("model", "prompt", "seed", "creation_identifier", "tool", "url", "json")
    OUTPUT_TOOLTIPS = ("The model that made the result.", "The prompt it was made from.", "Its seed (to replay it).", "Its creation id.", "The Magnific tool that ran.", "Where the result can be downloaded.", "The whole entry as JSON.")
    FUNCTION = "unpack"
    CATEGORY = CATEGORY

    def unpack(self, metadata: str, index: int):
        entry = creation_metadata.unpack(metadata, index)
        return (
            creation_metadata.unpack_text(entry, "model"),
            creation_metadata.unpack_text(entry, "prompt"),
            creation_metadata.unpack_int(entry, "seed"),
            creation_metadata.unpack_text(entry, "creation_identifier"),
            creation_metadata.unpack_text(entry, "tool"),
            creation_metadata.unpack_text(entry, "url"),
            creation_metadata.unpack_json(entry),
        )


NODE_CLASS_MAPPINGS = {
    "MagnificSaveTo": MagnificSaveTo,
    "MagnificGenerateImage": MagnificGenerateImage,
    "MagnificUpscaleImage": MagnificUpscaleImage,
    "MagnificStockSearch": MagnificStockSearch,
    "MagnificGenerateVideo": MagnificGenerateVideo,
    "MagnificUpscaleVideo": MagnificUpscaleVideo,
    "MagnificGenerateMusic": MagnificGenerateMusic,
    "MagnificVoiceover": MagnificVoiceover,
    "MagnificSkinEnhancer": MagnificSkinEnhancer,
    "MagnificRemoveBackground": MagnificRemoveBackground,
    "MagnificRetouch": MagnificRetouch,
    "MagnificStockPicker": MagnificStockPicker,
    "MagnificCreationPicker": MagnificCreationPicker,
    "MagnificLibraryReference": MagnificLibraryReference,
    "MagnificMetadata": MagnificMetadata,
}

# Presented as OmniScale (owner D0.14); the class ids stay Magnific* so saved workflows load.
NODE_DISPLAY_NAME_MAPPINGS = {
    "MagnificSaveTo": "OmniScale Save To (project/folder)",
    "MagnificGenerateImage": "OmniScale Generate Image",
    "MagnificUpscaleImage": "OmniScale Upscale Image",
    "MagnificStockSearch": "OmniScale Stock Search",
    "MagnificGenerateVideo": "OmniScale Generate Video",
    "MagnificUpscaleVideo": "OmniScale Upscale Video",
    "MagnificGenerateMusic": "OmniScale Generate Music",
    "MagnificVoiceover": "OmniScale Voiceover (TTS)",
    "MagnificSkinEnhancer": "OmniScale Skin Enhancer",
    "MagnificRemoveBackground": "OmniScale Remove Background",
    "MagnificRetouch": "OmniScale Retouch (inpaint)",
    "MagnificStockPicker": "OmniScale Stock Picker",
    "MagnificCreationPicker": "OmniScale Creation Picker",
    "MagnificLibraryReference": "OmniScale Library Reference",
    "MagnificMetadata": "OmniScale Metadata (unpack)",
}

# One-sentence node descriptions (shown in the node library and the node's help). Set here, in one table, so the
# classes above stay as they are; every sentence states what the node does in this file, nothing more.
_DESCRIPTIONS = {
    MagnificSaveTo: "Pick the project (and optionally a folder) your new creations are stored in; leave it "
                    "unconnected and results go to your Personal project.",
    MagnificLibraryReference: "Pick a Library asset (character, style, element or location) from your own or the "
                              "public catalog to guide a generation; chain several through the chain socket.",
    MagnificGenerateImage: "Generate an image from a prompt on your signed-in account; resolution choices follow "
                           "the chosen model.",
    MagnificUpscaleImage: "Upscale an image 2x to 16x: creative invents new detail, precision stays faithful to "
                          "the input.",
    MagnificStockSearch: "Search stock images and return one result, picked by index or at random; the info "
                         "output lists what else matched.",
    MagnificGenerateVideo: "Generate a video from a prompt (and optional start image); duration, resolution and "
                           "aspect ratio follow the chosen model.",
    MagnificUpscaleVideo: "Upscale a video clip (creative, precision or topaz mode), optionally raising the frame "
                          "rate; preview runs only 12 frames.",
    MagnificGenerateMusic: "Generate music from a description (genre, mood, instruments, lyrics), with or without "
                           "vocals.",
    MagnificVoiceover: "Speak the given text with a voice from your account (text to speech).",
    MagnificSkinEnhancer: "Add natural skin detail to a portrait: faithful keeps the face as it is, creative and "
                          "flexible change more.",
    MagnificRemoveBackground: "Cut the subject out of an image; returns the cutout and its MASK.",
    MagnificRetouch: "Edit only the masked area of an image: replace it with what the prompt says, or erase it.",
    MagnificStockPicker: "Search stock images and pick one from a thumbnail grid; the clicked image becomes the "
                         "IMAGE output.",
    MagnificCreationPicker: "Browse a project or folder and reuse one of your earlier creations (image, video or "
                            "audio, by content_type); the picked one fills the matching output.",
    MagnificMetadata: "Unpack a result's metadata into plain outputs: model, prompt, seed, creation id, tool and "
                      "url.",
}
for _cls, _text in _DESCRIPTIONS.items():
    if not getattr(_cls, "DESCRIPTION", None):
        _cls.DESCRIPTION = _text
