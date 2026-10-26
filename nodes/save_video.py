"""Save Video (C2C) — streaming video / image-sequence writer."""

from __future__ import annotations

import hashlib
from typing import Any

import torch

from .c2c_video.save_encode import (
    effective_naming,
    encode_video,
    pad_edge_replicate,
    resolve_output_path,
)
from .c2c_video import colour_cm as CM
from .c2c_video import source_meta as SMETA
from .c2c_video import save_matrix as SM
from .c2c_video import handles_io as HIO
from .c2c_video import save_naming as SN
from .c2c_video import save_encode as SE
from .c2c_video.save_formats import (
    SaveVideoError,
    available_formats,
)
from .c2c_video.save_ocio import (
    colorspace_choices,
    default_colorspace_in,
    default_colorspace_out_exr,
)


class SaveVideoC2C:
    """Write IMAGE batches to pro video formats with preview and scrub UI."""

    CATEGORY = "C2C/Video"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("filenames", "preview")
    FUNCTION = "execute"
    DESCRIPTION = (
        "Save an IMAGE batch as H.264/H.265/ProRes/DNxHR/FFV1/WebM/GIF/WebP "
        "or 16-bit PNG / EXR sequences. Streams chunk-by-chunk within the RAM "
        "budget, muxes audio, applies OCIO colour transforms, and shows a "
        "browser preview with timeline scrub after each run."
    )

    @classmethod
    def INPUT_TYPES(cls):
        spaces = colorspace_choices()
        default_in = default_colorspace_in()
        return {
            "required": {
                # L9.40: "custom" = the five controls below (container / codec / profile / bit depth / chroma);
                # a saved workflow keeps its old single-list label and encodes exactly as before
                "format": (([] if SM.disabled() else [SM.CUSTOM]) + available_formats(), {
                    "default": SM.CUSTOM if not SM.disabled() else available_formats()[0],
                    "tooltip": "custom: choose container, codec, profile, bit depth and chroma below. The other "
                               "entries are the old single list, kept so saved workflows encode as before."}),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 0.001, "max": 240.0, "step": 0.001,
                    "tooltip": "Frame rate for the written file. When video_info is "
                    "wired and fps is left at 24, loaded_fps from the loader is used.",
                }),
                "filename_prefix": ("STRING", {"default": "C2C/video"}),
                "quality": ("INT", {
                    "default": 80, "min": 0, "max": 100, "step": 1,
                    "tooltip": "Encoder quality where the format supports it "
                    "(CRF / qscale / ProRes profile). Hidden for lossless formats.",
                }),
            },
            "optional": {
                # first optional slot: saved workflows keep images on input 0 (it used to be the only required socket)
                "images": ("IMAGE", {"tooltip": "The frames to save. Or wire latent + vae instead to decode and save "
                                                "as one stream."}),
                "audio": ("AUDIO",),
                "video_info": ("VHS_VIDEOINFO",),
                # L9.53: "auto (from Load)" = the space Load Video (C2C) converted the frames to / the file's own
                # space, so Save reverses exactly what Load did (a new node starts on it; saved values stay)
                "colorspace_in": ([CM.FROM_LOAD] + list(spaces), {"default": default_in}),
                "colorspace_out": (["same as input", CM.FROM_LOAD] + list(spaces), {"default": "same as input"}),
                # "Folder Version" (saved workflows) is the same as Folder Version Incrementer (L9.40)
                "naming": (SN.NAMING_CHOICES + ["Folder Version"], {"default": "default", "tooltip":
                    "Folder Version Incrementer: the version folder and name come from the Folder Version "
                    "Incrementer node in this graph (no wire needed), or from the incrementer's defaults when there "
                    "is none; a wired subfolder still wins. ComfyUI counter: prefix_00001 like Save Image. default: "
                    "the C2C setting (Folder Version Incrementer unless you changed it)."}),
                "naming_resolved": ("STRING", {"default": "", "tooltip": "Set by the UI from c2c.saveVideo.naming."}),
                "subfolder": ("STRING", {"default": ""}),
                "save_output": ("BOOLEAN", {"default": True}),
                "latent": ("LATENT", {"tooltip": "Stream save: a video latent decoded straight into the file, in order, "
                                                 "keeping the VAE's causal state - the full frame batch is never held in "
                                                 "RAM. Wire this and vae instead of images."}),
                "vae": ("VAE", {"tooltip": "The VAE for latent (stream save)."}),
                "tile_mode": (["auto", "off", "manual"], {
                    "default": "auto",
                    "tooltip": "Stream save, Wan VAEs: auto tiles only when the frame would not fit the GPU (2K on 8 GB -> "
                               "512 px). Each tile is decoded over the whole clip in one causal pass and blended on disk, "
                               "so the clip never sits in RAM. off: no tiles. manual: tile_size."}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64,
                                      "tooltip": "Tile size in pixels when tile_mode is manual."}),
                # L2.53 (owner A14) - appended, so saved workflows keep their widget order
                "save_metadata": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Write '<name>.png' next to the saved file: its first frame with the workflow and prompt "
                               "inside - drag it into ComfyUI to rebuild the graph. The video itself never carries the "
                               "workflow."}),
                # L9.40 (owner 2026-10-23) - appended, so saved workflows keep their widget order
                "container": (list(SM.CONTAINERS), {"default": SM.DEFAULT_CHOICE.container,
                              "tooltip": "The file type. Used when format is custom."}),
                "codec": (list(SM.CODECS), {"default": SM.DEFAULT_CHOICE.codec,
                          "tooltip": "The video codec; the list follows the container."}),
                "profile": (SM.all_profiles(), {"default": SM.DEFAULT_CHOICE.profile,
                            "tooltip": "The codec's profile (ProRes 422 HQ, DNxHR HQX, H.265 Main 10, EXR ZIP ...)."}),
                "bit_depth": (list(SM.BIT_DEPTHS), {"default": SM.DEFAULT_CHOICE.bit_depth,
                              "tooltip": "Bits per channel the profile allows. EXR: 16 = half float, 32 = float."}),
                "chroma": (list(SM.CHROMAS), {"default": SM.DEFAULT_CHOICE.chroma,
                           "tooltip": "Chroma subsampling: 4:2:0 (delivery), 4:2:2 (edit), 4:4:4 / RGB (full)."}),
                "color_primaries": (list(SM.PRIMARIES), {"default": "auto", "tooltip":
                    "Colour primaries written into the file. auto: from colorspace_out (sRGB / Rec.709 named, "
                    "others untagged). For YUV codecs the conversion matrix follows (BT.2020 -> BT.2020 NCL)."}),
                "color_transfer": (list(SM.TRANSFERS), {"default": "auto", "tooltip":
                    "Transfer function written into the file (it labels; the pixels come from colorspace_out)."}),
                "color_range": (list(SM.RANGES), {"default": "auto", "tooltip":
                    "YUV codecs: limited (16-235, broadcast, the default) or full (0-255). Changes the conversion "
                    "and the tag together. RGB codecs are always full range."}),
                # L9.40: handles, export range, preview first - appended
                "handles_trim": (list(HIO.TRIM_MODES), {"default": "auto", "tooltip":
                    "auto: trim exactly the handles Load Video (C2C) added (read from video_info; fit's target "
                    "length wins). manual: trim_frames on trim_side. off: write every frame that arrives."}),
                "trim_frames": ("INT", {"default": 8, "min": 0, "max": 100000,
                                        "tooltip": "manual trim: frames to cut (side both: on each end)."}),
                "trim_side": (list(HIO.SIDES), {"default": "head", "tooltip": "manual trim: which end."}),
                "range_in": ("INT", {"default": 0, "min": 0, "max": 10_000_000, "tooltip":
                    "Export range, first frame (of the trimmed clip). Set it with the player's In handle. 0 / 0 = "
                    "the whole clip."}),
                "range_out": ("INT", {"default": 0, "min": 0, "max": 10_000_000, "tooltip":
                    "Export range, last frame (inclusive). 0 = to the end."}),
                "write_mode": (["write", "preview first"], {"default": "write", "tooltip":
                    "preview first: a run shows a quick preview of exactly what would be written (handles trimmed, "
                    "range applied) and writes nothing; the player's Write file button then writes it - only the "
                    "encode runs again, the graph above is cached."}),
                "write_token": ("STRING", {"default": "", "tooltip": "Set by the player's Write file button."}),
                # L9.54 (owner 2026-10-25: Format -> Subformat -> only that subformat's options) - appended
                "format_family": (list(SM.FAMILIES), {"default": "Video", "tooltip":
                    "Video, Image Sequence, or Custom (container -> codec -> profile -> bit depth -> chroma)."}),
                "subformat": ([SM.FROM_CONTROLS] + list(SM.SUBFORMATS), {"default": SM.FROM_CONTROLS, "tooltip":
                    "The output format. It fixes the container and codec; only its own options show. 'from the "
                    "controls' = the five controls exactly (saved workflows)."}),
                "encoder_preset": (list(SE.X264_PRESETS), {"default": "default", "tooltip":
                    "H.264 / H.265 speed vs size (NVENC and AV1 get the matching step). default = the encoder's own."}),
                "gop": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip":
                    "Keyframe interval in frames (0 = the encoder's own). 1 = every frame a keyframe (edit-friendly)."}),
                "bitrate_kbps": ("INT", {"default": 0, "min": 0, "max": 2_000_000, "tooltip":
                    "Average bitrate in kbit/s instead of constant quality (0 = constant quality from quality)."}),
                "channels": (["auto", "RGB", "RGBA"], {"default": "auto", "tooltip":
                    "auto: alpha when the frames have it. RGB: never alpha. RGBA: alpha always (opaque when the "
                    "frames have none) - where the format can store it."}),
                "png_compression": ("INT", {"default": -1, "min": -1, "max": 9, "tooltip":
                    "PNG zlib level 0 (fast, big) - 9 (slow, small); -1 = the writer's default. Lossless either way."}),
                # L9.55 (owner 2026-10-25): which upstream Load Video this Save follows - auto = through each node's
                # first image input; a node id pins one (the node's picker writes it). Appended: saved workflows = auto.
                "follow_load": ("STRING", {"default": HIO.FOLLOW_AUTO, "tooltip":
                    "Which Load Video this Save takes handles, frame rate, colour and name from. auto: the one reached "
                    "through each node's first image input; a node id (the picker on the node) pins that one. A wired "
                    "video_info always wins."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def IS_CHANGED(cls, format, fps, filename_prefix, quality, images=None,  # noqa: A002
                   audio=None, video_info=None, colorspace_in=None, colorspace_out=None,
                   naming="default", naming_resolved="", subfolder="", save_output=True, **kw):
        h = hashlib.md5()
        settings = {
            "format": format, "fps": fps, "filename_prefix": filename_prefix,
            "quality": quality, "colorspace_in": colorspace_in,
            "colorspace_out": colorspace_out, "naming": naming,
            "naming_resolved": naming_resolved, "subfolder": subfolder,
            "save_output": save_output,
        }
        settings.update({k: v for k, v in kw.items() if isinstance(v, (str, int, float, bool))})   # L9.40 widgets
        for k, v in sorted(settings.items()):
            h.update(f"{k}={v}".encode())
        if torch.is_tensor(images):
            h.update(images.cpu().numpy().tobytes())
        if isinstance(audio, dict) and torch.is_tensor(audio.get("waveform")):
            h.update(str(tuple(audio["waveform"].shape)).encode())
        if isinstance(video_info, dict):
            h.update(str(sorted(video_info.items())).encode())
        return h.hexdigest()

    def execute(
        self,
        format,  # noqa: A002
        fps,
        filename_prefix,
        quality=80,
        images=None,
        audio=None,
        video_info=None,
        colorspace_in=None,
        colorspace_out="same as input",
        naming="default",
        naming_resolved="",
        subfolder="",
        save_output=True,
        latent=None,
        vae=None,
        tile_mode="auto",
        tile_size=512,
        save_metadata=True,
        container=None,
        codec=None,
        profile=None,
        bit_depth=None,
        chroma=None,
        color_primaries="auto",
        color_transfer="auto",
        color_range="auto",
        handles_trim="auto",
        trim_frames=8,
        trim_side="head",
        range_in=0,
        range_out=0,
        write_mode="write",
        write_token="",
        format_family="Video",
        subformat=SM.FROM_CONTROLS,
        encoder_preset="default",
        gop=0,
        bitrate_kbps=0,
        channels="auto",
        png_compression=-1,
        follow_load=HIO.FOLLOW_AUTO,
        prompt=None,
        extra_pnginfo=None,
        unique_id=None,
    ):
        settings = {k: v for k, v in locals().items()
                    if k not in ("self", "images", "audio", "latent", "vae", "prompt", "extra_pnginfo", "unique_id",
                                 "write_token", "write_mode", "video_info", "range_in", "range_out")}
        stream_info = None
        if latent is not None or vae is not None:
            if images is not None:
                raise SaveVideoError("Wire images, OR latent + vae - not both.")
            if latent is None or vae is None:
                raise SaveVideoError("Stream save needs both latent and vae wired.")
            from .c2c_video.stream_decode import stream_decode, stream_decode_images
            z = latent.get("samples") if isinstance(latent, dict) else latent
            tile_px = 0
            if torch.is_tensor(z) and z.ndim == 4:      # L2.53: image latents, decoded in batches into the file
                tile_mode = "off"
            if tile_mode == "manual":
                tile_px = int(tile_size)
            elif tile_mode != "off":
                from .hdr_color_science import C2CVAEQualityDecode
                z = latent.get("samples") if isinstance(latent, dict) else None
                if torch.is_tensor(z):
                    tile_px = C2CVAEQualityDecode._auto_tile_px(vae, z, False)
            if torch.is_tensor(z) and z.ndim == 4:
                images, stream_info = stream_decode_images(vae, latent)
            else:
                images, stream_info = stream_decode(vae, latent, tile_px=tile_px)
            src = latent.get("c2c_source_frames") if isinstance(latent, dict) else None
            if isinstance(src, int) and 0 < src < images.shape[0]:
                images.shape = (src,) + tuple(images.shape[1:])        # the frames the clip really had
        if images is None:
            raise SaveVideoError("Connect an IMAGE batch, or latent + vae, to save.")
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.shape[0] == 0:
            raise SaveVideoError("IMAGE batch is empty — nothing to save.")

        d = SM.DEFAULT_CHOICE
        choice = SM.Choice(container or d.container, codec or d.codec, profile or d.profile,
                           str(bit_depth or d.bit_depth), chroma or d.chroma)
        if str(format) == SM.CUSTOM:                 # L9.54: a subformat fixes container + codec (API prompts too)
            choice = SM.choice_for_subformat(str(subformat or SM.FROM_CONTROLS), choice)
        extras = SE.Extras(encoder_preset=str(encoder_preset or "default"), gop=int(gop or 0),
                           bitrate_kbps=int(bitrate_kbps or 0), channels=str(channels or "auto"),
                           png_compression=int(-1 if png_compression is None else png_compression))
        colour = SM.Colour(color_primaries or "auto", color_transfer or "auto", color_range or "auto")
        SM.check_colour(colour)
        spec, format_how = SM.resolve(str(format), choice, alpha=images.shape[-1] >= 4)
        # L9.52 (owner 2026-10-25): no video_info wire -> follow images (or latent) back to the Load Video it came from
        # and use the video_info that Load produced (handles, fps). A wired video_info always wins.
        traced_note = None
        link: dict = {}                                  # L9.55: which Load this Save follows, for the node's report
        if video_info is None and isinstance(prompt, dict) and unique_id is not None:
            mine = (prompt.get(str(unique_id)) or {}).get("inputs") or {}
            which = ("images", "IMAGE") if HIO._is_link(mine.get("images")) else ("latent", "LATENT")
            video_info, traced_note = HIO.traced_video_info(prompt, unique_id, *which, follow=follow_load, link=link)
        link_report = None if not link else {k: link.get(k) for k in ("load", "candidates", "ambiguous", "chosen")}
        link_warn = [link["warning"]] if link.get("warning") else []
        eff_fps = float(fps)
        if video_info and abs(eff_fps - 24.0) < 1e-6:
            eff_fps = float(video_info.get("loaded_fps", eff_fps))

        cm_plan = None
        if CM.FROM_LOAD in (str(colorspace_in or ""), str(colorspace_out or "")):
            rec = video_info.get(CM.INFO_KEY) if isinstance(video_info, dict) else None
            cm_plan = CM.save_plan(rec, colorspace_in, colorspace_out, default_colorspace_in())
            cs_in, cs_out = cm_plan.cs_in, cm_plan.cs_out
        else:
            cs_in = colorspace_in or default_colorspace_in()
            cs_out = colorspace_out or "same as input"
            if cs_out == "same as input" and spec.codec.startswith("oiio_exr"):
                cs_out = default_colorspace_out_exr()
        exr_id = None
        if spec.codec.startswith("oiio_exr") and CM.config() is not None:
            exr_id = CM.interop_id(CM.config(), cs_in if cs_out in ("same as input", "same", "") else cs_out)

        pad_note = None
        even_w, even_h = spec.chroma_even                             # 4:2:0 both, 4:2:2 width: pad, never refuse
        if (even_w or even_h) and stream_info is not None:
            n0, h0, w0, c0 = images.shape
            pw, ph = (w0 % 2 if even_w else 0), (h0 % 2 if even_h else 0)
            if pw or ph:                                              # pad each chunk as it arrives
                images._transform = lambda chunk: pad_edge_replicate(chunk, even_w, even_h)[0]
                images.shape = (n0, h0 + ph, w0 + pw, c0)
                pad_note = f"{w0}x{h0} -> {w0 + pw}x{h0 + ph}"
        elif even_w or even_h:
            images, pad_note = pad_edge_replicate(images, even_w, even_h)

        # L9.40: handles off, then the export range (in the trimmed clip's frames), then preview-or-write
        images, audio, trim_rep = HIO.trim_on_save(images, audio, video_info, mode=str(handles_trim),
                                                   frames=int(trim_frames), side=str(trim_side), fps=eff_fps)
        if traced_note and str(handles_trim) == "auto":
            trim_rep["from"] = (trim_rep["from"].replace(" added by Load Video", "") + " - " + traced_note
                                if video_info else trim_rep["from"] + " (" + traced_note + ")")
        n_full = int(images.shape[0])                 # the trimmed clip: the player's timeline, In / Out live on it
        fingerprint = _fingerprint(settings, images if stream_info is None else None, latent)
        common = {"handles": trim_rep, "fingerprint": fingerprint, "write_mode": write_mode, "frames_total": n_full}
        if str(write_mode) == "preview first" and str(write_token or "") != fingerprint:
            from .c2c_video.save_encode import write_proxy
            proxy = write_proxy(images, eff_fps, "save_preview_first")
            last = images[-1:] if torch.is_tensor(images) else (images.last_frame if images.last_frame is not None
                                                                 else torch.zeros(1, 64, 64, 3))
            if stream_info is not None:
                images.close()
            n_out, ph, pw = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
            ui_prev = dict(common, mode="preview", preview=proxy, source=proxy, frames=n_out, fps=eff_fps,
                           range=None, range_in=int(range_in), range_out=int(range_out),
                           width=pw, height=ph, format=spec.label, format_from=format_how, codec=spec.codec,
                           warnings=link_warn + ([] if proxy else ["The clip is too long for a preview proxy."]),
                           link=link_report,
                           message="Preview only - nothing written. Press Write file in the player to write it.")
            return {"ui": {"c2c_save_video": [ui_prev]}, "result": ("", last)}
        source_proxy = None
        if stream_info is None and not _proxy_off():         # a streamed save's frames are read once: no proxy
            from .c2c_video.save_encode import write_proxy
            source_proxy = write_proxy(images, eff_fps, "save_source")
        images, audio, range_note, (r_in, r_out) = HIO.export_range(images, audio, int(range_in), int(range_out),
                                                                     eff_fps)
        common.update(range=range_note, range_in=r_in, range_out=r_out)
        # L9.55: the delivered frames back to the source - its frame numbers, timecode, pixel aspect, reel
        carry = None
        src_rec = video_info.get(SMETA.INFO_KEY) if isinstance(video_info, dict) else None
        if isinstance(src_rec, dict):
            hrec = video_info.get(HIO.INFO_KEY) if isinstance(video_info.get(HIO.INFO_KEY), dict) else {}
            trim_exact = (int(trim_rep.get("head", 0)), int(trim_rep.get("tail", 0))) == \
                (int(hrec.get("head", 0)), int(hrec.get("tail", 0)))
            carry = SMETA.delivered(src_rec, int(r_in), int(images.shape[0]), trim_exact)

        _, h, w, _ = images.shape
        naming_eff = effective_naming(str(naming), str(naming_resolved or ""))
        version = None
        # L9.55: "auto (from Load)" = the followed Load Video's name_prefix + source stem + name_suffix
        load_name, name_note = None, None
        if str(filename_prefix or "").strip() == SN.FROM_LOAD:
            load_name, name_note = SN.name_from_load(prompt, unique_id, follow_load)
            if load_name is None:
                name_note = f"{name_note}: written as video"
        if load_name is not None:
            prefix_eff = load_name if str(subfolder or "").strip() else f"C2C/{load_name}"
        else:
            prefix_eff = "C2C/video" if str(filename_prefix or "").strip() == SN.FROM_LOAD else filename_prefix
        sub_eff, base_dir = subfolder, None
        if naming_eff == SN.FOLDER and not str(subfolder or "").strip() and bool(save_output):
            version = SN.resolve_version(prompt, unique_id)
            from .c2c_video.save_encode import _import_folder_paths
            out_root = _import_folder_paths().get_output_directory()
            prefix_eff = (SN.with_version(load_name, version["version_string"]) if load_name is not None
                          else version["filename_prefix"])
            if SN.outside_output(version["base_path"], out_root):
                base_dir, sub_eff = version["base_path"], version["subfolder_path"]
            else:
                import os as _os
                sub_eff = _os.path.relpath(version["version_path"], out_root).replace("\\", "/")
        naming_path = naming_eff
        if naming_eff == SN.FOLDER and not bool(save_output) and not str(subfolder or "").strip():
            naming_path = "ComfyUI counter"          # a temp (preview) file takes no version
        path_info = resolve_output_path(
            spec=spec,
            naming=naming_path,
            filename_prefix=prefix_eff,
            subfolder=sub_eff,
            save_output=bool(save_output),
            width=int(w),
            height=int(h),
            base_dir=base_dir,
        )

        # L9.52 (owner 2026-10-25: "dont write workflow metadata into video"): the workflow goes into '<name>.png' next
        # to the file (its first written frame), never into the video container
        first = images[:1] if torch.is_tensor(images) else None
        result = encode_video(
            images,
            spec,
            path_info=path_info,
            fps=eff_fps,
            quality=int(quality),
            colorspace_in=cs_in,
            colorspace_out=cs_out,
            audio=audio,
            pad_note=pad_note,
            colour=colour,
            extras=extras,
            cm=cm_plan.cm if cm_plan is not None else None,
            interop_id=exr_id,
            carry=carry,
        )
        workflow_png = None
        if save_metadata:
            from .c2c_video.save_encode import write_workflow_png
            if first is None:
                first = getattr(images, "first_frame", None)
            workflow_png = write_workflow_png(first, path_info[0], prompt, extra_pnginfo)

        if stream_info is not None:
            result["warnings"] = list(result.get("warnings") or []) + list(stream_info.warnings)
            images.close()
            images = images.last_frame if images.last_frame is not None else torch.zeros(1, 64, 64, 3)
        filenames = "\n".join(result["paths"])
        ui_entry: dict[str, Any] = {
            "filename": result["filename"],
            "subfolder": result["subfolder"],
            "type": result["type"],
            "format": result["format"],
            "frames": result["frames"],
            "fps": result["fps"],
            "width": result["width"],
            "height": result["height"],
            "bit_depth": result["bit_depth"],
            "colorspace": result["colorspace"],
            "alpha": result["alpha"],
            "preview": result["preview"],
            "warnings": link_warn + list(result.get("warnings") or []),
            "link": link_report,
            "name_from": name_note,
            "carry": None if carry is None else {
                "first_number": (carry["numbers"] or [None])[0], "last_number": (carry["numbers"] or [None])[-1],
                "timecode": carry["timecode"], "pixel_aspect": carry["pixel_aspect"], "reel": carry["reel"],
                "note": carry["note"]},
            "codec": spec.codec,
            "format_from": format_how,
            "naming": naming_eff,
            "version": None if version is None else {k: version[k] for k in ("version_string", "origin",
                                                                              "version_path", "node")},
            "pix_fmt": spec.pix_fmt_for(bool(result["alpha"])),
            "profile": spec.profile,
            "colour_tags": {"primaries": colour.primaries, "transfer": colour.transfer, "range": colour.range},
            "stream": None if stream_info is None else {"method": stream_info.method, "exact": stream_info.exact,
                                                        "bounded": stream_info.bounded},
            "mode": "written",
            "workflow_png": workflow_png,
            "colour": None if cm_plan is None else dict(cm_plan.report(), interop_id=exr_id),
            "source": source_proxy,
            **common,
        }
        return {
            "ui": {"c2c_save_video": [ui_entry]},
            "result": (filenames, images),
        }


def _proxy_off() -> bool:
    from .c2c_video.save_encode import proxy_disabled
    return proxy_disabled()


def _fingerprint(settings: dict, images, latent) -> str:
    """What a preview showed: the settings plus a sample of the frames (or the latent). The Write button sends it
    back; a write whose frames or settings differ is previewed again instead (L9.40)."""
    h = hashlib.md5()
    for k in sorted(settings):
        h.update(f"{k}={settings[k]!r};".encode())
    src = images if torch.is_tensor(images) else None
    if src is None and isinstance(latent, dict) and torch.is_tensor(latent.get("samples")):
        src = latent["samples"]
    if src is not None:
        flat = src.detach().reshape(-1)
        step = max(1, flat.numel() // 65536)
        h.update(str(tuple(src.shape)).encode())
        h.update(flat[::step].float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


NODE_CLASS_MAPPINGS = {"SaveVideoC2C": SaveVideoC2C}
NODE_DISPLAY_NAME_MAPPINGS = {"SaveVideoC2C": "Save Video (C2C)"}
