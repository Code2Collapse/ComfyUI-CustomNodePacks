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
from .c2c_video.save_formats import (
    SaveVideoError,
    available_formats,
    resolve_format,
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
                "format": (available_formats(),),
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
                "colorspace_in": (spaces, {"default": default_in}),
                "colorspace_out": (["same as input"] + list(spaces), {"default": "same as input"}),
                "naming": (["default", "ComfyUI counter", "Folder Version"], {"default": "default"}),
                "naming_resolved": ("STRING", {"default": "", "tooltip": "Set by the UI from c2c.saveVideo.naming."}),
                "subfolder": ("STRING", {"default": ""}),
                "save_output": ("BOOLEAN", {"default": True}),
                "latent": ("LATENT", {"tooltip": "Stream save: a video latent decoded straight into the file, in order, "
                                                 "keeping the VAE's causal state - the full frame batch is never held in "
                                                 "RAM. Wire this and vae instead of images."}),
                "vae": ("VAE", {"tooltip": "The VAE for latent (stream save)."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, format, fps, filename_prefix, quality, images=None,  # noqa: A002
                   audio=None, video_info=None, colorspace_in=None, colorspace_out=None,
                   naming="default", naming_resolved="", subfolder="", save_output=True, **kw):
        h = hashlib.md5()
        for k, v in sorted({
            "format": format, "fps": fps, "filename_prefix": filename_prefix,
            "quality": quality, "colorspace_in": colorspace_in,
            "colorspace_out": colorspace_out, "naming": naming,
            "naming_resolved": naming_resolved, "subfolder": subfolder,
            "save_output": save_output,
        }.items()):
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
    ):
        stream_info = None
        if latent is not None or vae is not None:
            if images is not None:
                raise SaveVideoError("Wire images, OR latent + vae - not both.")
            if latent is None or vae is None:
                raise SaveVideoError("Stream save needs both latent and vae wired.")
            from .c2c_video.stream_decode import stream_decode
            images, stream_info = stream_decode(vae, latent)
            src = latent.get("c2c_source_frames") if isinstance(latent, dict) else None
            if isinstance(src, int) and 0 < src < images.shape[0]:
                images.shape = (src,) + tuple(images.shape[1:])        # the frames the clip really had
        if images is None:
            raise SaveVideoError("Connect an IMAGE batch, or latent + vae, to save.")
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.shape[0] == 0:
            raise SaveVideoError("IMAGE batch is empty — nothing to save.")

        spec = resolve_format(str(format))
        eff_fps = float(fps)
        if video_info and abs(eff_fps - 24.0) < 1e-6:
            eff_fps = float(video_info.get("loaded_fps", eff_fps))

        cs_in = colorspace_in or default_colorspace_in()
        cs_out = colorspace_out or "same as input"
        if cs_out == "same as input" and spec.codec.startswith("oiio_exr"):
            cs_out = default_colorspace_out_exr()

        pad_note = None
        if spec.yuv420 and stream_info is not None:
            n0, h0, w0, c0 = images.shape
            if h0 % 2 or w0 % 2:                                      # pad each chunk as it arrives
                images._transform = lambda chunk: pad_edge_replicate(chunk)[0]
                images.shape = (n0, h0 + h0 % 2, w0 + w0 % 2, c0)
                pad_note = f"{w0}x{h0} -> {w0 + w0 % 2}x{h0 + h0 % 2}"
        elif spec.yuv420:
            images, pad_note = pad_edge_replicate(images)

        _, h, w, _ = images.shape
        naming_eff = effective_naming(str(naming), str(naming_resolved or ""))
        path_info = resolve_output_path(
            spec=spec,
            naming=naming_eff,
            filename_prefix=filename_prefix,
            subfolder=subfolder,
            save_output=bool(save_output),
            width=int(w),
            height=int(h),
        )

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
        )

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
            "warnings": result.get("warnings") or [],
            "codec": spec.codec,
            "stream": None if stream_info is None else {"method": stream_info.method, "exact": stream_info.exact,
                                                        "bounded": stream_info.bounded},
        }
        return {
            "ui": {"c2c_save_video": [ui_entry]},
            "result": (filenames, images),
        }


NODE_CLASS_MAPPINGS = {"SaveVideoC2C": SaveVideoC2C}
NODE_DISPLAY_NAME_MAPPINGS = {"SaveVideoC2C": "Save Video (C2C)"}
