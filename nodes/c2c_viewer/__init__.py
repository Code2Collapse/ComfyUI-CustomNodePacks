"""Hidden helper nodes of the Matte viewer (L7.114, owner A14; owner answer round 5: "Yes, hidden helpers").

The viewer on Mask + Matting (js/c2c_viewer/) talks to the server only through ComfyUI's own queue: it posts small
helper prompts to /prompt (front of the queue) and reads the helpers' `ui` output from the websocket / /history.
No new HTTP route. The three helpers are registered (a prompt naming them validates) but DEV_ONLY, so frontend 1.53
keeps them out of the node library, the search box and the Add Node menu (unless Comfy.DevMode is on):

    C2CViewerPlateProxy  the plate as proxy JPEG frames
    C2CViewerDetect      one frame + text / points / boxes -> one mask per object (SAM 3 / 3.1)
    C2CViewerResult      the rendered alpha + plate as proxy frames, and the QC numbers
"""
from __future__ import annotations

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .plate_proxy import C2CViewerPlateProxy
    from .detect import C2CViewerDetect
    from .result import C2CViewerResult

    NODE_CLASS_MAPPINGS = {
        "C2CViewerPlateProxy": C2CViewerPlateProxy,
        "C2CViewerDetect": C2CViewerDetect,
        "C2CViewerResult": C2CViewerResult,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        "C2CViewerPlateProxy": "Viewer helper: Plate Proxy",
        "C2CViewerDetect": "Viewer helper: Detect",
        "C2CViewerResult": "Viewer helper: Result",
    }
except Exception as _exc:  # noqa: BLE001 - the helpers must never take Mask + Matting down with them
    import logging
    logging.getLogger("C2C.Viewer").warning("[C2C] Matte viewer helpers unavailable: %s", _exc)
