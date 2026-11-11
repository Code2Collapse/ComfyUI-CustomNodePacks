"""Facepoints engine (myned-ai/facepoints, Apache-2.0 code; CC BY-NC 4.0 weights, not shipped). See model.py / detect.py
for the PORTED FROM headers; NOTICE has the attribution."""
from .detect import SELF_BOX, Detector, box_of_points, square_box  # noqa: F401
from .model import read_config  # noqa: F401
