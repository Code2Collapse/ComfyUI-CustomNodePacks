"""C2C Owner-Hosted App Gateway (A26 L9.206).

A standalone service that publishes owner workflows as narrow apps: runners get a manifest of logical inputs and
outputs, never the ComfyUI graph. Run it as its own process (``python -m c2c_app_gateway serve --config <file>``) in
front of a ComfyUI worker bound to a private address. It is NOT imported by the ComfyUI pack.

Design: docs/research/L9.206_owner_app_gateway_design.md (workspace) - protects the workflow from remote runners, not
from an administrator of the host.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
