"""Endpoints and client identity for the Magnific MCP integration.

Defaults target production; every value is overridable via environment so the
plugin can point at staging or a dev tunnel without code changes (mirrors the
VITE_* envs of frontend/code/apps/editor-plugins).
"""

import os

# Streamable-HTTP MCP endpoint (POST JSON-RPC). Same edge the editor plugins use.
MCP_URL = os.environ.get("MAGNIFIC_MCP_URL", "https://mcp.magnific.com")

# Keycloak realm serving the OAuth 2.0 Device Authorization Grant (RFC 8628).
OAUTH_ISSUER = os.environ.get("MAGNIFIC_OAUTH_ISSUER", "https://auth.magnific.com/realms/mcp")

# Public client with the device grant enabled. Reuses the editor-plugins client
# until a dedicated `magnific-comfyui` client exists in the realm.
OAUTH_CLIENT_ID = os.environ.get("MAGNIFIC_OAUTH_CLIENT_ID", "magnific-editor-plugins")

OAUTH_SCOPE = os.environ.get("MAGNIFIC_OAUTH_SCOPE", "openid profile email mcp:custom-audience")

# Sent as X-Pikaso-Client on every MCP request for per-host attribution
# (allowlisted server-side in McpToolAnalytics::PLUGIN_HOSTS).
CLIENT_TAG = "plugin:comfyui"

CLIENT_NAME = "magnific-comfyui"


# The upstream version this was ported from. Upstream read it out of a sibling
# pyproject.toml; inside CNP that file is the whole pack's, whose version has
# nothing to do with Magnific's API contract. Worse, the lookup FAILING returns
# "0.0.0", which is below every conceivable minVersion floor and would have the
# service report the nodes as retired the moment they were ported.
#
# Bump this by hand when the upstream pack is re-synced, and say so in
# CREDITS.md. It is sent for attribution and compatibility only.
PLUGIN_VERSION = "0.7.0"

# Plugins distribution manifest: latest version ("update available") and
# minVersion (ops-managed killswitch floor) per host — see
# frontend/code/apps/editor-plugins/README.md "The manifest".
MANIFEST_URL = os.environ.get(
    "MAGNIFIC_PLUGINS_MANIFEST_URL", "https://cdn.magnific.com/mgf-pgs/manifest.json"
)
MANIFEST_HOST_KEY = "comfyui"
DOWNLOAD_PAGE_URL = "https://www.magnific.com/plugins"

# Re-checked at most once per hour, on node executions. The manifest itself has
# a 300s TTL; one hour bounds the killswitch reaction for long-lived ComfyUI
# processes while costing at most one tiny fetch per hour.
UPDATE_CHECK_TTL_SECONDS = 3600

# OAuth token blob (access + refresh token). Plain file with 0600 perms in the
# user's home — ComfyUI has no OS-keychain surface, and the server may run
# headless where a keyring is unavailable.
AUTH_FILE = os.environ.get(
    "MAGNIFIC_AUTH_FILE",
    os.path.join(os.path.expanduser("~"), ".magnific", "comfyui_auth.json"),
)

# Longest edge of the image/mask pair images_retouch is given. The retouch
# renderer runs synchronously inside the HTTP request and dies on the 30s
# max_execution_time with a full-resolution photo; the web editor caps its patch
# at 1024, and at 2048 for the full-output models, so nothing is lost by
# matching the higher of the two.
RETOUCH_MAX_EDGE = int(os.environ.get("MAGNIFIC_RETOUCH_MAX_EDGE", "2048"))

# creations_finalize_upload rejects images above 25MB — the same ceiling the
# other hosts guard (hosts/blender config, photoshopUxp, illustratorCep).
MAX_IMAGE_UPLOAD_BYTES = 25 * 1024 * 1024

HTTP_TIMEOUT_SECONDS = 30

# creations_wait long-polls up to 25s server-side; 120 polls ≈ 50 min, matching
# the panel's ceiling (generation never legitimately exceeds it).
WAIT_TIMEOUT_SECONDS = 25
WAIT_MAX_POLLS = 120
