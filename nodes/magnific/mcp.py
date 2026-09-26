"""Minimal MCP streamable-HTTP client: initialize handshake + tools/call.

Python port of editor-plugins/src/api/McpClient.ts. Single-response SSE bodies
are parsed eagerly — the plugin never needs server-initiated streams.
"""

import json
import threading
import urllib.error
import urllib.request
from typing import Optional

from . import auth, config, tls_trust

PROTOCOL_VERSION = "2025-06-18"

# JSON-RPC codes McpRequestAuthenticator reserves for feature gating.
MCP_PREMIUM_REQUIRED = -32001
MCP_PERMISSION_DENIED = -32002

_GATE_MESSAGES = {
    MCP_PREMIUM_REQUIRED: "Magnific tools require a Premium plan — upgrade at magnific.com.",
    MCP_PERMISSION_DENIED: "Your organization has disabled Magnific API access.",
}


def _error_detail(error: urllib.error.HTTPError) -> str:
    """The server's message and request_id, appended to a bare status code.
    Without it a 500 reaches the node as three digits, and the request_id that
    would find the failure in the server logs is thrown away with the body."""
    try:
        body = error.read().decode("utf-8", "replace")
    except OSError:
        return ""
    try:
        parsed = json.loads(body)
    except ValueError:
        text = " ".join(body.split())
        return f": {text[:300]}" if text else ""
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else {}
    if not isinstance(parsed, dict):
        return ""
    payload = parsed.get("error") if isinstance(parsed.get("error"), dict) else parsed
    message = payload.get("message") or parsed.get("message")
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    request_id = data.get("request_id") or parsed.get("request_id")
    detail = f": {message}" if message else ""
    return f"{detail} (request_id: {request_id})" if request_id else detail


class McpError(RuntimeError):
    def __init__(self, message: str, code: Optional[int] = None):
        super().__init__(message)
        self.code = code


class McpClient:
    def __init__(self) -> None:
        self._next_id = 1
        self._session_id: Optional[str] = None
        self._initialized = False
        self._lock = threading.Lock()

    def call_tool(self, name: str, arguments: Optional[dict] = None) -> dict:
        """Returns the tool result ({content, structuredContent, isError})."""
        with self._lock:
            self._ensure_initialized()
            result = self._request("tools/call", {"name": name, "arguments": arguments or {}})
        if isinstance(result, dict) and result.get("isError"):
            text = next(
                (entry.get("text") for entry in result.get("content", []) if entry.get("type") == "text"),
                None,
            )
            raise McpError(text or f"Tool {name} failed")
        return result if isinstance(result, dict) else {}

    def structured(self, name: str, arguments: Optional[dict] = None) -> dict:
        result = self.call_tool(name, arguments)
        structured = result.get("structuredContent")
        return structured if isinstance(structured, dict) else {}

    def text(self, name: str, arguments: Optional[dict] = None) -> str:
        """Joined text blocks — the lean TOON tools (folders_list,
        images_models_list) emit text instead of structuredContent."""
        result = self.call_tool(name, arguments)
        return "\n".join(
            entry.get("text", "") for entry in result.get("content", []) if entry.get("type") == "text"
        )

    def list_tools(self) -> list:
        """Tool definitions (name + inputSchema) — the schemas carry enums the
        server has no dedicated catalog tool for (e.g. music models)."""
        with self._lock:
            self._ensure_initialized()
            result = self._request("tools/list", {})
        return result.get("tools", []) if isinstance(result, dict) else []

    def reset(self) -> None:
        with self._lock:
            self._session_id = None
            self._initialized = False

    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": config.CLIENT_NAME, "version": config.PLUGIN_VERSION},
            },
        )
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self._initialized = True

    def _request(self, method: str, params: dict):
        request_id = self._next_id
        self._next_id += 1
        body, content_type = self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        message = self._find_response(body, content_type, request_id)
        if message is None:
            raise McpError(f"Empty MCP response for {method}")
        error = message.get("error")
        if error:
            code = error.get("code")
            raise McpError(_GATE_MESSAGES.get(code) or error.get("message") or "MCP error", code)
        return message.get("result")

    def _send(self, payload: dict):
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {auth.get_access_token()}",
            "X-Pikaso-Client": config.CLIENT_TAG,
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        request = urllib.request.Request(
            config.MCP_URL, data=json.dumps(payload).encode(), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=config.HTTP_TIMEOUT_SECONDS * 2, context=tls_trust.get_ssl_context()) as response:
                session_id = response.headers.get("mcp-session-id")
                if session_id:
                    self._session_id = session_id
                return response.read().decode(), response.headers.get("content-type", "")
        except urllib.error.HTTPError as error:
            if error.code == 401:
                # The bearer was rejected despite a fresh-looking expiry —
                # drop the MCP session AND the cached access token, or every
                # retry would resend the same rejected bearer until expiry.
                self._session_id = None
                self._initialized = False
                auth.invalidate_access_token()
                raise McpError("Magnific session expired — sign in again from the Magnific menu.", 401)
            raise McpError(f"MCP request failed (HTTP {error.code}){_error_detail(error)}") from error
        except urllib.error.URLError as error:
            raise McpError(f"Could not reach Magnific ({error.reason})") from error

    @staticmethod
    def _find_response(body: str, content_type: str, request_id: int) -> Optional[dict]:
        if "text/event-stream" in content_type:
            candidates = [
                line[len("data:"):].strip()
                for event in body.split("\n\n")
                for line in event.split("\n")
                if line.startswith("data:")
            ]
        else:
            candidates = [body]
        for candidate in candidates:
            if not candidate.strip():
                continue
            try:
                parsed = json.loads(candidate)
            except ValueError:
                continue  # non-JSON SSE event (e.g. ping)
            messages = parsed if isinstance(parsed, list) else [parsed]
            for message in messages:
                if isinstance(message, dict) and message.get("id") == request_id:
                    return message
        return None


_client: Optional[McpClient] = None
_client_lock = threading.Lock()


def client() -> McpClient:
    global _client
    with _client_lock:
        if _client is None:
            _client = McpClient()
        return _client
