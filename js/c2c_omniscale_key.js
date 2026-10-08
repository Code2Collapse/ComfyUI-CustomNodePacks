// C2C › OmniScale: one place to check the OmniScale (Magnific) connection (owner D0.14 / A9: "no option to add an
// API key for the OmniScale nodes").
//
// Two implementations register the same Magnific* node ids, and the pack that loads last wins:
//   - ComfyUI-OmniScale (installed): API key, masked, in Settings › C2C › OmniScale › API key.
//   - CustomNodePacks' own copy (OmniScale not installed): Magnific account sign-in (menu Magnific › Sign in).
// This check asks whichever copy is active and says what it needs. Both serve /magnific/status behind the
// per-process token from /magnific/csrf-token.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const CHECK_ID = "c2c.omniscale.check";

function toast(severity, detail) {
    try {
        app.extensionManager?.toast?.add?.({ severity, summary: "OmniScale", detail, life: 6000 });
    } catch { /* best-effort */ }
}

export async function checkOmniScale() {
    let token = "";
    try {
        const r = await api.fetchApi("/magnific/csrf-token");
        if (r.ok) token = (await r.json())?.token || "";
    } catch { /* handled below */ }
    if (!token) {
        toast("warn", "The OmniScale / Magnific routes did not answer. Is either pack loaded? Restart ComfyUI after installing.");
        return null;
    }
    let st = null;
    try {
        const r = await api.fetchApi("/magnific/status", { headers: { "X-Magnific-Token": token } });
        st = r.ok ? await r.json() : null;
    } catch { st = null; }
    if (!st) {
        toast("error", "Could not read the OmniScale status from the server.");
        return null;
    }
    if (st.auth_mode === "api_key") {
        if (st.api_key_configured) toast("success", "OmniScale pack active and an API key is set.");
        else toast("warn", "OmniScale pack active, but no API key yet: paste it in Settings › C2C › OmniScale › API key.");
    } else if (st.signed_in) {
        toast("success", "CustomNodePacks' copy is active and signed in to Magnific (account sign-in, no API key).");
    } else {
        toast("warn", "CustomNodePacks' copy is active: it needs the Magnific sign-in (menu Magnific › Sign in). "
            + "For an API key, install the OmniScale pack (ComfyUI-OmniScale); it replaces these nodes.");
    }
    return st;
}

function renderCheckButton() {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = "Check connection";
    btn.className = "p-button p-component p-button-sm p-button-secondary";
    btn.addEventListener("click", async () => {
        btn.disabled = true;
        try { await checkOmniScale(); } finally { btn.disabled = false; }
    });
    return btn;
}

app.registerExtension({
    name: "C2C.OmniScale.Check",
    settings: [
        {
            id: CHECK_ID,
            name: "Check the OmniScale connection (API key or Magnific sign-in)",
            tooltip: "Asks the active OmniScale / Magnific nodes whether they can reach the service: an API key set "
                + "(OmniScale pack) or a signed-in account (CustomNodePacks' copy).",
            category: ["C2C", "OmniScale", "Connection"],
            type: renderCheckButton,
            defaultValue: "",
        },
    ],
    commands: [
        { id: "c2c.omniscale.checkConnection", label: "C2C: Check the OmniScale connection", function: checkOmniScale },
    ],
});
