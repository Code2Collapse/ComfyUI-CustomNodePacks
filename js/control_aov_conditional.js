// ControlAOVC2C: HIDE parameters that don't apply to the current backend
// selection (user rule 2026-06-24: show a sub-option only when relevant, hide
// otherwise). Re-applied whenever depth_model / pose_model / edge_model change.
//   depth_size        -> only DepthAnything-family backbones (da_v2/da_v1/midas/da3)
//   depth_custom_ckpt -> DA-family + delegated depth_anything_* (NOT depthcrafter/depth_pro/dvd/off)
//   pose_ckpt         -> only the vitpose backend
//   run_canny/aperture-> only internal OpenCV Canny
//   canny_low/high    -> internal Canny + delegated "canny" preprocessor
import { app } from "../../scripts/app.js";

const DEPTH_SIZE_MODELS = ["da_v2", "da_v1", "midas", "da3"];

function setHidden(w, hidden) {
    if (!w) return;
    if (hidden) {
        if (!("__c2c_origType" in w)) w.__c2c_origType = w.type;
        if (!("__c2c_origComputeSize" in w)) w.__c2c_origComputeSize = w.computeSize;
        w.type = "hidden";
        w.computeSize = () => [0, -4];
        w.hidden = true;
        const el = w.element;
        if (el) {
            if (!("__c2c_origElDisplay" in w)) w.__c2c_origElDisplay = el.style.display;
            el.style.display = "none";
        }
    } else {
        if ("__c2c_origType" in w) { w.type = w.__c2c_origType; delete w.__c2c_origType; }
        if ("__c2c_origComputeSize" in w) {
            const cs = w.__c2c_origComputeSize;
            if (cs === undefined) delete w.computeSize; else w.computeSize = cs;
            delete w.__c2c_origComputeSize;
        }
        w.hidden = false;
        const el = w.element;
        if (el) { el.style.display = ("__c2c_origElDisplay" in w) ? (w.__c2c_origElDisplay ?? "") : ""; delete w.__c2c_origElDisplay; }
    }
}

function applyVisibility(node) {
    const get = (n) => node.widgets?.find(w => w.name === n);
    const dm = String(get("depth_model")?.value ?? "off");
    const nm = String(get("normal_model")?.value ?? "off");
    const pm = String(get("pose_model")?.value ?? "off");
    const em = String(get("edge_model")?.value ?? "internal_canny");
    const internalCanny = em === "internal_canny";
    const rz = String(get("resize")?.value ?? "off");
    const rzOn = rz !== "off";
    const fitv = String(get("fit")?.value ?? "crop");

    const visible = {
        depth_size:        DEPTH_SIZE_MODELS.includes(dm),
        depth_custom_ckpt: dm !== "off" && dm !== "depth_pro" && dm !== "depthcrafter" && dm !== "dvd",
        pose_ckpt:         pm === "vitpose",
        run_canny:         internalCanny,
        canny_aperture:    internalCanny,
        canny_low:         internalCanny || em === "canny",
        canny_high:        internalCanny || em === "canny",
        // a per-AOV weight / modifier only matters when that AOV is actually produced
        depth_weight:      dm !== "off",
        depth_invert:      dm !== "off",
        normal_weight:     nm !== "off",
        pose_weight:       pm !== "off",
        canny_weight:      em !== "off",
        // in-node resize — one dropdown drives width/height vs scale fields
        width:             rz === "width/height",
        height:            rz === "width/height",
        scale:             rz === "scale",
        divisible_by:      rzOn,
        fit:               rzOn,
        resize_filter:     rzOn,
        pad_color:         rzOn && fitv === "pad",
    };
    for (const w of node.widgets) {
        setHidden(w, (w.name in visible) ? !visible[w.name] : false);
    }
    const sz = node.computeSize();
    node.size[0] = Math.max(node.size[0], sz[0]);
    node.size[1] = sz[1];
    node.setDirtyCanvas(true, true);
}

function hookWidget(node, name) {
    const w = node.widgets?.find(x => x.name === name);
    if (!w) return;
    const orig = w.callback;
    w.callback = (v, ...rest) => {
        const r = orig?.call(w, v, ...rest);
        applyVisibility(node);
        return r;
    };
}

app.registerExtension({
    name: "C2C.ControlAOV.ConditionalUI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ControlAOVC2C") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            for (const n of ["depth_model", "normal_model", "pose_model", "edge_model",
                             "resize", "fit"]) hookWidget(this, n);
            setTimeout(() => applyVisibility(this), 0);
            return r;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            setTimeout(() => applyVisibility(this), 0);
            return r;
        };
    },
});
