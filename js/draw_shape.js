// DrawShapeMEC: conditional widget visibility based on `shape` dropdown.
// Pattern modeled on KJNodes — hide irrelevant widgets by overriding computeSize to [0,-4].
import { app } from "../../scripts/app.js";
import { setWidgetVisible } from "./_widget_visibility.js";

const SHAPE_FIELDS = {
    circle:            ["cx","cy","radius"],
    rectangle:         ["top_left_x","top_left_y","size_w","size_h"],
    ellipse:           ["cx","cy","rx","ry"],
    polygon:           ["points_json"],
    line:              ["top_left_x","top_left_y","x2","y2","thickness"],
    triangle:          ["cx","cy","radius"],
    star:              ["cx","cy","outer_r","inner_r","num_points"],
    diamond:           ["cx","cy","size_w","size_h"],
    cross:             ["cx","cy","cross_size","thickness"],
    rounded_rectangle: ["top_left_x","top_left_y","size_w","size_h","corner_radius"],
    heart:             ["cx","cy","radius"],
    arrow:             ["cx","cy","arrow_length","head_length","head_width","size_w","rotation"],
};
const ALWAYS_VISIBLE = new Set([
    "width","height","shape","value","feather","rotation","operation","batch_size","points_json"
]);

function applyShapeVisibility(node) {
    const shapeWidget = node.widgets?.find(w => w.name === "shape");
    if (!shapeWidget) return;
    const shape = shapeWidget.value;
    const allowed = new Set(SHAPE_FIELDS[shape] || []);
    for (const w of node.widgets) {
        if (ALWAYS_VISIBLE.has(w.name)) continue;
        // polygon's points_json is in ALWAYS_VISIBLE so we hide it for non-polygon below
        const shouldShow = allowed.has(w.name);
        setWidgetVisible(w, shouldShow);
    }
    // Special: points_json only for polygon
    const pj = node.widgets.find(w => w.name === "points_json");
    if (pj) {
        setWidgetVisible(pj, shape === "polygon");
    }
    const sz = node.computeSize();
    node.size[0] = Math.max(node.size[0], sz[0]);
    node.size[1] = sz[1];
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "MEC.DrawShapeMEC.ConditionalUI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "DrawShapeMEC") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const shapeWidget = this.widgets?.find(w => w.name === "shape");
            if (shapeWidget) {
                const origCallback = shapeWidget.callback;
                shapeWidget.callback = (v, ...rest) => {
                    const r2 = origCallback?.call(shapeWidget, v, ...rest);
                    applyShapeVisibility(this);
                    return r2;
                };
            }
            // Initial pass after widgets settle
            setTimeout(() => applyShapeVisibility(this), 0);
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            setTimeout(() => applyShapeVisibility(this), 0);
            return r;
        };
    },
});
