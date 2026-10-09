/**
 * C2C VAE Quality Decode: saved workflows from before tile_mode (L7.59).
 *
 * tile_mode (auto / off / manual) was added after clamp_output, so an older save carries 4 or 5 widget values and the
 * new widget takes its default, "auto". The owner chose auto as the default for everyone (2026-10-09); a save that had
 * asked for a tile size (tile_size > 0) keeps exactly that, as "manual".
 */
import { app } from "../../scripts/app.js";

const NODE = "C2CVAEQualityDecode";
const MODE_INDEX = 5;          // force_fp32, tile_size, apply_aces, exposure, clamp_output, tile_mode

if (!globalThis.__c2cVaeQualityDecodeExt) {
    globalThis.__c2cVaeQualityDecodeExt = true;
    app.registerExtension({
        name: "C2C.VAEQualityDecode",
        async beforeRegisterNodeDef(nodeType, nodeData) {
            if (nodeData?.name !== NODE) return;
            const migrate = (node) => {
                const legacy = node.__c2cLegacyTileSize;
                if (legacy === undefined) return;
                delete node.__c2cLegacyTileSize;
                const mode = node.widgets?.find((w) => w.name === "tile_mode");
                if (mode) mode.value = legacy > 0 ? "manual" : "auto";
            };
            // configure() itself, before any onConfigure: handlers in that chain pad widgets_values to the widget
            // count (classic renderer), after which an old save no longer looks old
            const configure = nodeType.prototype.configure;
            nodeType.prototype.configure = function (info) {
                const saved = info?.widgets_values;
                if (Array.isArray(saved) && saved.length <= MODE_INDEX) {
                    this.__c2cLegacyTileSize = Number(saved[1]) || 0;
                }
                return configure.apply(this, arguments);
            };
            // the front end writes widget values again after configure: migrate once the whole graph is in
            const after = nodeType.prototype.onAfterGraphConfigured;
            nodeType.prototype.onAfterGraphConfigured = function (...a) {
                const r = after?.apply(this, a);
                migrate(this);
                return r;
            };
        },
    });
}
