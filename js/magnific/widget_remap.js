// widgets_values by name across node-definition reorders.
//
// A saved workflow stores widget values as a bare positional array, so moving
// or inserting a widget in a node's INPUT_TYPES would silently hand each value
// to the wrong widget. Current frontends also list every widget by name in the
// node's `inputs` (`widget.name`), in the order it had when saved, which is
// enough to rebuild the array in the order the current definition expects.
//
// Older frontends recorded `widget.name` only for widgets converted to sockets.
// Such a partial list cannot say where the other values sit, so it is left
// alone: a complete recording leaves at most one serialized value unnamed (a
// widget the plugin adds at runtime, such as the cost estimate), and anything
// beyond that marks the list as partial.
//
// No ComfyUI import on purpose: the registry is passed in, so the same code
// runs under node for the tests.

const WIDGET_INPUT_TYPES = new Set(["STRING", "INT", "FLOAT", "BOOLEAN", "COMBO"]);

const MAX_UNNAMED_VALUES = 1;

// A `forceInput` primitive is a socket without a widget: it never takes a
// widgets_values slot even though its type says STRING.
const isWidgetSpec = (spec) =>
  !spec?.[1]?.forceInput && (Array.isArray(spec?.[0]) || WIDGET_INPUT_TYPES.has(spec?.[0]));

const widgetDefault = (spec) => {
  if (spec?.[1] && Object.hasOwn(spec[1], "default")) return spec[1].default;
  return Array.isArray(spec?.[0]) ? spec[0][0] : undefined;
};

// Widget names in definition order, with the seed's companion widget in the
// slot the frontend gives it, plus each widget's default for a slot the saved
// workflow predates.
export const currentWidgetLayout = (nodeData) => {
  const input = nodeData?.input;
  if (!input) return null;
  const names = [];
  const sockets = new Set();
  const defaults = new Map();
  let controlAfter = null;
  for (const group of [input.required ?? {}, input.optional ?? {}]) {
    for (const [name, spec] of Object.entries(group)) {
      if (!isWidgetSpec(spec)) {
        sockets.add(name);
        continue;
      }
      names.push(name);
      defaults.set(name, widgetDefault(spec));
      if (spec[1]?.control_after_generate) {
        names.push("control_after_generate");
        controlAfter = name;
      }
    }
  }
  return { names, sockets, defaults, controlAfter };
};

// Widget names in the order the workflow was saved with, or null when the
// list cannot be trusted. A `widget` entry on one of the definition's sockets
// (a forceInput STRING) owns no value and is skipped. A name the definition
// knows neither way is a retired widget whose value may still sit in the
// array, and nothing says whether it does, so the remap gives up rather than
// pair every later value with the wrong widget.
export const savedWidgetOrder = (node, layout) => {
  const known = new Set(layout.names);
  const names = [];
  for (const input of node.inputs ?? []) {
    const name = input.widget?.name;
    if (!name || layout.sockets.has(name)) continue;
    if (!known.has(name)) return null;
    names.push(name);
    if (name === layout.controlAfter) names.push("control_after_generate");
  }
  return names;
};

export const remapWidgetValues = (node, nodeData) => {
  const layout = currentWidgetLayout(nodeData);
  if (!layout || !Array.isArray(node.widgets_values)) return false;
  const saved = savedWidgetOrder(node, layout);
  if (!saved?.length) return false;
  const unnamed = node.widgets_values.length - saved.length;
  if (unnamed < 0 || unnamed > MAX_UNNAMED_VALUES) return false;
  if (saved.join("|") === layout.names.slice(0, saved.length).join("|")) return false;
  const byName = new Map(saved.map((name, index) => [name, node.widgets_values[index]]));
  const trailing = node.widgets_values.slice(saved.length);
  node.widgets_values = [
    ...layout.names.map((name) => (byName.has(name) ? byName.get(name) : layout.defaults.get(name))),
    ...trailing,
  ];
  return true;
};

// `registry` is LiteGraph.registered_node_types: type → class with `nodeData`.
export const remapMagnificWidgetValues = (graphData, registry) => {
  const graphs = [graphData, ...Object.values(graphData?.definitions?.subgraphs ?? {})];
  for (const graph of graphs) {
    for (const node of graph?.nodes ?? []) {
      if (typeof node?.type !== "string" || !node.type.startsWith("Magnific")) continue;
      remapWidgetValues(node, registry?.[node.type]?.nodeData);
    }
  }
};
