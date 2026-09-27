/**
 * c2c_ui/wolf.js — C2C brand mark (original geometric wolf head).
 *
 * License: Apache-2.0
 */

const SVG_NS = "http://www.w3.org/2000/svg";

/**
 * Inline SVG wolf head — angular ears, narrowing muzzle, eye slits as
 * negative space. Reads clearly at 20–48 px.
 *
 * @param {number} [size=40] CSS px width/height
 * @returns {SVGSVGElement}
 */
export function wolfMark(size = 40) {
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("viewBox", "0 0 48 48");
    svg.setAttribute("width", String(size));
    svg.setAttribute("height", String(size));
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "C2C");
    svg.classList.add("c2c-wolf-mark");

    const path = document.createElementNS(SVG_NS, "path");
    // One path, cut-outs via evenodd. Tall ears set close to the brow, cheek
    // ruffs, and a long V muzzle make it a wolf rather than a cat (the first
    // draft, ears at the corners of a round head, read as a cat).
    path.setAttribute("fill", "currentColor");
    path.setAttribute("fill-rule", "evenodd");
    path.setAttribute("d", [
        // Outline: left ear tip -> brow -> right ear tip -> cheek ruff -> muzzle -> back
        "M 11 3 L 17.5 14.5 L 24 12.5 L 30.5 14.5 L 37 3 L 41 19 L 45 27 L 37 31",
        "L 29.5 38.5 L 24 45.5 L 18.5 38.5 L 11 31 L 3 27 L 7 19 Z",
        // Slanted eyes
        "M 14.5 23.5 L 21 25 L 20 27.5 L 14 25.8 Z",
        "M 33.5 23.5 L 27 25 L 28 27.5 L 34 25.8 Z",
        // Nose
        "M 21.6 38.2 L 26.4 38.2 L 24 41 Z",
        // Inner ears
        "M 11.2 8.5 L 14.6 14.8 L 9.6 17.2 Z",
        "M 36.8 8.5 L 33.4 14.8 L 38.4 17.2 Z",
    ].join(" "));

    svg.appendChild(path);
    return svg;
}
