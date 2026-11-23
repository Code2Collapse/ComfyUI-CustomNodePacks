// _lockparams_core.js - "Lock parameter values" (V6-A, owner 2026-11-21): the pure core. The crypto envelope, which
// widgets lock, the neutral placeholders they keep, and how the real values enter the one queued prompt.
// Design: docs/h2vault/V6_cnp_vault_enhancements.md section 2 (workspace docs).
//
// Rules: NO imports (tests run it under node; the frontend loads every .js in js/, so it is side-effect free at
// import); WebCrypto only (PBKDF2-SHA256 -> AES-256-GCM, the key never leaves the browser and is non-extractable);
// never logs a value.
//
// License: Apache-2.0

export const LOCK_PROP = "c2c_locked_params";      // node.properties key holding the envelope
export const LOCKED_META = "c2c_locked";           // prompt entry _meta key: the input names that carry real values
export const PBKDF2_ITERATIONS = 600000;           // OWASP 2023 for PBKDF2-HMAC-SHA256
export const STATUS_WIDGET = "c2c_lock_status";    // the node's status card (never locked, never saved)
const VERSION = 1;
const KDF = "PBKDF2-SHA256";
const ITER_BOUNDS = [1000, 10000000];
const CONTROL_NAMES = new Set(["control_after_generate", "control_before_generate"]);

export class LockError extends Error {
  constructor(message) { super(message); this.name = "LockError"; }
}

const WRONG = "Wrong password, or the locked values were changed.";

// ── bytes ──────────────────────────────────────────────────────────────────────────────────────────────────────
function toB64(u8) {
  let s = "";
  for (let i = 0; i < u8.length; i++) s += String.fromCharCode(u8[i]);
  return btoa(s);
}

function fromB64(text) {
  if (typeof text !== "string" || !text || !/^[A-Za-z0-9+/]+={0,2}$/.test(text)) throw new LockError("bad base64");
  const s = atob(text);
  const u8 = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u8[i] = s.charCodeAt(i);
  return u8;
}

function randomBytes(n) {
  const u8 = new Uint8Array(n);
  globalThis.crypto.getRandomValues(u8);
  return u8;
}

const hex = (u8) => Array.from(u8, (b) => b.toString(16).padStart(2, "0")).join("");
const aad = (env) => new TextEncoder().encode(JSON.stringify({ v: env.v, type: env.type, names: env.names }));

// ── keys and envelopes ─────────────────────────────────────────────────────────────────────────────────────────
/** NFC on every side, never truncated: the same passphrase always derives the same key. */
export function normalizePassword(pw) {
  return String(pw ?? "").normalize("NFC");
}

export function newSalt() {
  return randomBytes(16);
}

/** PBKDF2-SHA256(password, salt, iterations) -> a non-extractable AES-GCM-256 key. */
export async function deriveKey(password, saltBytes, { iterations = PBKDF2_ITERATIONS, subtle = globalThis.crypto.subtle } = {}) {
  const base = await subtle.importKey("raw", new TextEncoder().encode(normalizePassword(password)), "PBKDF2", false,
    ["deriveKey"]);
  return subtle.deriveKey({ name: "PBKDF2", hash: "SHA-256", salt: saltBytes, iterations }, base,
    { name: "AES-GCM", length: 256 }, false, ["encrypt", "decrypt"]);
}

/** Encrypt one node's values under an already derived key (one key per lock operation, a fresh IV per node). */
export async function sealValues({ key, salt, iterations = PBKDF2_ITERATIONS, type, values, subtle = globalThis.crypto.subtle }) {
  const env = { v: VERSION, id: hex(randomBytes(16)), kdf: KDF, iter: iterations, salt: toB64(salt),
    iv: "", ct: "", names: Object.keys(values), type: String(type ?? "") };
  const iv = randomBytes(12);
  env.iv = toB64(iv);
  const data = new TextEncoder().encode(JSON.stringify({ values }));
  env.ct = toB64(new Uint8Array(await subtle.encrypt({ name: "AES-GCM", iv, additionalData: aad(env) }, key, data)));
  return env;
}

/** Structural checks BEFORE any key is derived. Throws LockError. */
export function checkEnvelope(env) {
  const bad = (why) => { throw new LockError(`These locked values are damaged (${why}).`); };
  if (!env || typeof env !== "object") bad("no envelope");
  if (env.v !== VERSION) bad("unknown version");
  if (env.kdf !== KDF) bad("unknown key derivation");
  if (!Number.isInteger(env.iter) || env.iter < ITER_BOUNDS[0] || env.iter > ITER_BOUNDS[1]) bad("iterations");
  if (typeof env.id !== "string" || !/^[0-9a-f]{32}$/.test(env.id)) bad("id");
  if (typeof env.type !== "string") bad("type");
  if (!Array.isArray(env.names) || !env.names.every((n) => typeof n === "string")) bad("names");
  try {
    if (fromB64(env.salt).length !== 16) bad("salt");
    if (fromB64(env.iv).length !== 12) bad("iv");
    if (fromB64(env.ct).length <= 16) bad("ciphertext");
  } catch (e) {
    if (e instanceof LockError && e.message.startsWith("These")) throw e;
    bad("encoding");
  }
}

/** Decrypt with a derived key -> {values}. A wrong key and any change to the envelope read the same. */
export async function openEnvelope(env, key, { subtle = globalThis.crypto.subtle } = {}) {
  checkEnvelope(env);
  let plain;
  try {
    plain = await subtle.decrypt({ name: "AES-GCM", iv: fromB64(env.iv), additionalData: aad(env) }, key, fromB64(env.ct));
  } catch (_) {
    throw new LockError(WRONG);
  }
  let out;
  try { out = JSON.parse(new TextDecoder().decode(plain)); } catch (_) { throw new LockError(WRONG); }
  if (!out || typeof out.values !== "object" || out.values === null) throw new LockError(WRONG);
  return { values: out.values };
}

/** Try one password on many envelopes: one key derivation per (salt, iterations) pair. */
export async function unlockEnvelopes(envs, password, { subtle = globalThis.crypto.subtle } = {}) {
  const opened = new Map();
  const failed = [];
  const keys = new Map();
  for (const env of envs) {
    try {
      checkEnvelope(env);
      const k = `${env.salt}|${env.iter}`;
      if (!keys.has(k)) keys.set(k, deriveKey(password, fromB64(env.salt), { iterations: env.iter, subtle }));
      opened.set(env.id, (await openEnvelope(env, await keys.get(k), { subtle })).values);
    } catch (_) {
      failed.push(env);
    }
  }
  return { opened, failed };
}

export function envelopeOf(node) {
  const env = node?.properties?.[LOCK_PROP];
  if (!env) return null;
  try { checkEnvelope(env); return env; } catch (_) { return null; }
}

// ── which widgets lock ─────────────────────────────────────────────────────────────────────────────────────────
const isPrimitive = (v) => typeof v === "string" || typeof v === "boolean" || (typeof v === "number" && Number.isFinite(v));

function linked(node, w) {
  return (node?.inputs || []).some((i) => i?.widget?.name === w.name && i.link != null);
}

/** {lock: widgets that will be encrypted, skipped: names of widgets whose value cannot be (custom editors)}. */
export function lockableWidgets(node) {
  const lock = [];
  const skipped = [];
  let prevLocked = false;
  for (const w of node?.widgets || []) {
    if (!w || typeof w.name !== "string") { prevLocked = false; continue; }
    const take = (() => {
      if (w.name === STATUS_WIDGET || w.serialize === false || w.options?.serialize === false) return false;
      if (w.type === "button") return false;
      if (CONTROL_NAMES.has(w.name)) return prevLocked && isPrimitive(w.value);
      if (linked(node, w)) return false;
      if (!isPrimitive(w.value)) { if (w.value !== null && w.value !== undefined) skipped.push(w.name); return false; }
      return true;
    })();
    if (take) lock.push(w);
    prevLocked = take;
  }
  return { lock, skipped };
}

/** The neutral value a locked widget keeps (what saves, exports and the canvas ever see). */
export function placeholderFor(w) {
  if (CONTROL_NAMES.has(w?.name)) return "fixed";
  if (w?.type === "combo") {
    let vals = w.options?.values;
    if (typeof vals === "function") { try { vals = vals(w); } catch (_) { vals = null; } }
    return Array.isArray(vals) && vals.length ? vals[0] : "";
  }
  const v = w?.value;
  if (typeof v === "number") {
    let d = Number(w.options?.default ?? 0);
    if (!Number.isFinite(d)) d = 0;
    const lo = Number(w.options?.min), hi = Number(w.options?.max);
    if (Number.isFinite(lo)) d = Math.max(lo, d);
    if (Number.isFinite(hi)) d = Math.min(hi, d);
    return d;
  }
  if (typeof v === "boolean") return false;
  return "";
}

// ── the queued prompt ──────────────────────────────────────────────────────────────────────────────────────────
const isLink = (v) => Array.isArray(v) && v.length === 2 && typeof v[0] === "string" && Number.isInteger(v[1]);

/**
 * Put this tab's decrypted values into the API prompt (mutates `output`). Only inputs the frontend already wrote as
 * plain values are replaced (never a link, never a key it left out); each touched entry gets
 * `_meta.c2c_locked = [names]` so the server keeps them out of history, PNG metadata and queue views.
 * An entry whose class_type is not the type the envelope was sealed for (an envelope copied onto another node, e.g.
 * one that echoes its input) is never filled: it is reported in `mismatched`.
 * @returns {{missing: {execId: string, env: object}[], mismatched: {execId: string, env: object, nodeType: string}[],
 *   injected: number}}
 */
export function injectLockedValues(output, { nodeFor, valuesFor }) {
  const missing = [];
  const mismatched = [];
  let injected = 0;
  for (const [execId, entry] of Object.entries(output || {})) {
    if (!entry || typeof entry !== "object" || !entry.inputs) continue;
    const env = envelopeOf(nodeFor(execId));
    if (!env) continue;
    if (String(entry.class_type ?? "") !== env.type) {
      mismatched.push({ execId, env, nodeType: String(entry.class_type ?? "") });
      continue;
    }
    const values = valuesFor(env);
    if (!values) { missing.push({ execId, env }); continue; }
    const names = [];
    for (const name of env.names) {
      if (!(name in entry.inputs) || isLink(entry.inputs[name]) || !(name in values)) continue;
      entry.inputs[name] = values[name];
      names.push(name);
    }
    entry._meta = { ...(entry._meta || {}), [LOCKED_META]: names };
    injected += names.length;
  }
  return { missing, mismatched, injected };
}
