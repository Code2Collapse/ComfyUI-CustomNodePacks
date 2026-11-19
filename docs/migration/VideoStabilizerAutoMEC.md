# VideoStabilizerAutoMEC -> VideoStabilizerMEC (method from force_backend)

**Status:** removed on 2026-11-18 (L7.65 P17). Saved workflows and API prompts migrate automatically.
`C2C_LEGACY_STABILIZERS=1` registers the old class again. While it is registered, nothing migrates it, and it warns
once per run.

| Old input / output | VideoStabilizerMEC | Conversion |
|---|---|---|
| frames | frames | link moved |
| frame_rate, padding_color | same names | values copied |
| preset (handheld_light / handheld_heavy / vehicle / tripod_lock) | preset | value copied; the preset drives the stabiliser settings, as it did before |
| force_backend | method | `auto` -> `auto`, `classic` -> `classic`, `flow` -> `dense_flow` (it always ran DIS) |
| (none) | framing_mode, transform_mode, camera_lock, strength, smooth, keep_fov, raft_iters, use_half, allow_download | successor defaults; these equal the placeholders the old node passed, and a preset overrides them |
| outputs 0, 1, 2 | outputs 0, 1, 2 | links moved by index; output 3 motion_meta is new |

Nothing is dropped.

**Parity test:**
- Setup: saved widgets `[16.0, flow, handheld_heavy, "0, 0, 0"]`, run through both migrations on the 10-frame clip.
- Frames and padding mask are bit-exact. The mask is non-empty: mean 0.051.
- Info: identical text, apart from the elapsed time.
- The API path is also covered by `tests/test_l765_p17_stabilizer_aliases.py`, which checks every force_backend value.
