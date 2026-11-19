# VideoStabilizerFlowMEC -> VideoStabilizerMEC (method = dense_flow)

**Status:** removed on 2026-11-18 (L7.65 P17). Saved workflows and API prompts migrate automatically.
`C2C_LEGACY_STABILIZERS=1` registers the old class again. While it is registered, nothing migrates it, and it warns
once per run.

The old node always ran OpenCV DIS optical flow, never RAFT. The migration therefore sets `dense_flow`, not
`raft_flow`.

| Old input / output | VideoStabilizerMEC | Conversion |
|---|---|---|
| frames | frames | link moved |
| frame_rate, framing_mode, transform_mode, camera_lock, strength, smooth, keep_fov, padding_color | same names | values copied |
| raft_iters, use_half | same names | values copied; dense_flow does not read them, and the old node did not either |
| (none) | method / preset | set to `dense_flow` / `manual` |
| (none) | allow_download | default `False` |
| outputs 0, 1, 2 | outputs 0, 1, 2 | links moved by index; output 3 motion_meta is new |

Nothing is dropped.

**Parity test:**
- Setup: saved widgets `[16.0, crop_and_pad, similarity, False, 0.8, 0.6, 0.6, "127, 127, 127", 12, True]`, run through both migrations on the 10-frame clip.
- Frames and padding mask are bit-exact (`torch.equal`). The mask is non-empty: mean 0.047.
- Info: identical text, apart from the elapsed time.
- Must-fail: running the successor with `classic` instead gives a max |delta| of 0.73.
