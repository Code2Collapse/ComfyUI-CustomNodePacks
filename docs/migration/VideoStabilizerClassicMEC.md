# VideoStabilizerClassicMEC -> VideoStabilizerMEC (method = classic)

**Status:** removed on 2026-11-18 (L7.65 P17). Saved workflows and API prompts migrate automatically.
Setting `C2C_LEGACY_STABILIZERS=1` registers the old class again, unchanged. While it is registered, nothing
migrates it. It logs one warning per run that names the replacement.

| Old input / output | VideoStabilizerMEC | Conversion |
|---|---|---|
| frames | frames | link moved |
| frame_rate, framing_mode, transform_mode, camera_lock, strength, smooth, keep_fov, padding_color | same names | values copied |
| (none) | method | set to `classic` |
| (none) | preset | set to `manual`, so the manual values above are used |
| (none) | raft_iters, use_half, allow_download | successor defaults (not read by classic) |
| out 0 stabilized_frames, 1 padding_mask, 2 info | outputs 0, 1, 2 | links moved by index |
| (none) | output 3 motion_meta | new; nothing is linked to it |

Nothing is dropped.

**Parity test** (`test_removed_stabilizer_alias_migrates_with_bit_exact_output[VideoStabilizerClassicMEC]`):
- Setup: a saved node with widgets `[16.0, crop, translation, True, 0.9, 0.8, 0.7, "10, 20, 30"]`.
- It goes through the real front-end migration (Node) and through the API-prompt migration. Both give the same inputs.
- VideoStabilizerMEC runs on a 10-frame 72x96 shaky clip and is compared with the old class.
- Frames and padding mask: `torch.equal`, bit-exact.
- Info: identical text, apart from the elapsed time.
- The test is not trivial. The output differs from the input (mean |delta| 0.161), and the wrong method gives a different picture (max |delta| 0.73 for Flow vs classic).
