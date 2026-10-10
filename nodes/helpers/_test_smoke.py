"""Smoke tests for C2C helpers nodes."""
import sys, os
# the pack root, so nodes.helpers keeps its package (helpers.py imports ``.._is_changed_util``)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from nodes import helpers as H

# 1. Batch Range (range, split - was Image Batch Split - and single frames - was Video Frame Extractor)
n = H.helpers.ImageBatchSliceMEC()
clip = torch.arange(10 * 4 * 4 * 3, dtype=torch.float32).reshape(10, 4, 4, 3)
out, count, *_ = n.slice(clip, 2, 8, 1)
assert out.shape[0] == 6 and count == 6
out_neg, *_ = n.slice(clip, 0, -1, 2)
assert out_neg.shape[0] == 5  # 0,2,4,6,8
a, ac, r, rc, total, is_video = n.slice(clip, 0, -1, 1, "split at index", 3, 0.5)
assert ac == 3 and rc == 7 and total == 10 and is_video
a, ac, r, rc, *_ = n.slice(clip, 0, -1, 1, "split at ratio", 0, 0.3)
assert ac == 3 and rc == 7
f, fc, *_ = n.slice(clip, 0, -1, 1, "middle frame")
assert fc == 1 and torch.equal(f[0], clip[5])
f, *_ = n.slice(clip, 0, -1, 1, "frame at index", 1, 0.5, 99)
assert torch.equal(f[0], clip[9])  # clamped to the last frame
print("OK Batch Range")

# 3. MaskBatchCombine
n3 = H.helpers.MaskBatchCombineMEC()
a = torch.tensor([[1.0, 0.0], [1.0, 1.0]]).unsqueeze(0)  # (1,2,2)
b = torch.tensor([[0.0, 0.0], [1.0, 1.0]]).unsqueeze(0)
assert n3.combine(a, b, "union")[0].sum().item() == 3
assert n3.combine(a, b, "intersect")[0].sum().item() == 2
assert n3.combine(a, b, "diff")[0].sum().item() == 1
print("OK MaskBatchCombine")

# 4. SeedList
n4 = H.helpers.SeedListMEC()
first, csv = n4.build(42, 4, "increment")
assert first == 42 and csv == "42,43,44,45"
first_h, _ = n4.build(42, 4, "hash")
assert isinstance(first_h, int)
print("OK SeedList")

# 5. ConditionalSwitch
n5 = H.helpers.ConditionalSwitchMEC()
assert n5.pick(True, "yes", "no") == ("yes",)
assert n5.pick(False, "yes", "no") == ("no",)
print("OK ConditionalSwitch")

# 6. TextTemplate
n6 = H.helpers.TextTemplateMEC()
out, = n6.format("a {a} on {b}", "cat", "mat", "", "")
assert out == "a cat on mat"
# unbalanced braces don't crash
out2, = n6.format("hello {a} {", "world", "", "", "")
assert "world" in out2
print("OK TextTemplate")

# 7. NumberLerp
n7 = H.helpers.NumberLerpMEC()
f, i = n7.lerp(0.0, 10.0, 0.5, "linear")
assert f == 5.0 and i == 5
f2, _ = n7.lerp(0.0, 10.0, 0.5, "smoothstep")
assert f2 == 5.0  # smoothstep(0.5) = 0.5
f3, _ = n7.lerp(0.0, 10.0, 1.5, "linear")
assert f3 == 10.0  # clamped
print("OK NumberLerp")

# 7. Size (custom size snapping - was Dimensions Snap - and presets)
n9 = H.helpers.AspectPresetMEC()
C = n9.CUSTOM
w, h = n9.pick(C, 1024, 64, 1000, 600, "down")
assert w == 960 and h == 576, (w, h)
w2, h2 = n9.pick(C, 1024, 64, 1000, 600, "up")
assert w2 == 1024 and h2 == 640
w3, h3 = n9.pick(C, 1024, 64, 1000, 600, "nearest")
assert w3 == 1024 and h3 == 576
w, h = n9.pick("16:9 landscape", 1024, 64)
assert w == 1024 and h == 576, (w, h)
w, h = n9.pick("Wan 480p land", 0, 64)
assert w == 832 and h == 448, (w, h)  # 480 -> 448 after snap-down
w, h = n9.pick("1:1 square", 1024, 64)
assert w == 1024 and h == 1024
print("OK Size")

# 8. Probe (image stats, mask coverage - was Mask Area Probe - and latent - was VAE Latent Inspector)
n10 = H.helpers.ImageStatsProbeMEC()
img = torch.full((1, 8, 8, 3), 0.5)
img[0, 0, 0, 0] = 1.0
out, report, mean, std, bright = n10.probe(images=img)[:5]
assert out is img
assert abs(mean - 0.5) < 0.01
assert "mean=" in report
m = torch.zeros(2, 4, 4)
m[0, :2, :] = 1.0  # 8/16 = 50%
m[1, :, :] = 1.0   # 100%
mm, cmean, cmin, cmax = n10.probe(mask=m, threshold=0.5)[5:9]
assert mm is m and cmean == 75.0 and cmin == 50.0 and cmax == 100.0
print("OK Probe")

# 9. ExecutionTimer
n12 = H.helpers.ExecutionTimerMEC()
import time
out1 = n12.tick("payload", "test", False)
assert out1[2] == 0.0  # first tick
time.sleep(0.05)
out2 = n12.tick("payload", "test", False)
assert out2[2] > 0.04
out3 = n12.tick("payload", "test", True)
assert out3[2] == 0.0  # reset
print("OK ExecutionTimer")

print("\nALL 9 HELPER NODES PASSED")
print("Registered classes:", list(H.NODE_CLASS_MAPPINGS.keys()))
