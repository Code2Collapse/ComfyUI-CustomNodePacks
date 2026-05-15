"""Smoke tests for Wan Director v1 nodes."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import wan_director as wd
from wan_director._common import parse_shotlist, DEFAULT_SHOTLIST_JSON

# Test 1: shotlist parse
shots = parse_shotlist(DEFAULT_SHOTLIST_JSON)
assert len(shots) == 3, shots
assert all("prompt" in s and "length" in s for s in shots)
print("OK shotlist parse:", len(shots), "shots")

# Test 2: WanShotListMEC
node = wd.WanShotListMEC()
out = node.build(DEFAULT_SHOTLIST_JSON)
assert isinstance(out[0], list) and len(out[0]) == 3
assert out[2] == 3
print("OK WanShotListMEC")

# Test 3: WanShotPickerMEC valid + out-of-range
picker = wd.WanShotPickerMEC()
prompt, neg, length, seed, w, h, cfg, steps = picker.pick(shots, 1)
assert "lone figure" in prompt
assert length == 81 and w == 832 and h == 480
# out of range
p2 = picker.pick(shots, 999)
assert "close-up" in p2[0].lower()
print("OK WanShotPickerMEC")

# Test 4: WanShotCountMEC
cnt = wd.WanShotCountMEC()
assert cnt.count(shots) == (3,)
assert cnt.count([]) == (0,)
print("OK WanShotCountMEC")

# Test 5: WanFrameBridgeMEC modes
br = wd.WanFrameBridgeMEC()
clip = torch.rand(8, 64, 64, 3)
last_frame = br.extract(clip, "last", 0, 3)[0]
assert last_frame.shape == (1, 64, 64, 3)
assert torch.allclose(last_frame[0], clip[7])
off_frame = br.extract(clip, "offset", 2, 3)[0]
assert torch.allclose(off_frame[0], clip[5])
avg_frame = br.extract(clip, "average", 0, 4)[0]
assert avg_frame.shape == (1, 64, 64, 3)
assert torch.allclose(avg_frame[0], clip[4:8].mean(dim=0), atol=1e-5)
print("OK WanFrameBridgeMEC (last/offset/average)")

# Test 6: WanShotConcatMEC no-xfade
cc = wd.WanShotConcatMEC()
a = torch.zeros(4, 32, 32, 3)
b = torch.ones(5, 32, 32, 3)
res = cc.concat(0, clip_1=a, clip_2=b)
assert res[0].shape[0] == 9 and res[1] == 9
print("OK WanShotConcatMEC no-xfade")

# Test 6b: WanShotConcatMEC with crossfade
a = torch.zeros(8, 32, 32, 3)
b = torch.ones(8, 32, 32, 3)
res = cc.concat(4, clip_1=a, clip_2=b)
# Expected length: 8 + 8 - 4 = 12
assert res[0].shape[0] == 12, f"expected 12 got {res[0].shape[0]}"
# Frame at the centre of the blend should be ~0.5
mid = res[0][6]  # middle of blend region
assert 0.2 < mid.mean().item() < 0.8, mid.mean().item()
print("OK WanShotConcatMEC xfade=4")

# Test 6c: ShotConcat with size mismatch (auto-resize)
a = torch.zeros(2, 32, 32, 3)
b = torch.ones(2, 64, 64, 3)
res = cc.concat(0, clip_1=a, clip_2=b)
assert res[0].shape == (4, 32, 32, 3)
print("OK WanShotConcatMEC auto-resize")

# Test 7: WanPromptScheduleMEC fizz
sc = wd.WanPromptScheduleMEC()
sched, total = sc.build(shots, "fizz", False)
assert '"0":' in sched and '"81":' in sched and '"162":' in sched
assert total == 81 * 3
print("OK WanPromptScheduleMEC fizz, total_frames=", total)

# Test 7b: simple style + use_negative
sched2, _ = sc.build(shots, "simple", True)
assert "0:" in sched2  # negs are empty by default but separators still present
print("OK WanPromptScheduleMEC simple+neg")

print("\nALL 12 WAN DIRECTOR TESTS PASSED")
