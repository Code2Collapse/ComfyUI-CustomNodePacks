"""
Integration test for LuminanceKeyerMEC – Option 4 deployment validation.

Tests with 4 real-scenario tensors:
  1. Pure white image  → highlights mask should be all-ones
  2. Pure black image  → highlights mask should be all-zeros
  3. Gradient image    → mask should grade smoothly
  4. Simulated dark video frame → auto mode should select highlights
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stub ComfyUI modules before importing node
for mod in ("folder_paths", "comfy", "comfy.model_management", "server"):
    if mod not in sys.modules:
        import types
        sys.modules[mod] = types.ModuleType(mod)

import torch
from nodes.luminance_keyer import LuminanceKeyerMEC

def run():
    node = LuminanceKeyerMEC()
    all_pass = True

    # ── Test 1: Pure white (1.0) ──────────────────────────────
    white = torch.ones(1, 256, 256, 3)
    mask, info = node.key_luminance(white, "highlights", 0.0, 1.0, 1.0, 1.0, False)
    wmin, wmax = mask.min().item(), mask.max().item()
    # Highlights mode: luminance=1.0 is above high=1.0 threshold → should be 1.0
    t1 = abs(wmin - 1.0) < 0.01 and abs(wmax - 1.0) < 0.01
    print(f"[{'PASS' if t1 else 'FAIL'}] Test 1: Pure white → highlights mask")
    print(f"       mask min={wmin:.4f}, max={wmax:.4f} (expected ~1.0)")
    print(f"       info: {info.replace(chr(10), ' | ')}")
    if not t1: all_pass = False

    # ── Test 2: Pure black (0.0) ──────────────────────────────
    black = torch.zeros(1, 256, 256, 3)
    mask, info = node.key_luminance(black, "highlights", 0.0, 1.0, 1.0, 1.0, False)
    bmin, bmax = mask.min().item(), mask.max().item()
    # Highlights: luminance=0.0 is at low=0.7 preset → should be 0.0
    t2 = abs(bmin) < 0.01 and abs(bmax) < 0.01
    print(f"\n[{'PASS' if t2 else 'FAIL'}] Test 2: Pure black → highlights mask")
    print(f"       mask min={bmin:.4f}, max={bmax:.4f} (expected ~0.0)")
    print(f"       info: {info.replace(chr(10), ' | ')}")
    if not t2: all_pass = False

    # ── Test 3: Horizontal gradient (0 → 1) ──────────────────
    grad = torch.linspace(0, 1, 256).unsqueeze(0).unsqueeze(0).expand(1, 256, 256)
    # grad shape: (1, 256, 256). Convert to (1, 256, 256, 3) by repeating across channels
    grad_img = grad.unsqueeze(-1).expand(1, 256, 256, 3)
    mask, info = node.key_luminance(grad_img, "custom", 0.2, 0.8, 1.0, 1.0, False)
    # Should have smooth gradient: left side (dark) → 0, right side (bright) → 1
    left_col = mask[0, :, 0].mean().item()    # luminance ~0.0, should be 0
    right_col = mask[0, :, -1].mean().item()  # luminance ~1.0, should be 1
    mid_col = mask[0, :, 128].mean().item()   # luminance ~0.5, should be intermediate
    unique_vals = mask.unique().numel()
    t3 = left_col < 0.05 and right_col > 0.95 and 0.1 < mid_col < 0.9 and unique_vals > 10
    print(f"\n[{'PASS' if t3 else 'FAIL'}] Test 3: Gradient → custom mode (0.2–0.8)")
    print(f"       left={left_col:.4f} (exp ~0), mid={mid_col:.4f} (exp 0.1–0.9), right={right_col:.4f} (exp ~1)")
    print(f"       unique values: {unique_vals} (exp >10, confirms smooth gradient)")
    print(f"       info: {info.replace(chr(10), ' | ')}")
    if not t3: all_pass = False

    # ── Test 4: Simulated dark video frame (mean ~0.15) ───────
    torch.manual_seed(42)
    dark_frame = torch.clamp(torch.randn(1, 480, 640, 3) * 0.08 + 0.15, 0, 1)
    mask, info = node.key_luminance(dark_frame, "auto", 0.0, 1.0, 1.0, 1.0, False)
    # Auto on dark image (mean~0.15 < 0.4) should select "highlights"
    auto_selected = "highlights" in info.lower()
    # The mask should be mostly zeros (few bright pixels in a dark frame)
    coverage = mask.mean().item()
    t4 = auto_selected and coverage < 0.2
    print(f"\n[{'PASS' if t4 else 'FAIL'}] Test 4: Dark video frame → auto mode")
    print(f"       auto-selected mode: {'highlights' if auto_selected else 'NOT highlights (!)'}")
    print(f"       mask coverage: {coverage*100:.1f}% (expected low, dark frame has few highlights)")
    print(f"       info: {info.replace(chr(10), ' | ')}")
    if not t4: all_pass = False

    # ── Test 5 (bonus): Invert verification ───────────────────
    mask_norm, _ = node.key_luminance(white, "highlights", 0.0, 1.0, 1.0, 1.0, False)
    mask_inv, _ = node.key_luminance(white, "highlights", 0.0, 1.0, 1.0, 1.0, True)
    diff = (mask_norm + mask_inv).mean().item()
    t5 = abs(diff - 1.0) < 0.01
    print(f"\n[{'PASS' if t5 else 'FAIL'}] Test 5: Invert complement check")
    print(f"       normal + inverted mean = {diff:.4f} (expected 1.0)")
    if not t5: all_pass = False

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    status = "ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED"
    print(f"  Integration result: {status}")
    print(f"{'='*60}")
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(run())
