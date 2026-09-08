"""Smoke test for VAEMergeMEC — synthetic VAE wrappers, every merge mode,
plus features A (auto-alpha), C (dry-run), D (recipe roundtrip).
Feature B (probe) needs real .encode/.decode; skipped here on purpose.
"""
import json, sys, traceback
sys.path.insert(0, r"D:\PROJECT\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-CustomNodePacks")

import torch
from nodes.vae_merge import VAEMergeMEC, MERGE_MODES


class FakeInner(torch.nn.Module):
    """Minimal SD-1.5-like VAE inner with the exact key prefixes the
    block-classifier expects."""
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder_conv_in = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.encoder_conv_out = torch.nn.Conv2d(4, 8, 3, padding=1)
        self.encoder_norm_out = torch.nn.GroupNorm(2, 8)
        self.encoder_down_0 = torch.nn.Conv2d(4, 4, 1)
        self.encoder_down_1 = torch.nn.Conv2d(4, 4, 1)
        self.encoder_down_2 = torch.nn.Conv2d(4, 4, 1)
        self.encoder_mid = torch.nn.Conv2d(4, 4, 1)
        self.decoder_conv_in = torch.nn.Conv2d(4, 4, 1)
        self.decoder_conv_out = torch.nn.Conv2d(4, 3, 3, padding=1)
        self.decoder_norm_out = torch.nn.GroupNorm(2, 4)
        self.decoder_up_0 = torch.nn.Conv2d(4, 4, 1)
        self.decoder_up_1 = torch.nn.Conv2d(4, 4, 1)
        self.decoder_up_2 = torch.nn.Conv2d(4, 4, 1)
        self.decoder_mid = torch.nn.Conv2d(4, 4, 1)

    def state_dict(self, *a, **k):
        # Rewrite to dotted keys the regex expects: encoder.conv_in.*
        sd = super().state_dict(*a, **k)
        out = {}
        for k_, v in sd.items():
            # encoder_conv_in.weight  ->  encoder.conv_in.weight
            new = k_.replace("encoder_conv_in", "encoder.conv_in")
            new = new.replace("encoder_conv_out", "encoder.conv_out")
            new = new.replace("encoder_norm_out", "encoder.norm_out")
            new = new.replace("encoder_down_0", "encoder.down.0")
            new = new.replace("encoder_down_1", "encoder.down.1")
            new = new.replace("encoder_down_2", "encoder.down.2")
            new = new.replace("encoder_mid", "encoder.mid")
            new = new.replace("decoder_conv_in", "decoder.conv_in")
            new = new.replace("decoder_conv_out", "decoder.conv_out")
            new = new.replace("decoder_norm_out", "decoder.norm_out")
            new = new.replace("decoder_up_0", "decoder.up.0")
            new = new.replace("decoder_up_1", "decoder.up.1")
            new = new.replace("decoder_up_2", "decoder.up.2")
            new = new.replace("decoder_mid", "decoder.mid")
            out[new] = v
        return out

    def load_state_dict(self, sd, strict=True):
        # Stub — accept anything, return a NamedTuple-like object
        class R:
            missing_keys = []
            unexpected_keys = []
        return R()


class FakeVAE:
    def __init__(self, seed):
        self.first_stage_model = FakeInner(seed)


def main():
    vae_a = FakeVAE(1)
    vae_b = FakeVAE(2)
    vae_c = FakeVAE(3)

    node = VAEMergeMEC()
    failures = []
    for mode in MERGE_MODES:
        try:
            kwargs = dict(
                vae_a=vae_a, vae_b=vae_b, merge_mode=mode,
                alpha=0.4, beta=0.6, brightness=0.05, contrast=0.05,
                use_blocks=True,
                block_conv_in=0.3, block_conv_out=0.7, block_norm_out=0.5,
                block_0=0.2, block_1=0.4, block_2=0.6, block_3=0.5, block_mid=0.5,
                device="cpu", dry_run=False,
            )
            if mode in ("add_difference", "triple_sum", "smooth_add_diff"):
                kwargs["vae_c"] = vae_c
            out_vae, info_s, recipe_s, probe_s = node.merge(**kwargs)
            info = json.loads(info_s)
            assert info["status"] == "ok", f"{mode}: status={info['status']}"
            assert info["merge_mode"] == mode
            assert "block_similarity" in info
            print(f"  OK  {mode:18s}  arch={info['architecture']}  keys={info['key_count']}")
        except Exception as e:
            failures.append((mode, str(e)))
            print(f"  FAIL {mode}: {e}")
            traceback.print_exc()

    # Feature C: dry_run
    try:
        out_vae, info_s, recipe_s, probe_s = node.merge(
            vae_a=vae_a, vae_b=vae_b, merge_mode="weighted_sum",
            alpha=0.5, dry_run=True,
        )
        info = json.loads(info_s)
        assert info["status"] == "dry_run"
        assert out_vae is vae_a
        recipe = json.loads(recipe_s)
        assert recipe["schema"] == "mec.vae_merge.recipe.v1"
        print("  OK  dry_run                returns vae_a + recipe")
    except Exception as e:
        failures.append(("dry_run", str(e))); print(f"  FAIL dry_run: {e}")

    # Feature A: auto_alpha
    try:
        out_vae, info_s, recipe_s, probe_s = node.merge(
            vae_a=vae_a, vae_b=vae_b, merge_mode="weighted_sum",
            alpha=0.5, auto_alpha=True, dry_run=True,
        )
        info = json.loads(info_s)
        assert info["auto_alpha"] is True
        assert "auto_alpha_blocks" in info
        print(f"  OK  auto_alpha             {len(info['auto_alpha_blocks'])} blocks computed")
    except Exception as e:
        failures.append(("auto_alpha", str(e))); print(f"  FAIL auto_alpha: {e}")

    # Feature D: recipe roundtrip
    try:
        _, _, recipe_s, _ = node.merge(
            vae_a=vae_a, vae_b=vae_b, merge_mode="sigmoid",
            alpha=0.77, beta=0.33, dry_run=True,
        )
        out_vae, info_s, _, _ = node.merge(
            vae_a=vae_a, vae_b=vae_b, merge_mode="weighted_sum",
            alpha=0.0, beta=0.0, dry_run=True, recipe_in=recipe_s,
        )
        info = json.loads(info_s)
        assert info["merge_mode"] == "sigmoid", f"recipe override failed: {info['merge_mode']}"
        assert abs(info["alpha"] - 0.77) < 1e-6
        assert info["recipe_loaded"] is True
        print("  OK  recipe roundtrip       widgets overridden by recipe_in")
    except Exception as e:
        failures.append(("recipe_roundtrip", str(e))); print(f"  FAIL recipe: {e}")

    print()
    if failures:
        print(f"FAIL: {len(failures)} test(s) failed: {[f[0] for f in failures]}")
        sys.exit(1)
    print(f"ALL PASS ({len(MERGE_MODES)} modes + 3 feature tests)")


if __name__ == "__main__":
    main()
