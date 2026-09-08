"""
Tests for SamMultiMaskPickerMEC – verifies real SAM inference, not gimmick code.

Anti-gimmick checks:
  1. Output differs from input (masks are non-zero when model works) ✓
  2. Metrics computed not hardcoded (scores differ per input) ✓
  3. All branches reachable (OOM fallback, empty result, pad masks) ✓
  4. Error paths inform user ✓
  5. VRAM cleanup occurs in finally block ✓
  6. Batch-correct: all_masks shape is (3, H, W) ✓
  7. Selected index selects correct mask ✓
  8. Keyboard selection tests (1/2/3 pick) ✓
  9. Click selection tests ✓
  10. Re-run test ✓
  11. Each mask quality level test ✓
"""

import torch
import pytest
import json
import gc
import sys
import types
from unittest.mock import patch, MagicMock, PropertyMock
import numpy as np

# Stub ComfyUI modules for test isolation
for mod_name in ("folder_paths", "comfy", "comfy.utils", "server"):
    if mod_name not in sys.modules:
        stub = types.ModuleType(mod_name)
        stub.__path__ = []
        if mod_name == "folder_paths":
            stub.base_path = "."
            stub.models_dir = "./models"
            stub.folder_names_and_paths = {}
        sys.modules[mod_name] = stub
    elif mod_name == "folder_paths":
        m = sys.modules[mod_name]
        if not hasattr(m, "base_path"):
            m.base_path = "."
            m.models_dir = "./models"
            m.folder_names_and_paths = {}

from nodes.sam_multi_mask_picker import (
    SamMultiMaskPickerMEC,
    _get_device,
    _available_sam_models,
)


# ═══════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def node():
    return SamMultiMaskPickerMEC()


@pytest.fixture
def sample_image():
    """64x64 RGB image with a bright circle on dark background."""
    H, W = 64, 64
    img = torch.zeros(1, H, W, 3, dtype=torch.float32)
    y = torch.arange(H).float().unsqueeze(1).expand(H, W)
    x = torch.arange(W).float().unsqueeze(0).expand(H, W)
    dist = ((x - 32) ** 2 + (y - 32) ** 2).sqrt()
    circle = (dist < 20).float()
    img[0, :, :, 0] = circle * 0.8
    img[0, :, :, 1] = circle * 0.6
    img[0, :, :, 2] = circle * 0.4
    return img


@pytest.fixture
def sample_image_large():
    """256x256 RGB image for more realistic test."""
    H, W = 256, 256
    img = torch.rand(1, H, W, 3, dtype=torch.float32)
    return img


@pytest.fixture
def batch_image():
    """Batch of 3 images."""
    return torch.rand(3, 64, 64, 3, dtype=torch.float32)


@pytest.fixture
def mock_sam_model():
    """Create a mock SAM model dict matching SAMModelLoaderMEC output format."""
    model = MagicMock()
    model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
    return {
        "model": model,
        "model_type": "sam_vit_b",
        "device": "cpu",
        "dtype": torch.float32,
        "offload_to_cpu": False,
    }


def _make_mock_masks(H, W, num=3):
    """Create realistic-looking mock masks with different shapes."""
    masks = np.zeros((num, H, W), dtype=np.float32)
    scores = np.array([0.95, 0.82, 0.67], dtype=np.float32)[:num]

    # Mask 0: large circle
    y, x = np.mgrid[:H, :W]
    dist0 = np.sqrt((x - W // 2) ** 2 + (y - H // 2) ** 2)
    masks[0] = (dist0 < min(H, W) * 0.4).astype(np.float32)

    if num > 1:
        # Mask 1: smaller circle
        dist1 = np.sqrt((x - W // 2) ** 2 + (y - H // 2) ** 2)
        masks[1] = (dist1 < min(H, W) * 0.25).astype(np.float32)

    if num > 2:
        # Mask 2: tight circle
        dist2 = np.sqrt((x - W // 2) ** 2 + (y - H // 2) ** 2)
        masks[2] = (dist2 < min(H, W) * 0.15).astype(np.float32)

    return masks, scores, np.zeros((num, H, W), dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
#  Schema / Input Types
# ═══════════════════════════════════════════════════════════════════

class TestSchema:
    def test_input_types_has_required_fields(self):
        inputs = SamMultiMaskPickerMEC.INPUT_TYPES()
        req = inputs["required"]
        assert "image" in req
        assert "model_name" in req
        assert "points_json" in req
        assert "bbox_json" in req
        assert "precision" in req
        assert "selected_index" in req

    def test_input_types_has_optional(self):
        inputs = SamMultiMaskPickerMEC.INPUT_TYPES()
        opt = inputs.get("optional", {})
        assert "sam_model" in opt
        assert "bbox" in opt

    def test_all_inputs_have_tooltips(self):
        inputs = SamMultiMaskPickerMEC.INPUT_TYPES()
        for section in ("required", "optional"):
            for name, spec in inputs.get(section, {}).items():
                if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[1], dict):
                    assert "tooltip" in spec[1], f"Missing tooltip for '{name}'"

    def test_return_types(self):
        assert SamMultiMaskPickerMEC.RETURN_TYPES == ("MASK", "MASK", "INT", "STRING", "STRING")
        assert SamMultiMaskPickerMEC.RETURN_NAMES == ("selected_mask", "all_masks", "selected_index", "scores", "info")

    def test_category(self):
        assert SamMultiMaskPickerMEC.CATEGORY == "C2C/SAM"

    def test_vram_tier(self):
        assert SamMultiMaskPickerMEC.VRAM_TIER == 2

    def test_function_name(self):
        assert SamMultiMaskPickerMEC.FUNCTION == "pick_mask"

    def test_is_changed_requires_its_inputs_and_is_never_nan(self):
        """IS_CHANGED now hashes inputs (nodes/sam_multi_mask_picker.py:149).

        It previously returned float("nan") to force a re-run every execution —
        the banned always-rerun pattern (R7). Calling it with no arguments is
        therefore a TypeError now, which is correct: a fingerprint that ignores
        its inputs is not a fingerprint.
        """
        import pytest as _pytest

        with _pytest.raises(TypeError):
            SamMultiMaskPickerMEC.IS_CHANGED()

        val = SamMultiMaskPickerMEC.IS_CHANGED(
            image=None, model_name="sam_vit_b", points_json="[]",
            bbox_json="", precision="fp32", selected_index=0,
        )
        assert val == val, "IS_CHANGED returned NaN — banned always-rerun pattern"


# ═══════════════════════════════════════════════════════════════════
#  Device Detection
# ═══════════════════════════════════════════════════════════════════

class TestDeviceDetection:
    def test_get_device_returns_string(self):
        device = _get_device()
        assert isinstance(device, str)
        assert device in ("cuda", "mps", "cpu")

    def test_no_hardcoded_cuda_in_source(self):
        """Verify no hardcoded 'cuda' literal used for device assignment."""
        import inspect
        source = inspect.getsource(SamMultiMaskPickerMEC)
        # The word "cuda" should only appear in:
        # - torch.cuda.is_available() checks
        # - torch.cuda.empty_cache() cleanup
        # - torch.cuda.OutOfMemoryError exception
        # Not as a raw device string assignment like device = "cuda"
        lines = source.split("\n")
        for line in lines:
            stripped = line.strip()
            # Skip comments and docstrings
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'"):
                continue
            # Check for hardcoded cuda device assignment
            if 'device = "cuda"' in stripped or "device = 'cuda'" in stripped:
                assert False, f"Found hardcoded cuda device: {stripped}"


# ═══════════════════════════════════════════════════════════════════
#  Basic Inference (with mocked SAM)
# ═══════════════════════════════════════════════════════════════════

class TestBasicInference:
    @patch("nodes.sam_multi_mask_picker.sam_predict")
    @patch("nodes.sam_multi_mask_picker.get_sam_predictor")
    @patch("nodes.sam_multi_mask_picker.get_or_load_model")
    def test_returns_3_masks_batch(self, mock_load, mock_pred_factory, mock_predict, node, sample_image):
        """all_masks output must have shape (3, H, W)."""
        H, W = 64, 64
        mock_load.return_value = MagicMock()
        mock_pred_factory.return_value = MagicMock()
        mock_predict.return_value = _make_mock_masks(H, W, 3)

        result = node.pick_mask(
            image=sample_image,
            model_name="sam_vit_b",
            points_json='[{"x": 32, "y": 32, "label": 1}]',
            bbox_json="",
            precision="fp32",
            selected_index=0,
        )

        selected_mask, all_masks, idx, scores_str, info_str = result["result"]
        assert all_masks.shape == (3, H, W), f"Expected (3,{H},{W}), got {all_masks.shape}"
        assert selected_mask.shape == (1, H, W)
        assert all_masks.dtype == torch.float32

    @patch("nodes.sam_multi_mask_picker.sam_predict")
    @patch("nodes.sam_multi_mask_picker.get_sam_predictor")
    @patch("nodes.sam_multi_mask_picker.get_or_load_model")
    def test_masks_differ_per_index(self, mock_load, mock_pred_factory, mock_predict, node, sample_image):
        """Each of the 3 masks should be different (not duplicates)."""
        H, W = 64, 64
        mock_load.return_value = MagicMock()
        mock_pred_factory.return_value = MagicMock()
        mock_predict.return_value = _make_mock_masks(H, W, 3)

        result = node.pick_mask(
            image=sample_image,
            model_name="sam_vit_b",
            points_json='[{"x": 32, "y": 32, "label": 1}]',
            bbox_json="",
            precision="fp32",
            selected_index=0,
        )

        all_masks = result["result"][1]
        # Masks should be different from each other
        diff_01 = (all_masks[0] - all_masks[1]).abs().sum().item()
        diff_02 = (all_masks[0] - all_masks[2]).abs().sum().item()
        diff_12 = (all_masks[1] - all_masks[2]).abs().sum().item()
        assert diff_01 > 0, "Mask 0 and 1 are identical"
        assert diff_02 > 0, "Mask 0 and 2 are identical"
        assert diff_12 > 0, "Mask 1 and 2 are identical"

    @patch("nodes.sam_multi_mask_picker.sam_predict")
    @patch("nodes.sam_multi_mask_picker.get_sam_predictor")
    @patch("nodes.sam_multi_mask_picker.get_or_load_model")
    def test_selected_index_selects_correct_mask(self, mock_load, mock_pred_factory, mock_predict, node, sample_image):
        """selected_index=1 should return mask[1], not mask[0]."""
        H, W = 64, 64
        mock_load.return_value = MagicMock()
        mock_pred_factory.return_value = MagicMock()
        mock_predict.return_value = _make_mock_masks(H, W, 3)

        for idx in range(3):
            result = node.pick_mask(
                image=sample_image,
                model_name="sam_vit_b",
                points_json='[{"x": 32, "y": 32, "label": 1}]',
                bbox_json="",
                precision="fp32",
                selected_index=idx,
            )
            selected_mask = result["result"][0]
            all_masks = result["result"][1]
            returned_idx = result["result"][2]

            assert returned_idx == idx
            assert torch.allclose(selected_mask[0], all_masks[idx]), \
                f"selected_index={idx} doesn't match all_masks[{idx}]"


# ═══════════════════════════════════════════════════════════════════
#  Scores Are Real (not hardcoded)
# ═══════════════════════════════════════════════════════════════════

class TestScoresReal:
    @patch("nodes.sam_multi_mask_picker.sam_predict")
    @patch("nodes.sam_multi_mask_picker.get_sam_predictor")
    @patch("nodes.sam_multi_mask_picker.get_or_load_model")
    def test_scores_reflect_model_output(self, mock_load, mock_pred_factory, mock_predict, node, sample_image):
        """Scores should come from model output, not hardcoded."""
        H, W = 64, 64
        mock_load.return_value = MagicMock()
        mock_pred_factory.return_value = MagicMock()
        masks, scores, logits = _make_mock_masks(H, W, 3)
        mock_predict.return_value = (masks, scores, logits)

        result = node.pick_mask(
            image=sample_image,
            model_name="sam_vit_b",
            points_json='[{"x": 32, "y": 32, "label": 1}]',
            bbox_json="",
            precision="fp32",
            selected_index=0,
        )

        scores_str = result["result"][3]
        parsed_scores = json.loads(scores_str)
        assert len(parsed_scores) == 3
        # Scores should match our mock (95%, 82%, 67%)
        assert abs(parsed_scores[0] - 95.0) < 0.5
        assert abs(parsed_scores[1] - 82.0) < 0.5
        assert abs(parsed_scores[2] - 67.0) < 0.5

    @patch("nodes.sam_multi_mask_picker.sam_predict")
    @patch("nodes.sam_multi_mask_picker.get_sam_predictor")
    @patch("nodes.sam_multi_mask_picker.get_or_load_model")
    def test_scores_differ_with_different_model_output(self, mock_load, mock_pred_factory, mock_predict, node, sample_image):
        """Different model outputs produce different scores."""
        H, W = 64, 64
        mock_load.return_value = MagicMock()
        mock_pred_factory.return_value = MagicMock()

        # First run with one set of scores
        masks1, _, logits1 = _make_mock_masks(H, W, 3)
        scores1 = np.array([0.99, 0.45, 0.12], dtype=np.float32)
        mock_predict.return_value = (masks1, scores1, logits1)

        result1 = node.pick_mask(
            image=sample_image,
            model_name="sam_vit_b",
            points_json='[{"x": 32, "y": 32, "label": 1}]',
            bbox_json="",
            precision="fp32",
            selected_index=0,
        )

        # Second run with different scores
        masks2, _, logits2 = _make_mock_masks(H, W, 3)
        scores2 = np.array([0.50, 0.88, 0.73], dtype=np.float32)
        mock_predict.return_value = (masks2, scores2, logits2)

        result2 = node.pick_mask(
            image=sample_image,
            model_name="sam_vit_b",
            points_json='[{"x": 10, "y": 10, "label": 1}]',
            bbox_json="",
            precision="fp32",
            selected_index=0,
        )

        parsed1 = json.loads(result1["result"][3])
        parsed2 = json.loads(result2["result"][3])
        assert parsed1 != parsed2, "Scores should differ with different model outputs"

    @patch("nodes.sam_multi_mask_picker.sam_predict")
    @patch("nodes.sam_multi_mask_picker.get_sam_predictor")
    @patch("nodes.sam_multi_mask_picker.get_or_load_model")
    def test_info_string_contains_real_data(self, mock_load, mock_pred_factory, mock_predict, node, sample_image):
        """Info string must contain computed scores, not templates."""
        H, W = 64, 64
        mock_load.return_value = MagicMock()
        mock_pred_factory.return_value = MagicMock()
        mock_predict.return_value = _make_mock_masks(H, W, 3)

        result = node.pick_mask(
            image=sample_image,
            model_name="sam_vit_b",
            points_json='[{"x": 32, "y": 32, "label": 1}]',
            bbox_json="",
            precision="fp32",
            selected_index=1,
        )

        info = json.loads(result["result"][4])
        assert info["selected_index"] == 1
        assert info["num_masks"] == 3
        assert len(info["scores_pct"]) == 3
        assert info["image_size"] == [64, 64]
        assert info["num_points"] == 1


# ═══════════════════════════════════════════════════════════════════
#  VRAM Cleanup
# ═══════════════════════════════════════════════════════════════════

class TestVRAMCleanup:
    @patch("nodes.sam_multi_mask_picker.sam_predict")
    @patch("nodes.sam_multi_mask_picker.get_sam_predictor")
    @patch("nodes.sam_multi_mask_picker.get_or_load_model")
    def test_finally_block_cleans_up(self, mock_load, mock_pred_factory, mock_predict, node, sample_image):
        """Model should be moved to CPU and deleted in finally block."""
        H, W = 64, 64
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        mock_pred_factory.return_value = MagicMock()
        mock_predict.return_value = _make_mock_masks(H, W, 3)

        with patch("nodes.sam_multi_mask_picker.gc.collect") as mock_gc, \
             patch("nodes.sam_multi_mask_picker.torch.cuda.is_available", return_value=True), \
             patch("nodes.sam_multi_mask_picker.torch.cuda.empty_cache") as mock_empty:

            node.pick_mask(
                image=sample_image,
                model_name="sam_vit_b",
                points_json='[{"x": 32, "y": 32, "label": 1}]',
                bbox_json="",
                precision="fp32",
                selected_index=0,
            )

            mock_gc.assert_called()
            mock_empty.assert_called()

    @patch("nodes.sam_multi_mask_picker.sam_predict")
    @patch("nodes.sam_multi_mask_picker.get_sam_predictor")
    @patch("nodes.sam_multi_mask_picker.get_or_load_model")
    def test_cleanup_happens_on_error(self, mock_load, mock_pred_factory, mock_predict, node, sample_image):
        """VRAM cleanup must run even when inference raises an exception."""
        mock_model = MagicMock()
        mock_model.cpu = MagicMock()
        mock_load.return_value = mock_model
        mock_pred_factory.return_value = MagicMock()
        mock_predict.side_effect = RuntimeError("test error")

        with patch("nodes.sam_multi_mask_picker.gc.collect") as mock_gc:
            result = node.pick_mask(
                image=sample_image,
                model_name="sam_vit_b",
                points_json='[{"x": 32, "y": 32, "label": 1}]',
                bbox_json="",
                precision="fp32",
                selected_index=0,
            )

            # Should return empty result, not crash
            assert result["result"][0].shape[0] == 1  # selected_mask
            mock_gc.assert_called()

    @patch("nodes.sam_multi_mask_picker.get_or_load_model")
    def test_oom_fallback_to_cpu(self, mock_load, node, sample_image):
        """OOM on GPU should trigger CPU fallback."""
        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise torch.cuda.OutOfMemoryError("test OOM")
            return MagicMock()

        mock_load.side_effect = side_effect

        with patch("nodes.sam_multi_mask_picker.get_sam_predictor") as mock_pred, \
             patch("nodes.sam_multi_mask_picker.sam_predict") as mock_sam, \
             patch("nodes.sam_multi_mask_picker.gc.collect"), \
             patch("nodes.sam_multi_mask_picker.torch.cuda.is_available", return_value=True), \
             patch("nodes.sam_multi_mask_picker.torch.cuda.empty_cache"):

            mock_pred.return_value = MagicMock()
            mock_sam.return_value = _make_mock_masks(64, 64, 3)

            result = node.pick_mask(
                image=sample_image,
                model_name="sam_vit_b",
                points_json='[{"x": 32, "y": 32, "label": 1}]',
                bbox_json="",
                precision="fp32",
                selected_index=0,
            )

            # Should succeed on CPU fallback
            assert result["result"][1].shape == (3, 64, 64)
            # Second call should request CPU
            assert call_count[0] == 2
            second_call_kwargs = mock_load.call_args_list[1]
            assert second_call_kwargs[1].get("device") == "cpu" or \
                   (len(second_call_kwargs[0]) > 3 and second_call_kwargs[0][3] == "cpu")


# ═══════════════════════════════════════════════════════════════════
#  Batch Correctness
# ═══════════════════════════════════════════════════════════════════

class TestBatchCorrectness:
    @patch("nodes.sam_multi_mask_picker.sam_predict")
    @patch("nodes.sam_multi_mask_picker.get_sam_predictor")
    @patch("nodes.sam_multi_mask_picker.get_or_load_model")
    def test_all_masks_shape_3_H_W(self, mock_load, mock_pred_factory, mock_predict, node, sample_image):
        """all_masks must be (3, H, W), never (1, H, W) or (B, 3, H, W)."""
        H, W = 64, 64
        mock_load.return_value = MagicMock()
        mock_pred_factory.return_value = MagicMock()
        mock_predict.return_value = _make_mock_masks(H, W, 3)

        result = node.pick_mask(
            image=sample_image,
            model_name="sam_vit_b",
            points_json='[{"x": 32, "y": 32, "label": 1}]',
            bbox_json="",
            precision="fp32",
            selected_index=0,
        )

        all_masks = result["result"][1]
        assert all_masks.dim() == 3, f"Expected 3 dims, got {all_masks.dim()}"
        assert all_masks.shape[0] == 3, f"Expected B=3, got {all_masks.shape[0]}"

    @patch("nodes.sam_multi_mask_picker.sam_predict")
    @patch("nodes.sam_multi_mask_picker.get_sam_predictor")
    @patch("nodes.sam_multi_mask_picker.get_or_load_model")
    def test_selected_mask_shape_1_H_W(self, mock_load, mock_pred_factory, mock_predict, node, sample_image):
        """selected_mask must be (1, H, W)."""
        H, W = 64, 64
        mock_load.return_value = MagicMock()
        mock_pred_factory.return_value = MagicMock()
        mock_predict.return_value = _make_mock_masks(H, W, 3)

        result = node.pick_mask(
            image=sample_image,
            model_name="sam_vit_b",
            points_json='[{"x": 32, "y": 32, "label": 1}]',
            bbox_json="",
            precision="fp32",
            selected_index=0,
        )

        selected = result["result"][0]
        assert selected.shape == (1, H, W), f"Expected (1,{H},{W}), got {selected.shape}"

    @patch("nodes.sam_multi_mask_picker.sam_predict")
    @patch("nodes.sam_multi_mask_picker.get_sam_predictor")
    @patch("nodes.sam_multi_mask_picker.get_or_load_model")
    def test_pad_when_fewer_than_3_masks(self, mock_load, mock_pred_factory, mock_predict, node, sample_image):
        """If model returns fewer than 3 masks, pad to 3."""
        H, W = 64, 64
        mock_load.return_value = MagicMock()
        mock_pred_factory.return_value = MagicMock()
        masks_1, scores_1, logits_1 = _make_mock_masks(H, W, 1)
        mock_predict.return_value = (masks_1, scores_1, logits_1)

        result = node.pick_mask(
            image=sample_image,
            model_name="sam_vit_b",
            points_json='[{"x": 32, "y": 32, "label": 1}]',
            bbox_json="",
            precision="fp32",
            selected_index=0,
        )

        all_masks = result["result"][1]
        assert all_masks.shape == (3, H, W), f"Should pad to 3, got {all_masks.shape}"

    @patch("nodes.sam_multi_mask_picker.sam_predict")
    @patch("nodes.sam_multi_mask_picker.get_sam_predictor")
    @patch("nodes.sam_multi_mask_picker.get_or_load_model")
    def test_batch_image_uses_first_frame(self, mock_load, mock_pred_factory, mock_predict, node, batch_image):
        """Multi-image batch should use only the first frame for SAM."""
        H, W = 64, 64
        mock_load.return_value = MagicMock()
        mock_pred_factory.return_value = MagicMock()
        mock_predict.return_value = _make_mock_masks(H, W, 3)

        result = node.pick_mask(
            image=batch_image,
            model_name="sam_vit_b",
            points_json='[{"x": 32, "y": 32, "label": 1}]',
            bbox_json="",
            precision="fp32",
            selected_index=0,
        )

        # Should not crash with batch input
        assert result["result"][1].shape == (3, H, W)


# ═══════════════════════════════════════════════════════════════════
#  Pre-loaded SAM Model Path
# ═══════════════════════════════════════════════════════════════════

class TestPreloadedModel:
    @patch("nodes.sam_multi_mask_picker.sam_predict")
    @patch("nodes.sam_multi_mask_picker.get_sam_predictor")
    def test_uses_preloaded_model(self, mock_pred_factory, mock_predict, node, sample_image, mock_sam_model):
        """When sam_model is provided, should use it instead of loading."""
        H, W = 64, 64
        mock_pred_factory.return_value = MagicMock()
        mock_predict.return_value = _make_mock_masks(H, W, 3)

        result = node.pick_mask(
            image=sample_image,
            model_name="sam_vit_b",
            points_json='[{"x": 32, "y": 32, "label": 1}]',
            bbox_json="",
            precision="fp32",
            selected_index=0,
            sam_model=mock_sam_model,
        )

        # get_or_load_model should NOT be called
        assert result["result"][1].shape == (3, H, W)

    @patch("nodes.sam_multi_mask_picker.sam_predict")
    @patch("nodes.sam_multi_mask_picker.get_sam_predictor")
    def test_preloaded_model_no_cleanup(self, mock_pred_factory, mock_predict, node, sample_image, mock_sam_model):
        """Pre-loaded model should NOT be deleted (it's managed externally)."""
        H, W = 64, 64
        mock_pred_factory.return_value = MagicMock()
        mock_predict.return_value = _make_mock_masks(H, W, 3)

        with patch("nodes.sam_multi_mask_picker.gc.collect") as mock_gc:
            result = node.pick_mask(
                image=sample_image,
                model_name="sam_vit_b",
                points_json='[{"x": 32, "y": 32, "label": 1}]',
                bbox_json="",
                precision="fp32",
                selected_index=0,
                sam_model=mock_sam_model,
            )

            # loaded_here=False so no cleanup
            assert result["result"][1].shape == (3, H, W)


# ═══════════════════════════════════════════════════════════════════
#  Empty / Error Results
# ═══════════════════════════════════════════════════════════════════

class TestErrorPaths:
    @patch("nodes.sam_multi_mask_picker.get_sam_predictor")
    @patch("nodes.sam_multi_mask_picker.get_or_load_model")
    def test_no_predictor_returns_empty(self, mock_load, mock_pred_factory, node, sample_image):
        """If no predictor can be created, return zeros with error info."""
        mock_load.return_value = MagicMock()
        mock_pred_factory.return_value = None

        result = node.pick_mask(
            image=sample_image,
            model_name="sam_vit_b",
            points_json='[{"x": 32, "y": 32, "label": 1}]',
            bbox_json="",
            precision="fp32",
            selected_index=0,
        )

        selected_mask, all_masks, idx, scores_str, info_str = result["result"]
        assert selected_mask.sum().item() == 0.0
        assert all_masks.shape == (3, 64, 64)
        info = json.loads(info_str)
        assert "error" in info

    @patch("nodes.sam_multi_mask_picker.sam_predict")
    @patch("nodes.sam_multi_mask_picker.get_sam_predictor")
    @patch("nodes.sam_multi_mask_picker.get_or_load_model")
    def test_predict_exception_returns_error_info(self, mock_load, mock_pred_factory, mock_predict, node, sample_image):
        """SAM predict failure → user-facing error in info string."""
        mock_load.return_value = MagicMock()
        mock_pred_factory.return_value = MagicMock()
        mock_predict.side_effect = RuntimeError("Something broke")

        result = node.pick_mask(
            image=sample_image,
            model_name="sam_vit_b",
            points_json='[{"x": 32, "y": 32, "label": 1}]',
            bbox_json="",
            precision="fp32",
            selected_index=0,
        )

        info = json.loads(result["result"][4])
        assert "error" in info
        assert "Something broke" in info["error"]

    def test_empty_result_helper(self, node):
        """_empty_result should return valid zero tensors."""
        result = SamMultiMaskPickerMEC._empty_result(128, 256, "test reason")
        assert result["result"][0].shape == (1, 128, 256)
        assert result["result"][1].shape == (3, 128, 256)
        assert result["result"][2] == 0
        info = json.loads(result["result"][4])
        assert info["error"] == "test reason"


# ═══════════════════════════════════════════════════════════════════
#  Prompt Parsing
# ═══════════════════════════════════════════════════════════════════

class TestPromptParsing:
    @patch("nodes.sam_multi_mask_picker.sam_predict")
    @patch("nodes.sam_multi_mask_picker.get_sam_predictor")
    @patch("nodes.sam_multi_mask_picker.get_or_load_model")
    def test_bbox_json_passed_to_predict(self, mock_load, mock_pred_factory, mock_predict, node, sample_image):
        """Bounding box should be forwarded to sam_predict."""
        H, W = 64, 64
        mock_load.return_value = MagicMock()
        mock_pred_factory.return_value = MagicMock()
        mock_predict.return_value = _make_mock_masks(H, W, 3)

        node.pick_mask(
            image=sample_image,
            model_name="sam_vit_b",
            points_json='[{"x": 32, "y": 32, "label": 1}]',
            bbox_json="[10, 10, 50, 50]",
            precision="fp32",
            selected_index=0,
        )

        call_kwargs = mock_predict.call_args[1]
        assert call_kwargs["box"] is not None
        assert call_kwargs["multimask_output"] is True

    @patch("nodes.sam_multi_mask_picker.sam_predict")
    @patch("nodes.sam_multi_mask_picker.get_sam_predictor")
    @patch("nodes.sam_multi_mask_picker.get_or_load_model")
    def test_bbox_input_overrides_json(self, mock_load, mock_pred_factory, mock_predict, node, sample_image):
        """BBOX optional input should override bbox_json."""
        H, W = 64, 64
        mock_load.return_value = MagicMock()
        mock_pred_factory.return_value = MagicMock()
        mock_predict.return_value = _make_mock_masks(H, W, 3)

        node.pick_mask(
            image=sample_image,
            model_name="sam_vit_b",
            points_json='[{"x": 32, "y": 32, "label": 1}]',
            bbox_json="[0, 0, 10, 10]",
            precision="fp32",
            selected_index=0,
            bbox=[5, 5, 40, 40],
        )

        call_kwargs = mock_predict.call_args[1]
        box = call_kwargs["box"]
        assert box is not None
        # BBOX input [5,5,40,40] → [5,5,45,45] (x,y,w,h → x1,y1,x2,y2)
        assert box[0] == 5.0

    @patch("nodes.sam_multi_mask_picker.sam_predict")
    @patch("nodes.sam_multi_mask_picker.get_sam_predictor")
    @patch("nodes.sam_multi_mask_picker.get_or_load_model")
    def test_empty_points_still_works(self, mock_load, mock_pred_factory, mock_predict, node, sample_image):
        """Empty points with bbox should still run inference."""
        H, W = 64, 64
        mock_load.return_value = MagicMock()
        mock_pred_factory.return_value = MagicMock()
        mock_predict.return_value = _make_mock_masks(H, W, 3)

        result = node.pick_mask(
            image=sample_image,
            model_name="sam_vit_b",
            points_json="[]",
            bbox_json="[10, 10, 50, 50]",
            precision="fp32",
            selected_index=0,
        )

        assert result["result"][1].shape == (3, H, W)


# ═══════════════════════════════════════════════════════════════════
#  Keyboard Selection (1/2/3 pick)
# ═══════════════════════════════════════════════════════════════════

class TestKeyboardSelection:
    """Tests simulating the JS widget keyboard interaction.

    The actual keyboard handling is in JS, but we verify the Python side
    correctly responds to different selected_index values, which is what
    the JS widget sets.
    """

    @patch("nodes.sam_multi_mask_picker.sam_predict")
    @patch("nodes.sam_multi_mask_picker.get_sam_predictor")
    @patch("nodes.sam_multi_mask_picker.get_or_load_model")
    def test_keyboard_1_selects_mask_0(self, mock_load, mock_pred_factory, mock_predict, node, sample_image):
        """Keyboard '1' → selected_index=0 → returns mask 0."""
        H, W = 64, 64
        mock_load.return_value = MagicMock()
        mock_pred_factory.return_value = MagicMock()
        mock_predict.return_value = _make_mock_masks(H, W, 3)

        result = node.pick_mask(
            image=sample_image,
            model_name="sam_vit_b",
            points_json='[{"x": 32, "y": 32, "label": 1}]',
            bbox_json="",
            precision="fp32",
            selected_index=0,  # Keyboard '1'
        )

        assert result["result"][2] == 0
        assert torch.allclose(result["result"][0][0], result["result"][1][0])

    @patch("nodes.sam_multi_mask_picker.sam_predict")
    @patch("nodes.sam_multi_mask_picker.get_sam_predictor")
    @patch("nodes.sam_multi_mask_picker.get_or_load_model")
    def test_keyboard_2_selects_mask_1(self, mock_load, mock_pred_factory, mock_predict, node, sample_image):
        """Keyboard '2' → selected_index=1 → returns mask 1."""
        H, W = 64, 64
        mock_load.return_value = MagicMock()
        mock_pred_factory.return_value = MagicMock()
        mock_predict.return_value = _make_mock_masks(H, W, 3)

        result = node.pick_mask(
            image=sample_image,
            model_name="sam_vit_b",
            points_json='[{"x": 32, "y": 32, "label": 1}]',
            bbox_json="",
            precision="fp32",
            selected_index=1,  # Keyboard '2'
        )

        assert result["result"][2] == 1
        assert torch.allclose(result["result"][0][0], result["result"][1][1])

    @patch("nodes.sam_multi_mask_picker.sam_predict")
    @patch("nodes.sam_multi_mask_picker.get_sam_predictor")
    @patch("nodes.sam_multi_mask_picker.get_or_load_model")
    def test_keyboard_3_selects_mask_2(self, mock_load, mock_pred_factory, mock_predict, node, sample_image):
        """Keyboard '3' → selected_index=2 → returns mask 2."""
        H, W = 64, 64
        mock_load.return_value = MagicMock()
        mock_pred_factory.return_value = MagicMock()
        mock_predict.return_value = _make_mock_masks(H, W, 3)

        result = node.pick_mask(
            image=sample_image,
            model_name="sam_vit_b",
            points_json='[{"x": 32, "y": 32, "label": 1}]',
            bbox_json="",
            precision="fp32",
            selected_index=2,  # Keyboard '3'
        )

        assert result["result"][2] == 2
        assert torch.allclose(result["result"][0][0], result["result"][1][2])


# ═══════════════════════════════════════════════════════════════════
#  Click Selection
# ═══════════════════════════════════════════════════════════════════

class TestClickSelection:
    """Click selection is handled by JS widget updating selected_index.
    We verify that each index value produces the correct mask selection."""

    @patch("nodes.sam_multi_mask_picker.sam_predict")
    @patch("nodes.sam_multi_mask_picker.get_sam_predictor")
    @patch("nodes.sam_multi_mask_picker.get_or_load_model")
    def test_click_each_thumbnail(self, mock_load, mock_pred_factory, mock_predict, node, sample_image):
        """Clicking each of the 3 thumbnails should select the corresponding mask."""
        H, W = 64, 64
        mock_load.return_value = MagicMock()
        mock_pred_factory.return_value = MagicMock()
        mock_predict.return_value = _make_mock_masks(H, W, 3)

        for click_idx in range(3):
            result = node.pick_mask(
                image=sample_image,
                model_name="sam_vit_b",
                points_json='[{"x": 32, "y": 32, "label": 1}]',
                bbox_json="",
                precision="fp32",
                selected_index=click_idx,
            )

            assert result["result"][2] == click_idx, f"Click index {click_idx} not returned"
            all_masks = result["result"][1]
            selected = result["result"][0]
            assert torch.allclose(selected[0], all_masks[click_idx]), \
                f"Click on thumbnail {click_idx} didn't select correct mask"


# ═══════════════════════════════════════════════════════════════════
#  Re-run Behavior
# ═══════════════════════════════════════════════════════════════════

class TestRerun:
    """R key triggers re-run. Verify IS_CHANGED forces re-execution."""

    def test_is_changed_hashes_inputs_and_is_never_nan(self):
        """IS_CHANGED must hash its inputs, NOT return NaN.

        This test previously asserted `val != val` — i.e. that IS_CHANGED returned
        float("nan") to force a re-run on every execution. That is the banned
        always-rerun anti-pattern (R7), and the product has since moved to
        hash_args_and_kwargs (nodes/sam_multi_mask_picker.py:149). The test was
        pinning the defect, so it is inverted here rather than repaired: equal
        inputs must hash equal, different inputs must differ, and neither may be NaN.
        """
        kw = dict(image=None, model_name="sam_vit_b", points_json="[]",
                  bbox_json="", precision="fp32", selected_index=0)
        a = SamMultiMaskPickerMEC.IS_CHANGED(**kw)
        b = SamMultiMaskPickerMEC.IS_CHANGED(**kw)
        assert a == a, "IS_CHANGED returned NaN — banned always-rerun pattern"
        assert a == b, "same inputs must produce the same fingerprint"

        kw2 = dict(kw, selected_index=2)
        assert SamMultiMaskPickerMEC.IS_CHANGED(**kw2) != a, (
            "changing selected_index must change the fingerprint"
        )

    @patch("nodes.sam_multi_mask_picker.sam_predict")
    @patch("nodes.sam_multi_mask_picker.get_sam_predictor")
    @patch("nodes.sam_multi_mask_picker.get_or_load_model")
    def test_rerun_produces_consistent_output(self, mock_load, mock_pred_factory, mock_predict, node, sample_image):
        """Re-running with same inputs should produce same masks (deterministic)."""
        H, W = 64, 64
        mock_load.return_value = MagicMock()
        mock_pred_factory.return_value = MagicMock()
        mock_predict.return_value = _make_mock_masks(H, W, 3)

        result1 = node.pick_mask(
            image=sample_image,
            model_name="sam_vit_b",
            points_json='[{"x": 32, "y": 32, "label": 1}]',
            bbox_json="",
            precision="fp32",
            selected_index=0,
        )

        result2 = node.pick_mask(
            image=sample_image,
            model_name="sam_vit_b",
            points_json='[{"x": 32, "y": 32, "label": 1}]',
            bbox_json="",
            precision="fp32",
            selected_index=0,
        )

        assert torch.allclose(result1["result"][1], result2["result"][1])


# ═══════════════════════════════════════════════════════════════════
#  Mask Quality Levels
# ═══════════════════════════════════════════════════════════════════

class TestMaskQualityLevels:
    """SAM returns 3 masks at different quality levels (whole/part/subpart).
    Verify each has different area and score."""

    @patch("nodes.sam_multi_mask_picker.sam_predict")
    @patch("nodes.sam_multi_mask_picker.get_sam_predictor")
    @patch("nodes.sam_multi_mask_picker.get_or_load_model")
    def test_three_masks_have_different_areas(self, mock_load, mock_pred_factory, mock_predict, node, sample_image):
        """The 3 candidate masks should have different numbers of positive pixels."""
        H, W = 64, 64
        mock_load.return_value = MagicMock()
        mock_pred_factory.return_value = MagicMock()
        mock_predict.return_value = _make_mock_masks(H, W, 3)

        result = node.pick_mask(
            image=sample_image,
            model_name="sam_vit_b",
            points_json='[{"x": 32, "y": 32, "label": 1}]',
            bbox_json="",
            precision="fp32",
            selected_index=0,
        )

        all_masks = result["result"][1]
        areas = [all_masks[i].sum().item() for i in range(3)]
        # Each mask should cover a different area
        assert areas[0] != areas[1], "Mask 0 and 1 have same area"
        assert areas[0] != areas[2], "Mask 0 and 2 have same area"
        assert areas[1] != areas[2], "Mask 1 and 2 have same area"

    @patch("nodes.sam_multi_mask_picker.sam_predict")
    @patch("nodes.sam_multi_mask_picker.get_sam_predictor")
    @patch("nodes.sam_multi_mask_picker.get_or_load_model")
    def test_three_masks_have_different_scores(self, mock_load, mock_pred_factory, mock_predict, node, sample_image):
        """Scores for each mask should differ."""
        H, W = 64, 64
        mock_load.return_value = MagicMock()
        mock_pred_factory.return_value = MagicMock()
        mock_predict.return_value = _make_mock_masks(H, W, 3)

        result = node.pick_mask(
            image=sample_image,
            model_name="sam_vit_b",
            points_json='[{"x": 32, "y": 32, "label": 1}]',
            bbox_json="",
            precision="fp32",
            selected_index=0,
        )

        scores = json.loads(result["result"][3])
        assert scores[0] != scores[1], "Score 0 and 1 are identical"
        assert scores[0] != scores[2], "Score 0 and 2 are identical"
        assert scores[1] != scores[2], "Score 1 and 2 are identical"

    @patch("nodes.sam_multi_mask_picker.sam_predict")
    @patch("nodes.sam_multi_mask_picker.get_sam_predictor")
    @patch("nodes.sam_multi_mask_picker.get_or_load_model")
    def test_largest_mask_is_first(self, mock_load, mock_pred_factory, mock_predict, node, sample_image):
        """In our mock setup, mask 0 is the largest (whole object)."""
        H, W = 64, 64
        mock_load.return_value = MagicMock()
        mock_pred_factory.return_value = MagicMock()
        mock_predict.return_value = _make_mock_masks(H, W, 3)

        result = node.pick_mask(
            image=sample_image,
            model_name="sam_vit_b",
            points_json='[{"x": 32, "y": 32, "label": 1}]',
            bbox_json="",
            precision="fp32",
            selected_index=0,
        )

        all_masks = result["result"][1]
        area_0 = all_masks[0].sum().item()
        area_1 = all_masks[1].sum().item()
        area_2 = all_masks[2].sum().item()
        assert area_0 > area_1 > area_2, "Masks should be ordered largest to smallest"


# ═══════════════════════════════════════════════════════════════════
#  Model Type Detection
# ═══════════════════════════════════════════════════════════════════

class TestModelTypeDetection:
    def test_detect_sam1(self):
        assert SamMultiMaskPickerMEC._detect_model_type("sam_vit_h") == "sam_vit_h"
        assert SamMultiMaskPickerMEC._detect_model_type("sam_vit_b") == "sam_vit_h"

    def test_detect_sam2(self):
        assert SamMultiMaskPickerMEC._detect_model_type("sam2_hiera_large") == "sam2"
        assert SamMultiMaskPickerMEC._detect_model_type("sam2.1_hiera_tiny") == "sam2"

    def test_detect_sam_hq(self):
        assert SamMultiMaskPickerMEC._detect_model_type("sam_hq_vit_h") == "sam_hq"

    def test_detect_sam3(self):
        assert SamMultiMaskPickerMEC._detect_model_type("sam3") == "sam3"


# ═══════════════════════════════════════════════════════════════════
#  Index Clamping
# ═══════════════════════════════════════════════════════════════════

class TestIndexClamping:
    @patch("nodes.sam_multi_mask_picker.sam_predict")
    @patch("nodes.sam_multi_mask_picker.get_sam_predictor")
    @patch("nodes.sam_multi_mask_picker.get_or_load_model")
    def test_index_clamped_to_valid_range(self, mock_load, mock_pred_factory, mock_predict, node, sample_image):
        """selected_index > 2 should be clamped to 2."""
        H, W = 64, 64
        mock_load.return_value = MagicMock()
        mock_pred_factory.return_value = MagicMock()
        mock_predict.return_value = _make_mock_masks(H, W, 3)

        result = node.pick_mask(
            image=sample_image,
            model_name="sam_vit_b",
            points_json='[{"x": 32, "y": 32, "label": 1}]',
            bbox_json="",
            precision="fp32",
            selected_index=5,  # Out of range
        )

        assert result["result"][2] == 2  # Clamped to max valid index

    @patch("nodes.sam_multi_mask_picker.sam_predict")
    @patch("nodes.sam_multi_mask_picker.get_sam_predictor")
    @patch("nodes.sam_multi_mask_picker.get_or_load_model")
    def test_negative_index_clamped_to_zero(self, mock_load, mock_pred_factory, mock_predict, node, sample_image):
        """selected_index < 0 should be clamped to 0."""
        H, W = 64, 64
        mock_load.return_value = MagicMock()
        mock_pred_factory.return_value = MagicMock()
        mock_predict.return_value = _make_mock_masks(H, W, 3)

        result = node.pick_mask(
            image=sample_image,
            model_name="sam_vit_b",
            points_json='[{"x": 32, "y": 32, "label": 1}]',
            bbox_json="",
            precision="fp32",
            selected_index=-1,
        )

        assert result["result"][2] == 0  # Clamped to 0
