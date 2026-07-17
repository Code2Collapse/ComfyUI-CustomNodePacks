"""Pre-flight validation: the fatality preventer must fail locally with the
exact plain-English error when the backend lacks workflow node classes."""

import pytest

from renderfarm.preflight import required_classes, validate_backend_nodes


class FakeBackend:
    name = "aks-h100-pool"

    def __init__(self, classes):
        self._classes = classes

    def installed_node_classes(self):
        return self._classes


PROMPT = {
    "1": {"class_type": "LoadImage", "inputs": {}},
    "2": {"class_type": "WanFaceController3DV2", "inputs": {}},
    "3": {"class_type": "KSampler", "inputs": {}},
}


def test_required_classes():
    assert required_classes(PROMPT) == {"LoadImage", "WanFaceController3DV2", "KSampler"}


def test_all_present_passes():
    validate_backend_nodes(PROMPT, FakeBackend({"LoadImage", "WanFaceController3DV2",
                                                "KSampler", "Extra"}))


def test_missing_nodes_fail_locally_with_message():
    with pytest.raises(RuntimeError) as exc:
        validate_backend_nodes(PROMPT, FakeBackend({"LoadImage", "KSampler"}))
    msg = str(exc.value)
    assert "Backend missing required nodes. Please update Git repo." in msg
    assert "WanFaceController3DV2" in msg
    assert "aks-h100-pool" in msg


def test_unknown_backend_inventory_does_not_block():
    validate_backend_nodes(PROMPT, FakeBackend(None))  # gateway hides /object_info
