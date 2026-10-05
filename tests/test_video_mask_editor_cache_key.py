"""Issue #11.4: editing VideoMaskEditor keyframes must change the node's cache key.

The keyframes live in a server-side session store, not in a widget, so before the fix
IS_CHANGED hashed only the widget values and ComfyUI served the cached masks to a downstream
Preview Mask until some other widget was toggled.
"""
import uuid

import numpy as np
import pytest

from nodes import video_mask_editor as vme


@pytest.fixture
def sid():
    s = f"test-{uuid.uuid4()}"
    vme._get_session(s, create=True)
    yield s
    vme._SESSIONS.pop(s, None)


def key(sid):
    return vme.VideoMaskEditorMEC.IS_CHANGED(image=None, session_id=sid, tween_mode="dt",
                                             feather=0.0, threshold=0.0)


def mask(v):
    return np.full((8, 8), v, dtype=np.uint8)


def test_editing_a_keyframe_changes_the_cache_key(sid):
    k0 = key(sid)
    vme._SESSIONS[sid]["keyframes"][3] = mask(255)
    k1 = key(sid)
    assert k1 != k0
    vme._SESSIONS[sid]["keyframes"][3] = mask(128)        # edit the same frame
    k2 = key(sid)
    assert k2 not in (k0, k1)
    vme._SESSIONS[sid]["keyframes"][7] = mask(10)         # add another frame
    k3 = key(sid)
    assert k3 not in (k0, k1, k2)
    vme._SESSIONS[sid]["keyframes"].pop(7)                # delete it again
    assert key(sid) == k2


def test_re_pinning_an_identical_mask_keeps_the_cache_key(sid):
    vme._SESSIONS[sid]["keyframes"][0] = mask(200)
    k1 = key(sid)
    vme._SESSIONS[sid]["keyframes"][0] = mask(200)        # new array, same content
    assert key(sid) == k1


def test_keyframe_frame_index_is_part_of_the_key(sid):
    vme._SESSIONS[sid]["keyframes"][1] = mask(255)
    k1 = key(sid)
    vme._SESSIONS[sid]["keyframes"].clear()
    vme._SESSIONS[sid]["keyframes"][2] = mask(255)        # same mask, other frame
    assert key(sid) != k1


def test_unknown_or_empty_session_is_stable():
    assert vme.session_digest("") == vme.session_digest("no-such-session") == "no-session"
