# PORTED FROM: https://github.com/9nate-drake/Comfyui-SecNodes/tree/af39e6260bc414873fbb2dc8df355904d1609754/inference/sam2/__init__.py (Apache-2.0; original notices below kept)
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# C2C (L7.90): hydra removed - the SeC builder loads the YAML directly, so there is nothing to initialise.


def init_sam2_hydra():
    """Kept for callers; a no-op since the hydra dependency was removed."""
    return None
