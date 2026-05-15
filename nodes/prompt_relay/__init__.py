# Prompt Relay — module entry.
#
# Refined, multi-backend port of Gordon Chen's Prompt Relay
# (https://gordonchen19.github.io/Prompt-Relay/ and
#  https://github.com/GordonChen19/Prompt-Relay).
# Full attribution and license discussion in ./NOTICE.md.

from __future__ import annotations

from ._nodes import (
    PromptRelayAdvancedOptionsC2C,
    PromptRelayEncodeC2C,
    PromptRelayEncodeKijaiC2C,
    PromptRelayEncodeSmartC2C,
    PromptRelayRestoreKijaiC2C,
)

NODE_CLASS_MAPPINGS = {
    "PromptRelayEncodeC2C":          PromptRelayEncodeC2C,
    "PromptRelayEncodeSmartC2C":     PromptRelayEncodeSmartC2C,
    "PromptRelayEncodeKijaiC2C":     PromptRelayEncodeKijaiC2C,
    "PromptRelayRestoreKijaiC2C":    PromptRelayRestoreKijaiC2C,
    "PromptRelayAdvancedOptionsC2C": PromptRelayAdvancedOptionsC2C,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptRelayEncodeC2C":          "Prompt Relay Encode",
    "PromptRelayEncodeSmartC2C":     "Prompt Relay Encode (Smart)",
    "PromptRelayEncodeKijaiC2C":     "Prompt Relay Encode (Kijai)",
    "PromptRelayRestoreKijaiC2C":    "Prompt Relay Restore (Kijai)",
    "PromptRelayAdvancedOptionsC2C": "Prompt Relay Advanced Options",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
