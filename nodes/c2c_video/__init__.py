"""C2C video foundation - lazy handles, probe, decode-on-demand reader.

The engine moved to c2c_core/video/ (L9.73, the vendored shared core); this package keeps the pack-facing parts
(loaders, routes, savers, colour) and re-exports the engine API it always had, name by name.
"""

from __future__ import annotations

from .budget import BudgetError, budget_error_message, bytes_per_frame, frames_that_fit
from .handle import AudioDescriptor, C2CVideo, ColourTags, FileSource, SequenceSource, TransformOp
from .probe import IndexTable, demux_call_count, get_index_table, probe_file, probe_sequence, reset_demux_call_count
from .reader import drain_container_pool, iter_chunks, read
from .testmedia import decode_index, encode_index_pattern, expected_flat_rgb_patch, make_colour_test_clip, make_test_clip, make_test_sequence

__all__ = [
    "AudioDescriptor",
    "BudgetError",
    "C2CVideo",
    "ColourTags",
    "FileSource",
    "IndexTable",
    "SequenceSource",
    "TransformOp",
    "budget_error_message",
    "bytes_per_frame",
    "decode_index",
    "demux_call_count",
    "drain_container_pool",
    "encode_index_pattern",
    "expected_flat_rgb_patch",
    "frames_that_fit",
    "get_index_table",
    "iter_chunks",
    "make_colour_test_clip",
    "make_test_clip",
    "make_test_sequence",
    "probe_file",
    "probe_sequence",
    "read",
    "reset_demux_call_count",
]
