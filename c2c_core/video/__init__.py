"""C2C video foundation — lazy handles, probe, decode-on-demand reader."""

from __future__ import annotations

from .budget import BudgetError, budget_error_message, bytes_per_frame, frames_that_fit
from .handle import AudioDescriptor, C2CVideo, ColourTags, FileSource, SequenceSource, TransformOp
from .probe import IndexTable, demux_call_count, get_index_table, probe_file, probe_sequence, reset_demux_call_count
from .reader import drain_container_pool, frame_cache_clear, frame_cache_stats, iter_chunks, read
from .media import SOURCE_MODES, MediaFacts, apply_timeline, image_extensions, probe_media, source_numbers
from .sequence import (EDGE_POLICIES, MISSING_POLICIES, FrameRange, SequenceInfo, Timeline, build_timeline,
                       detect_from_file, parse_frame_range, scan_directory)
from .testmedia import decode_index, encode_index_pattern, expected_flat_rgb_patch, make_colour_test_clip, make_test_clip, make_test_sequence

__all__ = [
    "EDGE_POLICIES",
    "FrameRange",
    "MISSING_POLICIES",
    "MediaFacts",
    "SOURCE_MODES",
    "SequenceInfo",
    "Timeline",
    "apply_timeline",
    "build_timeline",
    "detect_from_file",
    "frame_cache_clear",
    "frame_cache_stats",
    "image_extensions",
    "parse_frame_range",
    "probe_media",
    "scan_directory",
    "source_numbers",
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
