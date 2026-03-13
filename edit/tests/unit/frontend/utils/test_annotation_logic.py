"""Unit tests for pure-logic helpers in annotation_ops.py (no Qt required)."""

import pytest

from edit.frontend.utils.annotation_ops import (
    _cascade_property_description,
    _frames_to_ranges,
)


class TestFramesToRanges:
    def test_empty_list(self):
        assert _frames_to_ranges([]) == ""

    def test_single_frame(self):
        assert _frames_to_ranges([5]) == "5"

    def test_contiguous_range(self):
        assert _frames_to_ranges([1, 2, 3]) == "1-3"

    def test_multiple_ranges(self):
        assert _frames_to_ranges([1, 2, 3, 7, 8]) == "1-3, 7-8"

    def test_all_singles(self):
        assert _frames_to_ranges([1, 3, 5]) == "1, 3, 5"

    def test_single_contiguous_pair(self):
        assert _frames_to_ranges([4, 5]) == "4-5"

    def test_mixed_ranges_and_singles(self):
        assert _frames_to_ranges([1, 2, 5, 9, 10, 11]) == "1-2, 5, 9-11"

    def test_single_element_range_not_shown_with_dash(self):
        result = _frames_to_ranges([7])
        assert "-" not in result
        assert result == "7"


class TestCascadePropertyDescription:
    def test_position_only(self):
        assert _cascade_property_description(1.0, 2.0, None, None, None) == "position"

    def test_size_only(self):
        assert _cascade_property_description(None, None, 10.0, 20.0, None) == "size"

    def test_rotation_only(self):
        assert _cascade_property_description(None, None, None, None, 0.5) == "rotation"

    def test_position_and_size(self):
        result = _cascade_property_description(1.0, 2.0, 10.0, 20.0, None)
        assert result == "position, size"

    def test_size_and_rotation(self):
        result = _cascade_property_description(None, None, 10.0, 20.0, 0.5)
        assert result == "size, rotation"

    def test_all_properties(self):
        result = _cascade_property_description(1.0, 2.0, 10.0, 20.0, 0.5)
        assert result == "position, size, rotation"

    def test_all_none_returns_fallback(self):
        result = _cascade_property_description(None, None, None, None, None)
        assert result == "properties"

    def test_partial_position_cx_only(self):
        # center_x alone still counts as position
        assert _cascade_property_description(1.0, None, None, None, None) == "position"

    def test_partial_position_cy_only(self):
        assert _cascade_property_description(None, 2.0, None, None, None) == "position"

    def test_partial_size_width_only(self):
        assert _cascade_property_description(None, None, 5.0, None, None) == "size"

    def test_partial_size_height_only(self):
        assert _cascade_property_description(None, None, None, 5.0, None) == "size"
