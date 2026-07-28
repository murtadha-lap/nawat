from __future__ import annotations

import pytest

from nawat.units import human_age, human_bytes, parse_size


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1024", 1024),
        ("16GB", 16 * 10**9),
        ("16 GB", 16 * 10**9),
        ("16GiB", 16 * 2**30),
        ("0.5TB", 5 * 10**11),
        ("200MB", 200 * 10**6),
        ("8T", 8 * 10**12),
    ],
)
def test_sizes_parse_with_decimal_and_binary_suffixes(text, expected):
    assert parse_size(text) == expected


def test_unreadable_sizes_say_how_to_write_one():
    with pytest.raises(ValueError, match="150GB"):
        parse_size("lots")


def test_unknown_unit_is_rejected():
    with pytest.raises(ValueError):
        parse_size("16 furlongs")


@pytest.mark.parametrize(
    "count,expected",
    [(0, "0 B"), (512, "512 B"), (16 * 10**9, "16.0 GB"), (212 * 10**6, "212 MB"), (10**12, "1.0 TB")],
)
def test_byte_counts_read_the_way_model_sizes_are_quoted(count, expected):
    assert human_bytes(count) == expected


def test_ages_are_compact():
    assert human_age(42) == "42s"
    assert human_age(3 * 3600) == "3h"
    assert human_age(2.5 * 86400) == "2d"
