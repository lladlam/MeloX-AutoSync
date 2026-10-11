"""Tests for the V1.2 AMLL version-consistency checker."""

from __future__ import annotations

import amll_version_check as av


def test_clock_supports_all_amll_forms():
    assert av._clock("3.480") == 3.48            # plain seconds (some AMLL files)
    assert abs(av._clock("00:07.101") - 7.101) < 1e-9   # MM:SS.mmm
    assert abs(av._clock("01:02:03.500") - 3723.5) < 1e-9  # HH:MM:SS.mmm
    assert av._clock("") is None
    assert av._clock("abc") is None


def test_ttml_line_starts_reads_local_name_and_reset_namespace(tmp_path):
    # <p> lives inside a div that resets the namespace, which a plain
    # namespace-qualified iter() would miss.
    ttml = tmp_path / "x.ttml"
    ttml.write_text(
        '<tt xmlns="http://www.w3.org/ns/ttml"><body><div xmlns="">'
        '<p begin="00:01.500">a</p><p begin="00:03.000">b</p>'
        "</div></body></tt>",
        encoding="utf-8",
    )
    assert av.ttml_line_starts(ttml) == [1.5, 3.0]
