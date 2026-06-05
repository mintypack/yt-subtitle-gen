from subtitles import Segment, format_timestamp, to_srt


def test_format_timestamp():
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(1.5) == "00:00:01,500"
    assert format_timestamp(3661.234) == "01:01:01,234"
    assert format_timestamp(-2) == "00:00:00,000"


def test_to_srt_renders_indexed_blocks_with_blank_line_separator():
    segments = [Segment(0.0, 1.5, " Hello"), Segment(1.5, 3.0, "World ")]
    expected = (
        "1\n00:00:00,000 --> 00:00:01,500\nHello\n"
        "\n"
        "2\n00:00:01,500 --> 00:00:03,000\nWorld\n"
    )
    assert to_srt(segments) == expected


def test_to_srt_empty():
    assert to_srt([]) == ""
