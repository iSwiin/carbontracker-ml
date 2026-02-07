from carbontracker.receipt_cleaning import is_junk_line, normalize_text


def test_normalize_text_basic():
    assert normalize_text("  HELLO\tWORLD  ") == "HELLO WORLD"
    assert normalize_text(None) == ""


def test_is_junk_line():
    assert is_junk_line("TOTAL 12.34")
    assert is_junk_line("   ")
    assert not is_junk_line("2% MILK 1GAL 4.29")
