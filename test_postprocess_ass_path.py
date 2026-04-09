from sosmed.postprocess import _escape_ass_path


def test_escape_ass_path_windows_drive_letter() -> None:
    raw = r"C:\Users\ASUS\AppData\Local\Temp\sosmed_sub_q4wp4x2h.ass"
    got = _escape_ass_path(raw)

    assert got == r"C\\:/Users/ASUS/AppData/Local/Temp/sosmed_sub_q4wp4x2h.ass"


def test_escape_ass_path_escapes_filter_special_chars() -> None:
    raw = r"C:\tmp\my file [subs],v1;ok.ass"
    got = _escape_ass_path(raw)

    assert got == r"C\\:/tmp/my\ file\ \[subs\]\,v1\;ok.ass"
