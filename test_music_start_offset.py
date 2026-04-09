from sosmed import music as music_module


def test_choose_music_start_offset_is_stable_and_avoids_intro(monkeypatch):
    monkeypatch.setattr(music_module, "get_media_duration", lambda _: 120.0)
    monkeypatch.setattr(music_module, "_compute_audio_window_rms", lambda *args, **kwargs: 1.0)

    offset_a = music_module.choose_music_start_offset(
        "music/ambient.mp3",
        42.0,
        seed_source="clip-a",
    )
    offset_b = music_module.choose_music_start_offset(
        "music/ambient.mp3",
        42.0,
        seed_source="clip-a",
    )
    offset_c = music_module.choose_music_start_offset(
        "music/ambient.mp3",
        42.0,
        seed_source="clip-b",
    )

    assert offset_a == offset_b
    assert offset_a != offset_c  # seed still affects candidate pool placement
    # Must leave enough tail for the full 42s clip (max start = 120 - 42 = 78).
    assert 10.0 <= offset_a <= 78.0


def test_choose_music_start_offset_prefers_louder_segment(monkeypatch):
    monkeypatch.setattr(music_module, "get_media_duration", lambda _: 120.0)
    monkeypatch.setattr(
        music_module,
        "_build_music_offset_candidates",
        lambda *args, **kwargs: [12.0, 25.0, 45.0, 90.0],
    )

    def fake_rms(_path, offset, _window):
        # Make one candidate clearly louder so selection is predictable.
        return 100.0 if offset == 45.0 else 1.0

    monkeypatch.setattr(music_module, "_compute_audio_window_rms", fake_rms)

    offset = music_module.choose_music_start_offset(
        "music/ambient.mp3",
        30.0,
        seed_source="clip-a",
    )

    assert offset == 45.0


def test_choose_music_start_offset_never_starts_too_late(monkeypatch):
    monkeypatch.setattr(music_module, "get_media_duration", lambda _: 60.0)
    monkeypatch.setattr(music_module, "_compute_audio_window_rms", lambda *args, **kwargs: 1.0)

    # For a 40s clip over a 60s track, offset must be <= 20s.
    offset = music_module.choose_music_start_offset(
        "music/ambient.mp3",
        40.0,
        seed_source="clip-coverage",
    )

    assert 0.0 <= offset <= 20.0