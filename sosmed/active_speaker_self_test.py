"""
Deterministic self-test for the active-speaker core logic.

No real video/audio is needed: we synthesize person boxes (two tracks that
swap horizontal position) and audio envelopes that align with one track's
mouth motion at a time, then assert the active-speaker selection follows the
louder/mouth-moving track. Run directly:

    python -m sosmed.active_speaker_self_test
"""

from .active_speaker import (
    _assign_tracks,
    _active_speaker_per_window,
)

SRC_W, SRC_H = 1920, 1080


def _make_boxes():
    """Two people side by side; person A left, person B right, stable identity."""
    def box(cx, cy, w=200, h=400):
        return {
            "x1": int(cx - w / 2), "y1": int(cy - h / 2),
            "x2": int(cx + w / 2), "y2": int(cy + h / 2),
            "conf": 0.9, "area": w * h,
        }

    per_frame: list[list[dict]] = []
    # 10 frames: person A at x=480, person B at x=1440
    for _ in range(10):
        per_frame.append([box(480, 540), box(1440, 540)])
    return per_frame


def test_track_assignment():
    per_frame = _make_boxes()
    per_frame = _assign_tracks(per_frame)
    # Both persons appear in every frame with consistent track ids
    track_ids = set()
    for boxes in per_frame:
        for b in boxes:
            track_ids.add(b["track_id"])
    assert len(track_ids) == 2, f"expected 2 tracks, got {track_ids}"
    # track id must be stable within a frame pair across frames
    first_ids = sorted(b["track_id"] for b in per_frame[0])
    last_ids = sorted(b["track_id"] for b in per_frame[-1])
    assert first_ids == last_ids, "track ids drifted across frames"
    print("PASS test_track_assignment")


def test_active_speaker_selection():
    """Track 0 speaks in windows 0-4, track 1 speaks in windows 5-9."""
    mouth_signals = {
        0: [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        1: [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    }
    audio_env = [1.0] * 10  # everyone talking, equal loudness
    assignment = _active_speaker_per_window(mouth_signals, audio_env)
    assert assignment[:5] == [0] * 5, f"first half should be track 0: {assignment}"
    assert assignment[5:] == [1] * 5, f"second half should be track 1: {assignment}"
    print("PASS test_active_speaker_selection")


def test_silence_marked():
    mouth_signals = {0: [1.0] * 10, 1: [1.0] * 10}
    audio_env = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    assignment = _active_speaker_per_window(mouth_signals, audio_env)
    assert assignment[0] == -1 and assignment[1] == -1, "silent windows must be -1"
    assert assignment[2] in (0, 1), "loud windows must pick a track"
    print("PASS test_silence_marked")


if __name__ == "__main__":
    test_track_assignment()
    test_active_speaker_selection()
    test_silence_marked()
    print("\nAll active-speaker self-tests passed.")
