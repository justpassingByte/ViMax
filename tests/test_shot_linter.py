import unittest

from agents import shot_linter
from interfaces.shot_description import ShotBriefDescription


def _shot(idx, vis, aud="", cam=0, dur=None, trans="cut"):
    return ShotBriefDescription(idx=idx, is_last=False, cam_idx=cam, visual_desc=vis, audio_desc=aud,
                                target_duration_sec=dur, transition_in=trans)


class ShotLinterGenericTests(unittest.TestCase):
    def test_strip_comic_artifacts_removes_void_bg_and_sparkles(self):
        out = shot_linter.strip_comic_artifacts("A hero on a plain white background with small black sparkle marks at the edges")
        self.assertNotIn("plain white background", out.lower())
        self.assertNotIn("sparkle marks", out.lower())
        self.assertIn("softly blurred", out.lower())

    def test_sanitize_motion_prompt_softens_gore(self):
        out = shot_linter.sanitize_motion_prompt("He clutches his bleeding chest, blood dripping, deep lacerations visible")
        self.assertNotIn("bleeding", out.lower())
        self.assertNotIn("blood", out.lower())
        self.assertNotIn("lacerat", out.lower())

    def test_detect_context_jump_interior_to_exterior(self):
        self.assertTrue(shot_linter.detect_context_jump("inside the pet store shopping", "out on the street crosswalk"))
        self.assertFalse(shot_linter.detect_context_jump("inside a room", "inside the same room"))

    def test_audit_flags_void_bg_and_repeated_cam(self):
        shots = [
            _shot(0, "A on a plain white background looking left", cam=1),
            _shot(1, "A looks right", cam=1),
            _shot(2, "A looks up", cam=1),
            _shot(3, "A looks down", cam=1),
        ]
        defects = shot_linter.audit_storyboard(shots)
        joined = " ".join(defects).lower()
        self.assertIn("void background", joined)
        self.assertIn("cam_idx", joined)

    def test_lint_does_not_inject_character_names(self):
        shots = [_shot(0, "A hero vanishes in a sudden blitz streak", aud="[Sound Effect] whoosh")]
        out = shot_linter.lint_and_auto_correct_storyboard(shots)
        text = (out[0].visual_desc + out[0].audio_desc).lower()
        for name in ("kayden", "jiwoo", "hojin", "slate-grey coat"):
            self.assertNotIn(name, text)
        # generic Sakuga tagging still applies
        self.assertIn("animefx", text)

    def test_dialogue_duration_protection(self):
        shots = [_shot(0, "A speaks", aud="[Speaker] A (Calm): 'Hi there'", dur=2.0)]
        out = shot_linter.lint_and_auto_correct_storyboard(shots)
        self.assertIsNone(out[0].target_duration_sec)


if __name__ == "__main__":
    unittest.main()
