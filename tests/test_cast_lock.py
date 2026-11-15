"""Tests for cast lock (phase 11) built from real broken shots in .working_dir."""

import unittest

from agents import cast_lock


# A single visible character (the cat, idx 1) whose motion prompt still points
# "toward the off-frame boy" (Seo Jiwoo, idx 2) — the exact recurrence bug.
CHARACTERS = [
    {"idx": 0, "identifier_in_scene": "Kayden (Human)"},
    {"idx": 1, "identifier_in_scene": "Kayden (Fat Orange Tabby Cat)"},
    {"idx": 2, "identifier_in_scene": "Seo Jiwoo"},
    {"idx": 3, "identifier_in_scene": "Store Owner"},
]

SHOT_CAT_ONLY = {
    "ff_vis_char_idxs": [1],
    "lf_vis_char_idxs": [1],
    "visual_desc": "The fat orange tabby cat fills the frame.",
    "motion_desc": (
        "Static camera held at ground level, with no pan, tilt, dolly or zoom. "
        "The fat orange-and-white tabby cat remains at the centre-right on all four paws, "
        "facing left toward the off-frame boy, holding its wary expression."
    ),
    "ff_desc": "Ground-level low-angle close-up of the cat.",
    "lf_desc": "",
    "audio_desc": "[Speaker] Kayden (Fat Orange Tabby Cat) (Puzzled): '...?'",
}


# Same scene but the motion prompt names the absent character explicitly.
SHOT_NAMES_ABSENT = dict(
    SHOT_CAT_ONLY,
    motion_desc="The cat sits still, looking down at Seo Jiwoo crouched below.",
)


class OffscreenDetectionTests(unittest.TestCase):
    def test_visible_set(self):
        self.assertEqual(cast_lock.visible_char_idxs(SHOT_CAT_ONLY), {1})

    def test_off_frame_phrase_detected(self):
        # the real cat shot only says "off-frame boy" (no literal name)
        self.assertTrue(cast_lock.references_offscreen_person(SHOT_CAT_ONLY["motion_desc"]))

    def test_detects_offscreen_named_character(self):
        off = cast_lock.offscreen_characters(SHOT_NAMES_ABSENT, CHARACTERS)
        ids = [cast_lock._char_id(c) for c in off]
        self.assertIn("Seo Jiwoo", ids)
        # the visible cat is never flagged as off-frame
        self.assertNotIn("Kayden (Fat Orange Tabby Cat)", ids)

    def test_shared_name_with_visible_char_not_flagged(self):
        # "Kayden (Human)" shares the name "Kayden" with the on-screen cat -> same
        # identity, must NOT be treated as a second off-frame character
        off = cast_lock.offscreen_characters(SHOT_NAMES_ABSENT, CHARACTERS)
        self.assertNotIn("Kayden (Human)", [cast_lock._char_id(c) for c in off])

    def test_visible_character_not_offscreen_even_if_named(self):
        # the cat is named in the audio but it IS in frame -> not off-frame
        off = cast_lock.offscreen_characters(SHOT_CAT_ONLY, CHARACTERS)
        self.assertNotIn(1, [cast_lock._char_idx(c) for c in off])


class MotionSanitizeTests(unittest.TestCase):
    def test_off_frame_boy_phrase_removed(self):
        off = ["Seo Jiwoo"]
        out = cast_lock.sanitize_offscreen_motion(SHOT_CAT_ONLY["motion_desc"], off)
        self.assertNotIn("off-frame boy", out.lower())
        self.assertNotIn("boy", out.lower())
        # gaze direction is preserved as neutral off-frame space
        self.assertIn("off-frame space", out.lower())

    def test_named_offscreen_reference_removed(self):
        motion = "The cat looks down at Seo Jiwoo and growls."
        out = cast_lock.sanitize_offscreen_motion(motion, ["Seo Jiwoo"])
        self.assertNotIn("Seo Jiwoo", out)
        self.assertNotIn("Jiwoo", out)
        self.assertIn("eyes lowered", out.lower())

    def test_camera_reveal_of_offscreen_is_held_static(self):
        motion = "The boy stands still, then the camera pans to reveal Seo Jiwoo in the doorway."
        out = cast_lock.sanitize_offscreen_motion(motion, ["Seo Jiwoo"])
        self.assertNotIn("Seo Jiwoo", out)
        self.assertIn("camera holds static", out.lower())

    def test_empty_offscreen_leaves_text_untouched(self):
        motion = "The cat blinks slowly."
        self.assertEqual(cast_lock.sanitize_offscreen_motion(motion, []), motion)

    def test_never_crashes_on_empty(self):
        self.assertEqual(cast_lock.sanitize_offscreen_motion("", ["X"]), "")


class CastLockBlockTests(unittest.TestCase):
    def test_single_visible_character_block(self):
        block = cast_lock.cast_lock_block(["Kayden (Fat Orange Tabby Cat)"], offscreen_present=True)
        self.assertIn("Exactly 1 character", block)
        self.assertIn("never pans", block.lower())
        self.assertIn("off-frame", block.lower())

    def test_no_characters_block(self):
        block = cast_lock.cast_lock_block([], offscreen_present=False)
        self.assertIn("No human or animal characters", block)


class SpeakerOffscreenTests(unittest.TestCase):
    def test_visible_speaker_is_not_offscreen(self):
        self.assertFalse(
            cast_lock.speaker_is_offscreen("Kayden (Fat Orange Tabby Cat)", SHOT_CAT_ONLY, CHARACTERS)
        )

    def test_absent_speaker_is_offscreen(self):
        self.assertTrue(
            cast_lock.speaker_is_offscreen("Seo Jiwoo", SHOT_CAT_ONLY, CHARACTERS)
        )

    def test_narrator_is_not_offscreen(self):
        self.assertFalse(
            cast_lock.speaker_is_offscreen("Narrator", SHOT_CAT_ONLY, CHARACTERS)
        )


class BackViewRegexTests(unittest.TestCase):
    """The look-down misread that made a face-forward shot be treated as a rear shot."""

    import re as _re

    STRICT = (
        r"\b(seen from behind|from behind|back view|rear view|head bowed|back turned|"
        r"facing away|facing wall|over-the-shoulder|back of (?:his|her|their|the) (?:head|body)|"
        r"obscured face|face obscured|insert shot|tightly focused on.*?hand|"
        r"focused on.*?hand|grounded hand|close-up on.*?hand|hand pressed|knuckles|foot|feet|boots)\b"
    )

    def test_looking_down_is_not_back_view(self):
        text = "the boy looking down as he speaks, face toward the camera"
        self.assertIsNone(self._re.search(self.STRICT, text))

    def test_real_rear_shot_still_detected(self):
        text = "the character seen from behind, head bowed"
        self.assertIsNotNone(self._re.search(self.STRICT, text))

    def test_bare_behind_no_longer_false_triggers(self):
        text = "a lamp glowing behind the desk while he faces forward"
        self.assertIsNone(self._re.search(self.STRICT, text))


if __name__ == "__main__":
    unittest.main()
