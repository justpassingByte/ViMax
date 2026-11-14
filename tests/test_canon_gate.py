import unittest

from agents.canon_gate import check_storyboard, repair_storyboard
from interfaces.manga import MangaBeat, MangaDialogueLine
from interfaces.shot_description import ShotBriefDescription


def _beat(beat_id, chars, dialogues=None):
    return MangaBeat(
        beat_id=beat_id,
        characters_present=chars,
        dialogues=dialogues or [],
    )


def _shot(idx, visual, audio="", is_last=False):
    return ShotBriefDescription(idx=idx, is_last=is_last, cam_idx=0, visual_desc=visual, audio_desc=audio)


class _Char:
    def __init__(self, name):
        self.identifier_in_scene = name


class CanonGateCheckTests(unittest.TestCase):
    def setUp(self):
        self.beats = [
            _beat("b0", ["Alice"], [MangaDialogueLine(speaker="Alice", line="Konnichiwa", vietsub="Xin chào")]),
            _beat("b1", ["Alice", "Bob"], [MangaDialogueLine(speaker="Bob", line="Hisashiburi", vietsub="Lâu rồi không gặp")]),
            _beat("b2", ["Bob"], [MangaDialogueLine(speaker="Bob", line="Jaa ne", vietsub="Tạm biệt")]),
        ]
        self.characters = [_Char("Alice"), _Char("Bob")]

    def test_detects_all_issue_kinds(self):
        storyboard = [
            # good shot citing b0, Alice says her line
            _shot(0, "[RefBeat: b0] <Alice> waves.", "[Speaker] Alice (Happy): 'Konnichiwa' (Vietsub: Xin chào)"),
            # uncited shot
            _shot(1, "<Alice> stands quietly."),
            # invented dialogue + foreign character (Carol not in b1)
            _shot(2, "[RefBeat: b1] <Carol> appears.", "[Speaker] Carol (Calm): 'Totally new line never in canon'"),
            # order violation: cites b0 again after b1
            _shot(3, "[RefBeat: b0] <Alice> looks back."),
            # duplicate dialogue: Alice's line again
            _shot(4, "[RefBeat: b0] <Alice> again.", "[Speaker] Alice (Happy): 'Konnichiwa' (Vietsub: Xin chào)", is_last=True),
        ]
        report = check_storyboard(storyboard, self.beats, self.characters, scene_idx=7)
        kinds = {i.kind for i in report.issues}
        self.assertFalse(report.passed)
        self.assertEqual(report.scene_idx, 7)
        self.assertIn("uncited", kinds)
        self.assertIn("invented_dialogue", kinds)
        self.assertIn("foreign_character", kinds)
        self.assertIn("order_violation", kinds)
        self.assertIn("duplicate_dialogue", kinds)
        # Bob's two lines never appear -> missing_dialogue
        self.assertIn("missing_dialogue", kinds)

    def test_unknown_beat(self):
        storyboard = [_shot(0, "[RefBeat: b99] <Alice> waves.", is_last=True)]
        report = check_storyboard(storyboard, self.beats, self.characters)
        self.assertIn("unknown_beat", {i.kind for i in report.issues})

    def test_clean_storyboard_passes(self):
        storyboard = [
            _shot(0, "[RefBeat: b0] <Alice> waves.", "[Speaker] Alice (Happy): 'Konnichiwa' (Vietsub: Xin chào)"),
            _shot(1, "[RefBeat: b1] <Bob> greets <Alice>.", "[Speaker] Bob (Warm): 'Hisashiburi' (Vietsub: Lâu rồi không gặp)"),
            _shot(2, "[RefBeat: b2] <Bob> leaves.", "[Speaker] Bob (Calm): 'Jaa ne' (Vietsub: Tạm biệt)", is_last=True),
        ]
        report = check_storyboard(storyboard, self.beats, self.characters)
        self.assertTrue(report.passed, msg=str([i.model_dump() for i in report.issues]))
        self.assertAlmostEqual(report.stats["beat_coverage"], 1.0)

    def test_flashback_allows_order_regression(self):
        storyboard = [
            _shot(0, "[RefBeat: b1] <Bob> greets <Alice>.", "[Speaker] Bob (Warm): 'Hisashiburi' (Vietsub: Lâu rồi không gặp)"),
            _shot(1, "[RefBeat: b0] [Flashback] <Alice> earlier.", "[Speaker] Alice (Happy): 'Konnichiwa' (Vietsub: Xin chào)"),
            _shot(2, "[RefBeat: b2] <Bob> leaves.", "[Speaker] Bob (Calm): 'Jaa ne' (Vietsub: Tạm biệt)", is_last=True),
        ]
        report = check_storyboard(storyboard, self.beats, self.characters)
        self.assertNotIn("order_violation", {i.kind for i in report.issues})


class CanonGateRepairTests(unittest.TestCase):
    def setUp(self):
        self.beats = [
            _beat("b0", ["Alice"], [MangaDialogueLine(speaker="Alice", line="Konnichiwa", vietsub="Xin chào")]),
            _beat("b1", ["Bob"], [MangaDialogueLine(speaker="Bob", line="Jaa ne", vietsub="Tạm biệt")]),
        ]
        self.characters = [_Char("Alice"), _Char("Bob")]

    def test_repair_yields_passing_report(self):
        storyboard = [
            _shot(0, "[RefBeat: b0] <Alice> waves.", "[Speaker] Alice (Happy): 'Konnichiwa' (Vietsub: Xin chào)"),
            _shot(1, "<Carol> appears.", "[Speaker] Carol: 'Invented line here'"),  # uncited + invented + foreign
            # b1 (Bob's 'Jaa ne') missing entirely
        ]
        report = check_storyboard(storyboard, self.beats, self.characters)
        self.assertFalse(report.passed)
        repaired = repair_storyboard(storyboard, self.beats, report, self.characters)
        # bad shot dropped, missing dialogue shot added
        self.assertTrue(all("Carol" not in s.visual_desc for s in repaired))
        after = check_storyboard(repaired, self.beats, self.characters)
        self.assertTrue(after.passed, msg=str([i.model_dump() for i in after.issues]))
        # indices re-numbered contiguously, last flagged
        self.assertEqual([s.idx for s in repaired], list(range(len(repaired))))
        self.assertTrue(repaired[-1].is_last)


class _CapturingChain:
    def __init__(self, sink):
        self.sink = sink

    async def ainvoke(self, messages, *args, **kwargs):
        self.sink["system"] = messages[0][1]
        from types import SimpleNamespace
        return SimpleNamespace(storyboard=[])


class _CapturingChat:
    def __init__(self, sink):
        self.sink = sink

    def __or__(self, other):
        return _CapturingChain(self.sink)


class GroundedTemplateSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_grounded_flag_selects_grounded_template(self):
        import asyncio
        from agents.storyboard_artist import StoryboardArtist
        from interfaces.character import CharacterInScene

        chars = [CharacterInScene(idx=0, identifier_in_scene="Alice", is_visible=True, static_features="woman")]

        default_sink: dict = {}
        artist = StoryboardArtist(chat_model=_CapturingChat(default_sink))
        await artist.design_storyboard(script="s", characters=chars, grounded=False)
        self.assertIn("professional storyboard artist with the following core skills", default_sink["system"])
        self.assertNotIn("GROUNDING RULES", default_sink["system"])

        grounded_sink: dict = {}
        artist2 = StoryboardArtist(chat_model=_CapturingChat(grounded_sink))
        await artist2.design_storyboard(script="s", characters=chars, grounded=True)
        self.assertIn("GROUNDING RULES", grounded_sink["system"])
        self.assertIn("[RefBeat:", grounded_sink["system"])


if __name__ == "__main__":
    unittest.main()
