import unittest

from typing import List

from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field

from interfaces import ShotBriefDescription
from utils.robust_json_parser import (
    TrailingCommaTolerantPydanticOutputParser,
    repair_unescaped_quotes_in_json,
)


class _Item(BaseModel):
    name: str
    tags: list[str] = Field(default_factory=list)


class _Container(BaseModel):
    items: list[_Item]


class _StoryboardContainer(BaseModel):
    storyboard: List[ShotBriefDescription]


class RobustJsonParserTests(unittest.TestCase):
    def setUp(self):
        self.parser = TrailingCommaTolerantPydanticOutputParser(pydantic_object=_Container)

    def test_parses_bare_json_label_prefix_from_muse_dom(self):
        raw = 'json\n{\n  "items": [\n    {"name": "Kayden", "tags": ["cat"]}\n  ]\n}'
        parsed = self.parser.parse(raw)
        self.assertEqual(parsed.items[0].name, "Kayden")

    def test_parses_bare_json_with_copy_button_label(self):
        raw = 'json\nCopy\n{\n  "items": [{"name": "Jiwoo", "tags": []}]\n}'
        parsed = self.parser.parse(raw)
        self.assertEqual(parsed.items[0].name, "Jiwoo")

    def test_parses_prose_plus_bracket_note_and_trailing_comma(self):
        raw = '[Note] Here is the JSON:\njson\n{\n  "items": [\n    {"name": "A", "tags": ["{braces}",],},\n  ],\n}\nDone!'
        parsed = self.parser.parse(raw)
        self.assertEqual(parsed.items[0].tags, ["{braces}"])

    def test_repairs_unescaped_dialogue_quotes_and_newlines_in_storyboard(self):
        sb_parser = TrailingCommaTolerantPydanticOutputParser(pydantic_object=_StoryboardContainer)
        raw = (
            'Here is the storyboard:\n'
            '{"storyboard": ['
            '{"idx": 0, "is_last": false, "cam_idx": 0, '
            '"visual_desc": "[RefBeat: ch01_b001] Medium shot of <Kayden (Human)>. '
            '<Kayden (Human)> says: "Kuh... Kuso, kono zama wo mite miro."", '
            '"audio_desc": "[Speaker] Kayden (Human): "Wait, what? "Run!" now."", '
            '"transition_in": "cut", "continuity_from_prev": false, "target_duration_sec": null}, '
            '{"idx": 1, "is_last": true, "cam_idx": 1, '
            '"visual_desc": "Line 1\nLine 2 with "inner" quote", '
            '"audio_desc": "[Sound Effect] Thunder!", '
            '"transition_in": "fade_black", "continuity_from_prev": false, "target_duration_sec": 4.0}'
            ']}\nEnd.'
        )
        parsed = sb_parser.parse(raw)
        self.assertEqual(len(parsed.storyboard), 2)
        self.assertIn('"Kuh... Kuso, kono zama wo mite miro."', parsed.storyboard[0].visual_desc)
        self.assertIn('"Wait, what? "Run!" now."', parsed.storyboard[0].audio_desc)
        self.assertEqual(parsed.storyboard[1].visual_desc, 'Line 1\nLine 2 with "inner" quote')

    def test_repair_preserves_already_escaped_quotes_and_internal_colons(self):
        already_valid = '{"items": [{"name": "Says: \\"Hi, team\\"", "tags": ["ok"]}]}'
        self.assertEqual(repair_unescaped_quotes_in_json(already_valid), already_valid)

    def test_rejects_schema_invalid_json(self):
        with self.assertRaises(OutputParserException):
            self.parser.parse('json\n{"wrong_key": []}')


if __name__ == "__main__":
    unittest.main()
