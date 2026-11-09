import unittest

from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field

from utils.robust_json_parser import TrailingCommaTolerantPydanticOutputParser


class _Item(BaseModel):
    name: str
    tags: list[str] = Field(default_factory=list)


class _Container(BaseModel):
    items: list[_Item]


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

    def test_rejects_schema_invalid_json(self):
        with self.assertRaises(OutputParserException):
            self.parser.parse('json\n{"wrong_key": []}')


if __name__ == "__main__":
    unittest.main()
