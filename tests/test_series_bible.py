import tempfile
import unittest
from pathlib import Path

from utils.series_bible import SeriesBible, load_series


def _make_series(root: Path):
    sdir = root / "series" / "demo"
    sdir.mkdir(parents=True)
    (sdir / "series.yaml").write_text(
        "title: Demo\nstyle_prompt: CLEAN 2D ANIME STYLE.\nnegative_prompt: photoreal\n",
        encoding="utf-8",
    )
    (sdir / "characters.yaml").write_text(
        """characters:
  - id: hero
    aliases: ["Alex", "Alex (Human)"]
    prompt_notes: "Alex has silver hair."
    wardrobe_rules:
      - from_scene: 0
        to_scene: 5
        outfit_text: "School uniform."
      - from_scene: 6
        to_scene: 999
        outfit_text: "Casual shirt."
    voice_description: "Teen boy voice."
""",
        encoding="utf-8",
    )


class SeriesBibleTests(unittest.TestCase):
    def test_loads_style_and_character_notes_by_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_series(root)
            bible = SeriesBible("demo", str(root))
            self.assertTrue(bible.available)
            self.assertEqual(bible.title(), "Demo")
            self.assertIn("CLEAN 2D ANIME", bible.style_prompt())
            notes_early = bible.character_notes("Alex", scene_idx=2)
            self.assertIn("silver hair", notes_early)
            self.assertIn("School uniform", notes_early)
            self.assertNotIn("Casual shirt", notes_early)
            notes_late = bible.character_notes("Alex (Human)", scene_idx=10)
            self.assertIn("Casual shirt", notes_late)
            self.assertEqual(bible.voice_description("Alex"), "Teen boy voice.")

    def test_missing_series_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            bible = SeriesBible("nope", tmp)
            self.assertFalse(bible.available)
            self.assertEqual(bible.style_prompt(), "")
            self.assertEqual(bible.character_notes("Alex", 0), "")
            self.assertEqual(bible.voice_description("Alex"), "")

    def test_blank_slug_is_empty(self):
        bible = load_series("", ".")
        self.assertFalse(bible.available)


def _make_relation_series(root: Path):
    sdir = root / "series" / "rel"
    sdir.mkdir(parents=True)
    (sdir / "series.yaml").write_text("title: Rel\n", encoding="utf-8")
    (sdir / "characters.yaml").write_text(
        """characters:
  - id: boy
    aliases: ["Hojin"]
    voice_description: "Young boy voice."
  - id: mother
    aliases: ["Hojin's Mother"]
    voice_description: "Adult woman voice."
  - id: cat
    aliases: ["Kayden (Fat Orange Tabby Cat)", "Fat Orange Tabby Cat"]
    voice_description: "Grumpy cat voice."
  - id: human
    aliases: ["Kayden", "Kayden (Human)"]
    voice_description: "Awakened adult male voice."
""",
        encoding="utf-8",
    )


class RelationMatchTests(unittest.TestCase):
    """A possessive/relational name must not borrow the owner's voice (phase 11)."""

    def test_relative_does_not_match_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_relation_series(root)
            b = SeriesBible("rel", str(root))
            self.assertEqual(b.voice_description("Hojin's Mother"), "Adult woman voice.")
            self.assertEqual(b.voice_description("Hojin"), "Young boy voice.")

    def test_qualified_form_prefers_its_own_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_relation_series(root)
            b = SeriesBible("rel", str(root))
            # the cat form gets the cat voice, not the bare "Kayden" human voice
            self.assertEqual(
                b.voice_description("Kayden (Fat Orange Tabby Cat)"), "Grumpy cat voice."
            )
            self.assertEqual(b.voice_description("Kayden"), "Awakened adult male voice.")


if __name__ == "__main__":
    unittest.main()
