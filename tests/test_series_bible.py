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


if __name__ == "__main__":
    unittest.main()
