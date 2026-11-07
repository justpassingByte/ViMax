import unittest
from types import SimpleNamespace
from utils.audio import (
    extract_dialogue_from_shot,
    extract_speech_from_shot,
    resolve_voice_for_speaker,
    DEFAULT_FEMALE_VOICE_ELEVENLABS,
    DEFAULT_MALE_VOICE_ELEVENLABS,
    DEFAULT_FEMALE_VOICE_EDGE_TTS,
    DEFAULT_MALE_VOICE_EDGE_TTS,
)
from agent_runtime.config import audio_character_voices
import tempfile
from pathlib import Path
import yaml
from unittest.mock import patch


class AudioMultiCharacterTests(unittest.TestCase):
    def test_extract_dialogue_speaker_tag(self):
        audio_desc = '[Speaker] Alice (Happy): "I found the secret key!"'
        speaker, text = extract_dialogue_from_shot(audio_desc=audio_desc)
        self.assertEqual(speaker, "Alice")
        self.assertEqual(text, "I found the secret key!")

    def test_extract_dialogue_character_bracket_says(self):
        visual_desc = '<Bob> (male, 30s) says: "Look out behind you!"'
        speaker, text = extract_dialogue_from_shot(visual_desc=visual_desc)
        self.assertEqual(speaker, "Bob")
        self.assertEqual(text, "Look out behind you!")

    def test_extract_dialogue_general_says(self):
        visual_desc = 'Charlie says: "We must hurry now."'
        speaker, text = extract_dialogue_from_shot(visual_desc=visual_desc)
        self.assertEqual(speaker, "Charlie")
        self.assertEqual(text, "We must hurry now.")

    def test_extract_dialogue_unattributed_quotes(self):
        audio_desc = '"A long time ago in a distant galaxy..."'
        speaker, text = extract_dialogue_from_shot(audio_desc=audio_desc)
        self.assertEqual(speaker, "")
        self.assertEqual(text, "A long time ago in a distant galaxy...")

    def test_backwards_compatible_extract_speech(self):
        visual_desc = '<Alice> says: "Testing speech"'
        text = extract_speech_from_shot(visual_desc=visual_desc)
        self.assertEqual(text, "Testing speech")

    def test_resolve_voice_explicit_mapping(self):
        mapping = {"Alice": "custom-voice-alice", "Bob": "custom-voice-bob"}
        voice = resolve_voice_for_speaker("Alice", character_voices=mapping)
        self.assertEqual(voice, "custom-voice-alice")

        # Case-insensitive
        voice_lower = resolve_voice_for_speaker("bob", character_voices=mapping)
        self.assertEqual(voice_lower, "custom-voice-bob")

    def test_resolve_voice_auto_female_detection(self):
        char = SimpleNamespace(
            identifier_in_scene="Luna",
            static_features="A young female wizard with silver hair",
            dynamic_features="Blue robes",
        )
        # ElevenLabs female
        v_eleven = resolve_voice_for_speaker("Luna", characters=[char], provider="elevenlabs")
        self.assertEqual(v_eleven, DEFAULT_FEMALE_VOICE_ELEVENLABS)

        # Edge-TTS female
        v_edge = resolve_voice_for_speaker("Luna", characters=[char], provider="edge-tts")
        self.assertEqual(v_edge, DEFAULT_FEMALE_VOICE_EDGE_TTS)

    def test_resolve_voice_auto_male_detection(self):
        char = SimpleNamespace(
            identifier_in_scene="Thorin",
            static_features="A sturdy male warrior with a thick beard",
            dynamic_features="Heavy armor",
        )
        # ElevenLabs male
        v_eleven = resolve_voice_for_speaker("Thorin", characters=[char], provider="elevenlabs")
        self.assertEqual(v_eleven, DEFAULT_MALE_VOICE_ELEVENLABS)

        # Edge-TTS male
        v_edge = resolve_voice_for_speaker("Thorin", characters=[char], provider="edge-tts")
        self.assertEqual(v_edge, DEFAULT_MALE_VOICE_EDGE_TTS)

    def test_resolve_voice_fallback(self):
        v = resolve_voice_for_speaker("Unknown", default_voice="my-default-voice")
        self.assertEqual(v, "my-default-voice")

    def test_audio_character_voices_config_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "configs"
            config_dir.mkdir()
            (config_dir / "agent.local.yaml").write_text(yaml.safe_dump({
                "audio": {
                    "character_voices": {
                        "Hero": "voice-123",
                        "Villain": "voice-456",
                    }
                }
            }), encoding="utf-8")
            cv = audio_character_voices(tmp)
            self.assertEqual(cv, {"Hero": "voice-123", "Villain": "voice-456"})


if __name__ == "__main__":
    unittest.main()
