import tempfile
import unittest
from pathlib import Path

from pipelines.script2video_pipeline import _resolve_character_acoustic_profile
from tools.audio_reference_probe import probe_audio_reference_support, effective_voice_mode


class _Char:
    def __init__(self, ident, static):
        self.identifier_in_scene = ident
        self.static_features = static
        self.dynamic_features = ""


class NativeFixedTests(unittest.TestCase):
    def test_native_fixed_is_context_independent(self):
        chars = [_Char("Hero", "athletic build adult man, trench coat")]
        calm = _resolve_character_acoustic_profile("Hero", chars, context_desc="calm talk", voice_mode="native_fixed")
        pained = _resolve_character_acoustic_profile("Hero", chars, context_desc="guttural groan of pain", voice_mode="native_fixed")
        self.assertEqual(calm["profile"], pained["profile"])

    def test_native_mode_keeps_context_variation(self):
        chars = [_Char("Hero", "athletic build adult man, trench coat")]
        calm = _resolve_character_acoustic_profile("Hero", chars, context_desc="calm talk", voice_mode="native")
        pained = _resolve_character_acoustic_profile("Hero", chars, context_desc="guttural groan of pain", voice_mode="native")
        self.assertNotEqual(calm["profile"], pained["profile"])


class FrozenVoiceTests(unittest.TestCase):
    """Episode-level freeze removes cross-scene / name-collision drift (phase 11)."""

    def _clear_memo(self, wd):
        from pipelines.script2video_pipeline import _VOICE_CACHE_MEMO
        _VOICE_CACHE_MEMO.pop(wd, None)

    def test_same_character_frozen_across_differing_scene_features(self):
        with tempfile.TemporaryDirectory() as wd:
            self._clear_memo(wd)
            # scene 1: features clearly mark an adult man
            scene1 = [_Char("Alex", "athletic build adult man, trench coat")]
            first = _resolve_character_acoustic_profile(
                "Alex", scene1, voice_mode="native_fixed", working_dir=wd
            )
            # scene 2: thinner features that would classify differently if recomputed
            scene2 = [_Char("Alex", "")]
            second = _resolve_character_acoustic_profile(
                "Alex", scene2, voice_mode="native_fixed", working_dir=wd
            )
            self.assertEqual(first["profile"], second["profile"])

    def test_two_forms_of_one_name_do_not_collide(self):
        with tempfile.TemporaryDirectory() as wd:
            self._clear_memo(wd)
            chars = [
                _Char("Alex (Human)", "athletic build adult man"),
                _Char("Alex (Fat Orange Tabby Cat)", "a fat orange tabby cat"),
            ]
            human = _resolve_character_acoustic_profile(
                "Alex (Human)", chars, voice_mode="native_fixed", working_dir=wd
            )
            cat = _resolve_character_acoustic_profile(
                "Alex (Fat Orange Tabby Cat)", chars, voice_mode="native_fixed", working_dir=wd
            )
            # distinct identities keep distinct frozen voices (no shared "Alex" slot)
            self.assertNotEqual(human["profile"], cat["profile"])

    def test_cache_file_is_written(self):
        with tempfile.TemporaryDirectory() as wd:
            self._clear_memo(wd)
            _resolve_character_acoustic_profile(
                "Alex", [_Char("Alex", "adult man")], voice_mode="native_fixed", working_dir=wd
            )
            self.assertTrue((Path(wd) / "voice_profiles.json").is_file())

    def test_native_mode_does_not_freeze(self):
        with tempfile.TemporaryDirectory() as wd:
            self._clear_memo(wd)
            _resolve_character_acoustic_profile(
                "Alex", [_Char("Alex", "adult man")], voice_mode="native", working_dir=wd
            )
            self.assertFalse((Path(wd) / "voice_profiles.json").is_file())


class _GenNoAudio:
    supports_audio_reference = False


class _GenYesAudio:
    def supports_audio_reference(self):
        return True


class ProbeTests(unittest.TestCase):
    def test_unsupported_provider_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(probe_audio_reference_support(_GenNoAudio(), tmp))
            # cached
            self.assertTrue((Path(tmp) / "provider_capabilities.json").is_file())
            mode = effective_voice_mode("native_with_reference", _GenNoAudio(), tmp)
            self.assertEqual(mode, "native_fixed")

    def test_supported_provider_keeps_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            mode = effective_voice_mode("native_with_reference", _GenYesAudio(), tmp)
            self.assertEqual(mode, "native_with_reference")

    def test_probe_fn_failure_is_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            def boom(_gen):
                raise RuntimeError("no")
            self.assertFalse(probe_audio_reference_support(object(), tmp, probe_fn=boom))

    def test_non_reference_mode_passthrough(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(effective_voice_mode("native_fixed", _GenNoAudio(), tmp), "native_fixed")


if __name__ == "__main__":
    unittest.main()
