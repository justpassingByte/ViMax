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
