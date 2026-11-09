import os
import subprocess
import unittest
from types import SimpleNamespace
from utils.audio import (
    extract_dialogue_from_shot,
    extract_speech_from_shot,
    extract_spoken_dialogues_for_subtitles,
    extract_vietsubs_from_script,
    build_subtitle_cues,
    write_srt_subtitles,
    write_ass_subtitles,
    burn_subtitles_into_video,
    mix_multitrack_video_and_audio,
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

    def test_extract_spoken_dialogues_ignores_sound_effect_quotes(self):
        sfx_only = '[Sound Effect] Tiếng nhai, tiếng "rừ rừ" khoan khoái của mèo.'
        self.assertEqual(extract_spoken_dialogues_for_subtitles(sfx_only), [])
        spoken = '[Sound Effect] Mưa rít qua. [Speaker] Kayden: "Kuzu domo ga..."'
        self.assertEqual(extract_spoken_dialogues_for_subtitles(spoken), [("Kayden", "Kuzu domo ga...")])

    def test_extract_vietsubs_and_build_cues_with_long_split(self):
        script = """
Shot 1 (20.0s):
Visual: Tower roof.
[Speaker] Narrator: "Kakuseisha..."
(Vietsub: Trong thế giới ngầm của các dị nhân, Kayden là cái tên đại diện cho sự bất khả chiến bại. Một kẻ độc hành nắm giữ sức mạnh tối thượng!)

Shot 2 (20.0s):
Visual: Quiet rain.

Shot 3 (25.0s):
[Speaker] Kayden: "Kuzu!"
(Vietsub: Lũ chuột nhắt các ngươi!)
"""
        subs = extract_vietsubs_from_script(script)
        self.assertEqual(set(subs.keys()), {0, 2})
        shots = [SimpleNamespace(idx=0), SimpleNamespace(idx=1), SimpleNamespace(idx=2)]
        cues = build_subtitle_cues(shots, [8.0, 8.0, 8.0], script_vietsubs=subs, max_chars_per_cue=75)
        self.assertEqual(len(cues), 3)
        self.assertTrue(0.0 < cues[0][0] < cues[0][1] < cues[1][0] < cues[1][1] <= 8.0)
        self.assertGreaterEqual(cues[2][0], 16.0)
        self.assertLessEqual(cues[2][1], 24.0)
        self.assertEqual(cues[2][2], "Lũ chuột nhắt các ngươi!")

    def test_mix_preserves_native_audio_and_burns_vietnamese_subtitles(self):
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            src_mp4 = tmp_dir / "src.mp4"
            bgm_wav = tmp_dir / "bgm.wav"
            out_mp4 = tmp_dir / "out.mp4"
            proc1 = subprocess.run(
                [ffmpeg, "-y", "-f", "lavfi", "-i", "color=c=0x112233:s=320x180:d=2",
                 "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                 "-c:v", "libx264", "-c:a", "aac", "-shortest", str(src_mp4)],
                capture_output=True, text=True,
            )
            self.assertEqual(proc1.returncode, 0, proc1.stderr)
            proc2 = subprocess.run(
                [ffmpeg, "-y", "-f", "lavfi", "-i", "sine=frequency=220:duration=2", str(bgm_wav)],
                capture_output=True, text=True,
            )
            self.assertEqual(proc2.returncode, 0, proc2.stderr)

            mix_multitrack_video_and_audio(
                video_path=str(src_mp4),
                output_path=str(out_mp4),
                speech_segments=[],
                bgm_path=str(bgm_wav),
                bgm_volume=0.1,
                preserve_video_audio=True,
            )
            self.assertTrue(out_mp4.exists())

            ass_path = tmp_dir / "sub.ass"
            srt_path = tmp_dir / "sub.srt"
            cues = [(0.1, 1.8, "Bá vương đã thức tỉnh!")]
            write_srt_subtitles(cues, str(srt_path))
            write_ass_subtitles(cues, str(ass_path), play_res_x=320, play_res_y=180, font_size=18)
            burn_subtitles_into_video(str(out_mp4), str(ass_path), str(out_mp4))
            self.assertTrue(srt_path.exists())
            self.assertTrue(out_mp4.exists())

            from moviepy import VideoFileClip
            clip = VideoFileClip(str(out_mp4))
            try:
                self.assertIsNotNone(clip.audio)
                self.assertGreater(clip.duration, 1.5)
            finally:
                clip.close()

    def test_resolve_ffmpeg_exe_skips_binary_without_ass_filter(self):
        from utils.audio import _resolve_ffmpeg_exe

        def fake_ass_check(exe):
            return exe == "/usr/bin/ffmpeg"

        with patch("imageio_ffmpeg.get_ffmpeg_exe", return_value="/opt/imageio/ffmpeg-noass"), \
             patch("utils.audio.os.path.exists", return_value=True), \
             patch("utils.audio.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("utils.audio._ffmpeg_supports_ass", side_effect=fake_ass_check):
            self.assertEqual(_resolve_ffmpeg_exe(require_ass=True), "/usr/bin/ffmpeg")
            self.assertEqual(_resolve_ffmpeg_exe(require_ass=False), "/opt/imageio/ffmpeg-noass")

    def test_burn_subtitles_retries_replace_and_falls_back_when_locked(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            src = tmp_dir / "final_video.mp4"
            ass = tmp_dir / "sub.ass"
            src.write_bytes(b"video")
            ass.write_text("[Script Info]\n", encoding="utf-8")

            def fake_run(cmd, **kwargs):
                Path(cmd[-1]).write_bytes(b"burned-video")
                return subprocess.CompletedProcess(cmd, 0, "", "")

            # Case 1: transient PermissionError then succeeds on retry
            attempts = {"n": 0}
            real_replace = os.replace

            def flaky_replace(s, d):
                attempts["n"] += 1
                if attempts["n"] == 1:
                    raise PermissionError("temporarily locked")
                return real_replace(s, d)

            with patch("utils.audio._resolve_ffmpeg_exe", return_value="ffmpeg"), \
                 patch("utils.audio.subprocess.run", side_effect=fake_run), \
                 patch("utils.audio.os.replace", side_effect=flaky_replace), \
                 patch("utils.audio.time.sleep", return_value=None):
                out_path = burn_subtitles_into_video(str(src), str(ass), str(src))

            self.assertEqual(Path(out_path).resolve(), src.resolve())
            self.assertEqual(src.read_bytes(), b"burned-video")
            self.assertEqual(attempts["n"], 2)

            # Case 2: primary destination permanently locked -> fallback to *_subtitled.mp4
            src.write_bytes(b"original-video")

            def locked_replace(s, d):
                if Path(d).resolve() == src.resolve():
                    raise PermissionError("locked by player")
                return real_replace(s, d)

            with patch("utils.audio._resolve_ffmpeg_exe", return_value="ffmpeg"), \
                 patch("utils.audio.subprocess.run", side_effect=fake_run), \
                 patch("utils.audio.os.replace", side_effect=locked_replace), \
                 patch("utils.audio.time.sleep", return_value=None):
                fallback_out = burn_subtitles_into_video(str(src), str(ass), str(src))

            expected_fallback = tmp_dir / "final_video_subtitled.mp4"
            self.assertEqual(Path(fallback_out).resolve(), expected_fallback.resolve())
            self.assertTrue(expected_fallback.exists())
            self.assertEqual(expected_fallback.read_bytes(), b"burned-video")
            self.assertEqual(src.read_bytes(), b"original-video")


if __name__ == "__main__":
    unittest.main()
