import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from agent_runtime.config import (
    api_provider_from_base_url,
    embedding_api_key,
    embedding_base_url,
    embedding_model,
    embedding_model_provider,
    image_api_key,
    image_base_url,
    image_model,
    image_num_candidates,
    llm_api_key,
    llm_base_url,
    llm_model,
    llm_model_provider,
    load_agent_config,
    reranker_api_key,
    reranker_base_url,
    reranker_model,
    video_api_key,
    video_base_url,
    video_model,
    video_provider,
    audio_model_provider,
    subtitles_enabled,
    subtitles_language,
    subtitles_burn_in,
    subtitles_font,
    subtitles_font_size,
)


class AgentConfigTests(unittest.TestCase):
    def setUp(self):
        load_agent_config.cache_clear()

    def tearDown(self):
        load_agent_config.cache_clear()

    def test_reads_agent_local_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "configs"
            config_dir.mkdir()
            (config_dir / "agent.local.yaml").write_text(yaml.safe_dump({
                "llm": {"model_provider": "openai", "model": "config-llm", "base_url": "https://config.test/v1", "api_key": "config-key"},
                "image": {"model": "config-image", "base_url": "https://image.test", "api_key": "image-key"},
                "video": {"model": "config-video", "base_url": "https://openrouter.ai/api/v1", "api_key": "video-key"},
                "embedding": {"model_provider": "openai", "model": "config-embedding", "base_url": "https://embedding.test/v1", "api_key": "embedding-key"},
                "reranker": {"model": "config-reranker", "base_url": "https://reranker.test", "api_key": "reranker-key"},
            }), encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(llm_model(tmp), "config-llm")
                self.assertEqual(llm_model_provider(tmp), "openai")
                self.assertEqual(llm_base_url(tmp), "https://config.test/v1")
                self.assertEqual(llm_api_key(tmp), "config-key")
                self.assertEqual(image_model(tmp), "config-image")
                self.assertEqual(image_base_url(tmp), "https://image.test")
                self.assertEqual(image_api_key(tmp), "image-key")
                self.assertEqual(video_model(tmp), "config-video")
                self.assertEqual(video_provider(tmp), "openrouter")
                self.assertEqual(video_base_url(tmp), "https://openrouter.ai/api/v1")
                self.assertEqual(video_api_key(tmp), "video-key")
                self.assertEqual(embedding_model_provider(tmp), "openai")
                self.assertEqual(embedding_model(tmp), "config-embedding")
                self.assertEqual(embedding_base_url(tmp), "https://embedding.test/v1")
                self.assertEqual(embedding_api_key(tmp), "embedding-key")
                self.assertEqual(reranker_model(tmp), "config-reranker")
                self.assertEqual(reranker_base_url(tmp), "https://reranker.test")
                self.assertEqual(reranker_api_key(tmp), "reranker-key")

    def test_environment_overrides_agent_local_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "configs"
            config_dir.mkdir()
            (config_dir / "agent.local.yaml").write_text(yaml.safe_dump({"llm": {"model": "config-llm", "api_key": "config-key"}}), encoding="utf-8")
            with patch.dict(os.environ, {"VIMAX_LLM_MODEL": "env-llm", "VIMAX_LLM_MODEL_PROVIDER": "openai", "VIMAX_LLM_API_KEY": "env-key", "VIMAX_VIDEO_BASE_URL": "https://openrouter.ai/api/v1", "VIMAX_EMBEDDING_MODEL": "env-embedding", "VIMAX_EMBEDDING_BASE_URL": "https://env-embedding.test/v1", "VIMAX_EMBEDDING_API_KEY": "env-embedding-key", "VIMAX_RERANKER_MODEL": "env-reranker", "VIMAX_RERANKER_BASE_URL": "https://env-reranker.test", "VIMAX_RERANKER_API_KEY": "env-reranker-key"}, clear=True):
                self.assertEqual(llm_model(tmp), "env-llm")
                self.assertEqual(llm_model_provider(tmp), "openai")
                self.assertEqual(llm_api_key(tmp), "env-key")
                self.assertEqual(video_provider(tmp), "openrouter")
                self.assertEqual(video_base_url(tmp), "https://openrouter.ai/api/v1")
                self.assertEqual(embedding_model(tmp), "env-embedding")
                self.assertEqual(embedding_base_url(tmp), "https://env-embedding.test/v1")
                self.assertEqual(embedding_api_key(tmp), "env-embedding-key")
                self.assertEqual(reranker_model(tmp), "env-reranker")
                self.assertEqual(reranker_base_url(tmp), "https://env-reranker.test")
                self.assertEqual(reranker_api_key(tmp), "env-reranker-key")

    def test_image_and_video_keys_fall_back_to_llm_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "configs"
            config_dir.mkdir()
            (config_dir / "agent.local.yaml").write_text(yaml.safe_dump({"llm": {"api_key": "shared-key"}}), encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(image_api_key(tmp), "shared-key")
                self.assertEqual(video_api_key(tmp), "shared-key")


class QualityConfigTests(unittest.TestCase):
    def setUp(self):
        load_agent_config.cache_clear()

    def tearDown(self):
        load_agent_config.cache_clear()

    def _write(self, tmp, section):
        config_dir = Path(tmp) / "configs"
        config_dir.mkdir()
        (config_dir / "agent.local.yaml").write_text(
            yaml.safe_dump({"quality": section} if section is not None else {}),
            encoding="utf-8",
        )

    def test_defaults_are_the_upgraded_behaviour(self):
        from agent_runtime import config as cfg
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, None)
            with patch.dict(os.environ, {}, clear=True):
                self.assertTrue(cfg.quality_canon_gate(tmp))
                self.assertTrue(cfg.quality_generic_linter(tmp))
                self.assertEqual(cfg.quality_series_slug(tmp), "")
                self.assertEqual(cfg.quality_ref_policy(tmp), "profile")
                self.assertEqual(cfg.quality_max_reference_images(tmp), 4)
                self.assertEqual(cfg.quality_embeddings(tmp), "dinov2")
                self.assertEqual(cfg.quality_text_embeddings(tmp), "e5")
                self.assertAlmostEqual(cfg.quality_scene_split_threshold(tmp), 0.80)
                self.assertTrue(cfg.quality_profiles(tmp))
                self.assertEqual(cfg.quality_profile_min_appearances(tmp), 5)
                self.assertTrue(cfg.quality_identity_qa(tmp))
                self.assertAlmostEqual(cfg.quality_identity_margin(tmp), 0.15)
                self.assertEqual(cfg.quality_max_keyframe_retakes(tmp), 2)
                self.assertEqual(cfg.quality_voice_mode(tmp), "native_fixed")
                self.assertTrue(cfg.quality_shot_planner(tmp))
                self.assertTrue(cfg.quality_story_memory(tmp))
                self.assertTrue(cfg.quality_exceptions_queue(tmp))
                self.assertEqual(cfg.quality_max_clip_retakes(tmp), 1)
                self.assertEqual(cfg.quality_chapter_budget_images(tmp), 0)
                self.assertEqual(cfg.quality_chapter_budget_clips(tmp), 0)

    def test_config_section_overrides_defaults(self):
        from agent_runtime import config as cfg
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, {
                "canon_gate": False,
                "series_slug": "eleceed",
                "ref_policy": "legacy",
                "embeddings": "off",
                "voice_mode": "native",
                "max_reference_images": 2,
                "identity_margin": 0.25,
            })
            with patch.dict(os.environ, {}, clear=True):
                self.assertFalse(cfg.quality_canon_gate(tmp))
                self.assertEqual(cfg.quality_series_slug(tmp), "eleceed")
                self.assertEqual(cfg.quality_ref_policy(tmp), "legacy")
                self.assertEqual(cfg.quality_embeddings(tmp), "off")
                self.assertEqual(cfg.quality_voice_mode(tmp), "native")
                self.assertEqual(cfg.quality_max_reference_images(tmp), 2)
                self.assertAlmostEqual(cfg.quality_identity_margin(tmp), 0.25)

    def test_invalid_enum_falls_back_to_default(self):
        from agent_runtime import config as cfg
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, {"ref_policy": "bogus", "embeddings": "siglip", "voice_mode": "tts_profile"})
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(cfg.quality_ref_policy(tmp), "profile")
                self.assertEqual(cfg.quality_embeddings(tmp), "dinov2")
                self.assertEqual(cfg.quality_voice_mode(tmp), "native_fixed")

    def test_env_overrides_config(self):
        from agent_runtime import config as cfg
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, {"canon_gate": True, "series_slug": "eleceed"})
            with patch.dict(os.environ, {
                "VIMAX_QUALITY_CANON_GATE": "false",
                "VIMAX_QUALITY_SERIES_SLUG": "other",
                "VIMAX_QUALITY_SHOT_PLANNER": "off",
            }, clear=True):
                self.assertFalse(cfg.quality_canon_gate(tmp))
                self.assertEqual(cfg.quality_series_slug(tmp), "other")
                self.assertFalse(cfg.quality_shot_planner(tmp))

    def test_video_provider_is_inferred_from_base_url(self):
        self.assertEqual(api_provider_from_base_url("https://openrouter.ai/api/v1"), "openrouter")
        self.assertEqual(api_provider_from_base_url("https://yunwu.ai/v1"), "yunwu")
        self.assertEqual(api_provider_from_base_url("https://example.com/v1"), "")

    def test_image_candidates_default_and_environment_override(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            self.assertEqual(image_num_candidates(tmp), 2)
            with patch.dict(os.environ, {"VIMAX_IMAGE_NUM_CANDIDATES": "1"}):
                self.assertEqual(image_num_candidates(tmp), 1)

    def test_image_candidates_use_workspace_config(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            config_dir = Path(tmp) / "configs"
            config_dir.mkdir()
            (config_dir / "agent.local.yaml").write_text("image_selection:\n  num_candidates: 3\n", encoding="utf-8")
            self.assertEqual(image_num_candidates(tmp), 3)
            with patch.dict(os.environ, {"VIMAX_IMAGE_NUM_CANDIDATES": "1"}):
                self.assertEqual(image_num_candidates(tmp), 1)

    def test_invalid_image_candidate_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for value in ("0", "-1", "1.5", "abc", ""):
                with self.subTest(value=value), patch.dict(os.environ, {"VIMAX_IMAGE_NUM_CANDIDATES": value}, clear=True):
                    with self.assertRaisesRegex(ValueError, "positive integer"):
                        image_num_candidates(tmp)


    def test_video_generator_builds_seedance_for_yunwu_seedance_model(self):
        from agent_runtime.vimax_adapters import _build_video_generator
        from tools.video_generator_doubao_seedance_yunwu_api import VideoGeneratorDoubaoSeedanceYunwuAPI
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "configs"
            config_dir.mkdir()
            (config_dir / "agent.local.yaml").write_text(yaml.safe_dump({
                "video": {
                    "model": "doubao-seedance-1-0-lite-t2v-250428",
                    "base_url": "https://yunwu.ai/volc/v1",
                    "api_key": "test-key"
                }
            }), encoding="utf-8")
            with patch("agent_runtime.config.load_agent_config", side_effect=lambda w=".": yaml.safe_load((config_dir / "agent.local.yaml").read_text(encoding="utf-8"))):
                gen = _build_video_generator()
                self.assertIsInstance(gen, VideoGeneratorDoubaoSeedanceYunwuAPI)
                self.assertEqual(gen.t2v_model, "doubao-seedance-1-0-lite-t2v-250428")
                self.assertEqual(gen.ff2v_model, "doubao-seedance-1-0-lite-i2v-250428")
                self.assertEqual(gen.base_url, "https://yunwu.ai/volc/v1")

    def test_video_provider_respects_explicit_model_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "configs"
            config_dir.mkdir()
            (config_dir / "agent.local.yaml").write_text(yaml.safe_dump({
                "video": {
                    "model_provider": "seedance",
                    "model": "seedance-2.0-fast",
                    "base_url": "https://yunwu.ai",
                    "api_key": "test-key"
                }
            }), encoding="utf-8")
            self.assertEqual(video_provider(tmp), "seedance")

    def test_subtitles_and_native_audio_config_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "configs"
            config_dir.mkdir()
            (config_dir / "agent.local.yaml").write_text(yaml.safe_dump({
                "audio": {"model_provider": "native", "bgm_file": ""},
                "subtitles": {"enabled": True, "language": "vi", "burn_in": True, "font": "Arial", "font_size": 36},
            }), encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(audio_model_provider(tmp), "native")
                self.assertTrue(subtitles_enabled(tmp))
                self.assertEqual(subtitles_language(tmp), "vi")
                self.assertTrue(subtitles_burn_in(tmp))
                self.assertEqual(subtitles_font(tmp), "Arial")
                self.assertEqual(subtitles_font_size(tmp), 36)
            with patch.dict(os.environ, {"VIMAX_SUBTITLES_ENABLED": "false", "VIMAX_SUBTITLES_BURN_IN": "0", "VIMAX_SUBTITLES_LANGUAGE": "en", "VIMAX_SUBTITLES_FONT": "Tahoma", "VIMAX_SUBTITLES_FONT_SIZE": "28"}, clear=True):
                self.assertFalse(subtitles_enabled(tmp))
                self.assertFalse(subtitles_burn_in(tmp))
                self.assertEqual(subtitles_language(tmp), "en")
                self.assertEqual(subtitles_font(tmp), "Tahoma")
                self.assertEqual(subtitles_font_size(tmp), 28)


if __name__ == "__main__":
    unittest.main()
