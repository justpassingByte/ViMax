"""Unit tests for AnimeDirectorGuard (ViMax Anime Director Mindset).
"""

from pathlib import Path
import pytest
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.anime_director_guard import AnimeDirectorGuard
from interfaces import ShotBriefDescription, ShotDescription, CharacterInScene


def test_context_jump_detection():
    # 1. Interior shop to exterior street
    prev_scene = "Pet shop aisles stocked with colorful cat snacks"
    curr_scene = "City street zebra crosswalk at golden sunset"
    assert AnimeDirectorGuard.detect_context_jump(prev_scene, curr_scene) is True

    # 2. Same location / no context jump
    s_a = "City street sidewalk at sunset"
    s_b = "City street crosswalk curb at sunset"
    assert AnimeDirectorGuard.detect_context_jump(s_a, s_b) is False

    # 3. Calm comedy to sudden peril
    s_calm = "Cheerful bedroom eating cat food"
    s_peril = "Sudden speeding SUV accident rescue"
    assert AnimeDirectorGuard.detect_context_jump(s_calm, s_peril) is True


def test_altitude_spatial_jump_detection():
    # Abrupt jump from sky establishing to ground alley
    prev_vis = "High-angle establishing shot over quiet multi-story rooftops and cloudy night sky with cyan lightning."
    curr_vis = "Kayden kneeling on wet asphalt against a cracked stone alley wall."
    assert AnimeDirectorGuard.detect_altitude_spatial_jump(prev_vis, curr_vis) is True

    # Already has descent motion
    curr_vis_descent = "Continuing camera descent into shadowy alleyway: Kayden kneeling on wet asphalt."
    assert AnimeDirectorGuard.detect_altitude_spatial_jump(prev_vis, curr_vis_descent) is False


def test_sakuga_frame_skip_auto_correction():
    # Anti-pattern: running carrying child across street
    bad_shot = ShotBriefDescription(
        idx=5,
        is_last=False,
        cam_idx=1,
        visual_desc="Seo Jiwoo is slowly running across the street carrying the small child in his arms away from the car.",
        audio_desc="[Sound Effect] Whoosh.",
        target_duration_sec=8.0,
    )

    corrected = AnimeDirectorGuard.lint_and_auto_correct_shot_brief(
        shot=bad_shot,
        shot_idx=5,
        total_shots=8,
    )

    assert "running across the street carrying" not in corrected.visual_desc
    assert "[AnimeFX: FrameSkip]" in corrected.visual_desc
    assert "[AnimeFX: SpeedStreak]" in corrected.visual_desc
    assert corrected.target_duration_sec == 1.5


def test_duplicate_victim_elimination():
    # Anti-pattern: 2 boys playing soccer in traffic
    bad_shot = ShotBriefDescription(
        idx=0,
        is_last=False,
        cam_idx=0,
        visual_desc="Two boys playing soccer kicking ball dangerously near the road curb.",
        audio_desc="[Sound Effect] Ambient traffic.",
    )

    corrected = AnimeDirectorGuard.lint_and_auto_correct_shot_brief(
        shot=bad_shot,
        shot_idx=0,
        total_shots=5,
    )

    assert "Two boys" not in corrected.visual_desc
    assert "single young boy" in corrected.visual_desc


def test_dramatic_camera_mapping():
    prev_shot = ShotBriefDescription(
        idx=1,
        is_last=False,
        cam_idx=0,
        visual_desc="Medium shot at eye level of a truck approaching.",
        audio_desc="",
    )
    # Next shot also eye level with panic -> should be upgraded to Extreme close-up on eyes
    next_shot = ShotBriefDescription(
        idx=2,
        is_last=False,
        cam_idx=1,
        visual_desc="Shot at eye level: Jiwoo wide eyes in sudden panic shouting.",
        audio_desc="[Speaker] Seo Jiwoo: 'Yabai!'",
    )

    corrected = AnimeDirectorGuard.lint_and_auto_correct_shot_brief(
        shot=next_shot,
        shot_idx=2,
        total_shots=6,
        prev_shot=prev_shot,
    )

    assert "Extreme close-up on eyes" in corrected.visual_desc


def test_dialogue_duration_protection():
    # Shot with dialogue should never have target_duration_sec shortened below spoken speech window
    dialogue_shot = ShotBriefDescription(
        idx=3,
        is_last=False,
        cam_idx=0,
        visual_desc="Jiwoo kneeling holding boy.",
        audio_desc="[Speaker] Seo Jiwoo: 'Haa... Daijoubu ka?'\n(Vietsub: Em không sao chứ?)",
        target_duration_sec=2.0,  # Erroneously short
    )

    corrected = AnimeDirectorGuard.lint_and_auto_correct_shot_brief(
        shot=dialogue_shot,
        shot_idx=3,
        total_shots=6,
    )

    assert corrected.target_duration_sec is None  # Preserves full ~6-8s speech duration!


def test_shot_description_scrubbing():
    sd = ShotDescription(
        idx=4,
        is_last=False,
        cam_idx=0,
        visual_desc="Jiwoo launches.",
        variation_type="medium",
        variation_reason="Action",
        ff_desc="Boy running across the street carrying the kid.",
        lf_desc="Still carrying while running.",
        motion_desc="The boy is slowly jogging while holding the child.",
        audio_desc="",
    )

    scrubbed = AnimeDirectorGuard.lint_and_auto_correct_shot_description(sd)

    assert "carrying while running" not in scrubbed.lf_desc
    assert "slowly jogging while holding" not in scrubbed.motion_desc
    assert "blurs in a directional supersonic wind rush" in scrubbed.motion_desc
