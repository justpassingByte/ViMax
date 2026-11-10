"""Unit tests for AnimeDirectorGuard (ViMax Anime Director Mindset).
Tests generic, parametric sakuga directing rules with ZERO story-specific hardcoding.
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
    s_peril = "Sudden speeding vehicle collision hazard"
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
    # Anti-pattern: clumsy running while carrying during high-speed action
    bad_shot = ShotBriefDescription(
        idx=5,
        is_last=False,
        cam_idx=1,
        visual_desc="The hero is slowly running across the street carrying the civilian away from danger.",
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
    assert corrected.target_duration_sec == 1.2


def test_dramatic_camera_mapping():
    prev_shot = ShotBriefDescription(
        idx=1,
        is_last=False,
        cam_idx=0,
        visual_desc="Medium shot at eye level of a threat approaching.",
        audio_desc="",
    )
    # Next shot also eye level with panic -> should be upgraded to Extreme close-up on eyes
    next_shot = ShotBriefDescription(
        idx=2,
        is_last=False,
        cam_idx=1,
        visual_desc="Shot at eye level: Character wide eyes in sudden panic shouting.",
        audio_desc="[Speaker] Character: 'Look out!'",
    )

    corrected = AnimeDirectorGuard.lint_and_auto_correct_shot_brief(
        shot=next_shot,
        shot_idx=2,
        total_shots=6,
        prev_shot=prev_shot,
    )

    assert "Extreme close-up on eyes" in corrected.visual_desc


def test_kinetic_touchdown_enhancement():
    # Anti-pattern: static flat landing copied 1:1 from comic
    static_touchdown_shot = ShotBriefDescription(
        idx=7,
        is_last=False,
        cam_idx=1,
        visual_desc="Touchdown and safe aftermath: The fighter already kneeling safely on the cobblestones holding the target.",
        audio_desc="[Speaker] Character: 'Are you hurt?'",
        target_duration_sec=None,
    )

    corrected = AnimeDirectorGuard.lint_and_auto_correct_shot_brief(
        shot=static_touchdown_shot,
        shot_idx=7,
        total_shots=9,
    )

    # Must inject kinetic deceleration friction & momentum dissipation
    assert "Kinetic deceleration friction" in corrected.visual_desc
    assert "cobblestones" in corrected.visual_desc
    assert "slide hard" in corrected.visual_desc
    assert "[AnimeFX: ActionBurst]" in corrected.visual_desc


def test_dialogue_duration_protection():
    # Shot with dialogue should never have target_duration_sec shortened below spoken speech window
    dialogue_shot = ShotBriefDescription(
        idx=3,
        is_last=False,
        cam_idx=0,
        visual_desc="Warrior kneeling on ground.",
        audio_desc="[Speaker] Warrior: 'Hold on tight, help is on the way.'\n(Vietsub: Giữ chặt nhé, cứu viện tới rồi.)",
        target_duration_sec=1.5,  # Erroneously short
    )

    corrected = AnimeDirectorGuard.lint_and_auto_correct_shot_brief(
        shot=dialogue_shot,
        shot_idx=3,
        total_shots=6,
    )

    assert corrected.target_duration_sec is None  # Preserves full speech duration!


def test_altitude_descent_bridge_in_storyboard():
    # Storyboard with sky shot followed by ground shot should inject camera descent bridge
    # but subsequent ground shots MUST NOT inherit the descent prefix!
    shots = [
        ShotBriefDescription(
            idx=0,
            is_last=False,
            cam_idx=0,
            visual_desc="High-angle establishing shot over cloudy night sky with cyan lightning.",
            audio_desc="[Sound Effect] Thunder.",
        ),
        ShotBriefDescription(
            idx=1,
            is_last=False,
            cam_idx=1,
            visual_desc="Wounded warrior kneeling in dark alley against cracked stone wall.",
            audio_desc="[Sound Effect] Rain drops.",
        ),
        ShotBriefDescription(
            idx=2,
            is_last=True,
            cam_idx=2,
            visual_desc="Extreme close-up side profile of the warrior gasping for air.",
            audio_desc="[Speaker] Warrior: 'Haa...'",
        ),
    ]

    corrected = AnimeDirectorGuard.lint_and_auto_correct_storyboard(shots)

    assert len(corrected) == 3
    assert "tilt-down tracking shot" in corrected[0].visual_desc
    assert "Dynamic camera tilt-down tracking descent continuing rapidly" in corrected[1].visual_desc
    # Crucial: Shot 2 is already at ground level and must NOT cascade the descent prefix
    assert "Dynamic camera tilt-down tracking descent" not in corrected[2].visual_desc


def test_shot_description_scrubbing():
    sd = ShotDescription(
        idx=4,
        is_last=False,
        cam_idx=0,
        visual_desc="Hero launches across asphalt.",
        variation_type="medium",
        variation_reason="Action",
        ff_desc="Hero running across the asphalt carrying the target.",
        lf_desc="Still carrying while running.",
        motion_desc="The hero is slowly jogging while holding the target.",
        audio_desc="",
    )

    scrubbed = AnimeDirectorGuard.lint_and_auto_correct_shot_description(sd)

    assert "carrying while running" not in scrubbed.lf_desc
    assert "slowly jogging while holding" not in scrubbed.motion_desc
    assert "directional supersonic velocity rush" in scrubbed.motion_desc
    assert "asphalt" in scrubbed.ff_desc
