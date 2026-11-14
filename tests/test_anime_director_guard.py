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

    # Must inject action burst for clean Sakuga recovery without clumsy real-world friction
    assert "[AnimeFX: ActionBurst]" in corrected.visual_desc
    assert "cobblestones" in corrected.visual_desc


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


def test_lightning_sakuga_geometry_and_pacing():
    shot = ShotBriefDescription(
        idx=0,
        is_last=False,
        cam_idx=0,
        visual_desc="High-angle shot with a thin diagonal cyan-white energy streak across the night sky with lightning.",
        audio_desc="[Sound Effect] Crackle.",
        target_duration_sec=3.8,  # Too slow
    )
    corrected = AnimeDirectorGuard.lint_and_auto_correct_shot_brief(shot, 0, 3)
    assert "violent, jagged cyan-blue electric lightning bolt with sharp branching tendrils" in corrected.visual_desc
    assert corrected.target_duration_sec == 1.0  # Snappy, no slow-motion


def test_spatial_anchor_hand_consistency():
    shot = ShotBriefDescription(
        idx=3,
        is_last=False,
        cam_idx=1,
        visual_desc="Warrior kneeling against a cracked stone wall on the right, with extended left hand pressed on the stone wall.",
        audio_desc="",
        target_duration_sec=3.0,
    )
    corrected = AnimeDirectorGuard.lint_and_auto_correct_shot_brief(shot, 3, 5)
    assert "bloodied right hand firmly braced against the stone wall on the right" in corrected.visual_desc
    assert "extended left hand pressed" not in corrected.visual_desc


def test_action_stagnation_advancement():
    shots = [
        ShotBriefDescription(
            idx=0,
            is_last=False,
            cam_idx=0,
            visual_desc="Warrior collapsed on ground, panting and coughing heavily with breath puffs.",
            audio_desc="[Speaker] Warrior: 'Haa...'",
        ),
        ShotBriefDescription(
            idx=1,
            is_last=True,
            cam_idx=1,
            visual_desc="Medium shot: Warrior panting and coughing heavily with breath puffs.",
            audio_desc="[Speaker] Warrior: 'Fuu...'",
        ),
    ]
    corrected = AnimeDirectorGuard.lint_and_auto_correct_storyboard(shots)
    assert "Progressive physical reaction" in corrected[1].visual_desc


def test_ban_eye_level_and_enforce_dynamic_angles():
    # 1. Medium 3/4 low-angle shot with wall brace -> upgraded to steep low-angle worm's-eye perspective
    shot1 = ShotBriefDescription(
        idx=4,
        is_last=False,
        cam_idx=4,
        visual_desc="Medium 3/4 low-angle shot in the dark narrow alleyway. <Kayden> has his right hand braced against the stone wall on the right for support.",
        audio_desc="[Speaker] Kayden: 'Damn it...'",
    )
    c1 = AnimeDirectorGuard.lint_and_auto_correct_shot_brief(shot1, 4, 8)
    assert "Medium 3/4" not in c1.visual_desc
    assert "worm's-eye perspective" in c1.visual_desc or "low-angle" in c1.visual_desc

    # 2. Medium shot with standing and staggering -> upgraded to dynamic low ground-level tracking / Dutch angle
    shot2 = ShotBriefDescription(
        idx=6,
        is_last=False,
        cam_idx=6,
        visual_desc="Medium shot in the dark narrow alleyway. Using his right hand on the stone wall on the right, <Kayden> pushes himself up to his feet and staggers forward.",
        audio_desc="",
    )
    c2 = AnimeDirectorGuard.lint_and_auto_correct_shot_brief(shot2, 6, 8)
    assert "Medium shot" not in c2.visual_desc
    assert any(kw in c2.visual_desc for kw in ["tracking shot", "low-angle", "worm's-eye", "Dutch-angle"])

    # 3. Keyframe ShotDescription purging of medium shot
    sd = ShotDescription(
        idx=6,
        is_last=False,
        cam_idx=6,
        visual_desc="Medium shot in the dark narrow alleyway. Kayden stands up.",
        motion_desc="Tracking camera at eye level moving slowly along with Kayden.",
        ff_desc="Medium shot in the dark narrow alleyway. Kayden has his right hand against the stone wall on the right.",
        lf_desc="Medium shot in the dark alleyway. Kayden leans against the stone wall.",
        variation_type="medium",
        variation_reason="",
        ff_vis_char_idxs=[],
        lf_vis_char_idxs=[],
        audio_desc="",
    )
    c_sd = AnimeDirectorGuard.lint_and_auto_correct_shot_description(sd)
    assert "Medium shot" not in c_sd.ff_desc
    assert "Medium shot" not in c_sd.lf_desc
    assert "at eye level" not in c_sd.motion_desc


def test_motion_prompt_safety_sanitization():
    raw_motion = "Kayden trembles slightly, his bloodied right hand staying braced flat against the cracked wall on the right and his left hand pressing tightly over his chest wounds. His chest rises and falls faintly with labored breathing, his jaw clenches harder as he grinds his teeth, and faint cyan electrical sparks flicker around the bleeding wounds under his left hand."
    sanitized = AnimeDirectorGuard.sanitize_motion_prompt_safety(raw_motion)
    assert "bloodied" not in sanitized
    assert "bleeding wounds" not in sanitized
    assert "chest wounds" not in sanitized
    assert "right hand staying braced flat" in sanitized
    assert "pressing tightly over his chest for support" in sanitized
    assert "around his torso" in sanitized




def test_character_injury_continuity_rear_view():
    sd = ShotDescription(
        idx=5,
        is_last=False,
        cam_idx=5,
        visual_desc="Over-the-shoulder POV shot: viewed from behind Kayden, showing the shoulder and upper back of his ragged dark charcoal coat as he looks at the cat.",
        motion_desc="Camera holds static.",
        ff_desc="Over-the-shoulder shot showing the shoulder and upper back of his ragged dark charcoal coat in the foreground.",
        lf_desc="Over-the-shoulder shot showing the back of his ragged dark charcoal coat.",
        variation_type="small",
        variation_reason="",
        ff_vis_char_idxs=[],
        lf_vis_char_idxs=[],
        audio_desc="",
    )
    c_sd = AnimeDirectorGuard.lint_and_auto_correct_shot_description(sd)
    assert "canonical matte medium-dark slate-grey hooded trench coat" in c_sd.ff_desc
    assert "dusty crimson scuffs" in c_sd.ff_desc
    assert "cyan electrical sparks" in c_sd.ff_desc


def test_monologue_to_discovery_crossfade_transition():
    shot_monologue = ShotBriefDescription(
        idx=4,
        is_last=False,
        cam_idx=4,
        visual_desc="Kayden grinds his teeth in bitter, pained anger against the stone wall on the right.",
        audio_desc="[Speaker] Kayden: 'Kuso, kono zama wo mite miro...'",
        target_duration_sec=None,
        transition_in="cut",
    )
    shot_cat_discovery = ShotBriefDescription(
        idx=5,
        is_last=False,
        cam_idx=5,
        visual_desc="Over-the-shoulder POV shot from behind Kayden: in the corner under a dim streetlamp, a chubby cat is sitting on a box.",
        audio_desc="[Thought] Kayden: '... Neko?'",
        target_duration_sec=None,
        transition_in="cut",  # Abrupt hard cut
    )
    corrected = AnimeDirectorGuard.lint_and_auto_correct_storyboard([shot_monologue, shot_cat_discovery])
    assert corrected[1].transition_in == "crossfade"


def test_recovery_pain_pacing_protection():
    # Heavy crash & recovery collapse should not be sped up or crushed to 1-2 seconds
    shot_collapse = ShotBriefDescription(
        idx=3,
        is_last=False,
        cam_idx=3,
        visual_desc="Kayden skids hard across the asphalt and collapses heavily onto one knee, coughing violently in pain.",
        audio_desc="[Sound Effect] Coughing.",
        target_duration_sec=1.5,  # Unrealistic rushed cut
    )
    corrected = AnimeDirectorGuard.lint_and_auto_correct_shot_brief(shot_collapse, 3, 8)
    assert corrected.target_duration_sec >= 3.5


def test_weak_wound_phrases_upgraded_to_severe_bleeding():
    text = "Kayden stands towering over the cat with dark-crimson cuts and faintly glowing cyan scratches across his chest and arms."
    upgraded = AnimeDirectorGuard.enforce_character_injury_continuity(text)
    assert "deep dark-crimson bleeding battle lacerations" in upgraded
    assert "faintly glowing cyan scratches" not in upgraded


def test_two_shot_priority_and_cat_not_staggering_subject():
    sd = ShotDescription(
        idx=7,
        is_last=True,
        cam_idx=7,
        visual_desc="Ground-level two-shot in the dark alleyway. <Kayden (Fat Orange Tabby Cat)> sits on the cardboard box. <Kayden (Human)> stands towering over the cat.",
        motion_desc="Camera holds static.",
        ff_desc="Ground-level low-angle two-shot: <Kayden (Fat Orange Tabby Cat)> sits on the box, <Kayden (Human)> stands towering over the cat.",
        lf_desc="Ground-level low-angle two-shot: <Kayden (Fat Orange Tabby Cat)> sits on the box, <Kayden (Human)> stands towering over the cat.",
        variation_type="large",
        variation_reason="",
        ff_vis_char_idxs=[0, 1],
        lf_vis_char_idxs=[0, 1],
        audio_desc="",
    )
    c_sd = AnimeDirectorGuard.lint_and_auto_correct_shot_description(sd)
    assert "tracking <Kayden (Fat Orange Tabby Cat)> pushing upright and staggering" not in c_sd.lf_desc
    assert "Ground-level two-shot" in c_sd.visual_desc


def test_no_diagonal_tilt_preservation():
    text = "Low-angle tracking shot along the asphalt pavement, upright vertical composition with no diagonal tilt."
    from agents.anime_director_guard import purify_and_enforce_dynamic_camera_angles
    result = purify_and_enforce_dynamic_camera_angles(text)
    assert "no diagonal tilt" in result


def test_pet_shop_indoor_guard_does_not_inject_alley_stone_wall_or_two_shot():
    from agents.anime_director_guard import purify_and_enforce_dynamic_camera_angles

    shot1_text = (
        "A straight-on close view of the densely stocked shelf wall of colorful cat snack packages. "
        "<Seo Jiwoo> appears at the lower right edge, mouth open in excited amazement."
    )
    out1 = purify_and_enforce_dynamic_camera_angles(shot1_text)
    assert "stone wall" not in out1.lower()
    assert "over-the-shoulder" not in out1.lower()

    shot2_text = (
        "A large medium close-up of <Seo Jiwoo> from the chest up inside the pet shop. "
        "Both hands grip his backpack straps."
    )
    out2 = purify_and_enforce_dynamic_camera_angles(shot2_text)
    assert "two-shot" not in out2.lower()
    assert "a large of" not in out2.lower()


def test_shot_description_accepts_whip_pan_and_normalizes_transition_variants():
    from interfaces.shot_description import ShotBriefDescription, ShotDescription

    brief = ShotBriefDescription(
        idx=0, is_last=True, cam_idx=0, visual_desc="Action tracking", audio_desc="whoosh", transition_in="whip_pan"
    )
    assert brief.transition_in == "whip_pan"

    sd = ShotDescription(
        idx=0, is_last=True, cam_idx=0, visual_desc="Action tracking",
        variation_type="small", variation_reason="static",
        ff_desc="start", lf_desc="end", motion_desc="pan", audio_desc="whoosh",
        transition_in="whip_pan",
    )
    assert sd.transition_in == "whip_pan"

    sd_alias = ShotDescription(
        idx=0, is_last=True, cam_idx=0, visual_desc="Action tracking",
        variation_type="small", variation_reason="static",
        ff_desc="start", lf_desc="end", motion_desc="pan", audio_desc="whoosh",
        transition_in="whip-pan",
    )
    assert sd_alias.transition_in == "whip_pan"

    sd_unknown = ShotDescription(
        idx=0, is_last=True, cam_idx=0, visual_desc="Action tracking",
        variation_type="small", variation_reason="static",
        ff_desc="start", lf_desc="end", motion_desc="pan", audio_desc="whoosh",
        transition_in="iris_wipe",
    )
    assert sd_unknown.transition_in == "cut"


def test_audit_storyboard_defects_detects_comic_artifacts_repetition_and_long_blur():
    from interfaces.shot_description import ShotBriefDescription
    from agents.anime_director_guard import AnimeDirectorGuard

    shots = [
        ShotBriefDescription(
            idx=0, is_last=False, cam_idx=0, target_duration_sec=5.0, transition_in="fade_black",
            visual_desc="A disorienting tilted view dissolving into dense white and gold vertical speed lines near a street sign. No characters visible.",
            audio_desc="[Sound Effect] Rush",
        ),
        ShotBriefDescription(
            idx=1, is_last=False, cam_idx=0,
            visual_desc="Medium shot with two overlapping poses of <CharacterA> inside the shop.",
            audio_desc="[Speaker] CharacterA: 'Oh!'",
        ),
        ShotBriefDescription(
            idx=2, is_last=False, cam_idx=0,
            visual_desc="Two small close-up panels placed side by side on a plain light background.",
            audio_desc="[Sound Effect] Rustle",
        ),
        ShotBriefDescription(
            idx=3, is_last=True, cam_idx=0,
            visual_desc="At the top in the distance <CharacterA> runs; in the large foreground <CharacterA> turns back.",
            audio_desc="[Speaker] CharacterA: 'Wait!'",
        ),
    ]
    defects = AnimeDirectorGuard.audit_storyboard_defects(shots)
    joined = "\n".join(defects)
    assert "speed/blur transition lingers" in joined
    assert "clones multiple poses" in joined
    assert "split side-by-side comic inset panels" in joined
    assert "blank white/light void background" in joined
    assert "distance + foreground" in joined
    assert "repeat the same static cam_idx=0" in joined


def test_sanitize_generic_comic_panel_artifacts_and_deduplicates_indices():
    from interfaces.character import CharacterInScene
    from interfaces.shot_description import ShotDescription
    from agents.anime_director_guard import AnimeDirectorGuard

    chars = [
        CharacterInScene(idx=7, identifier_in_scene="Hero", is_visible=True, static_features="child", dynamic_features="yellow jacket"),
    ]
    sd = ShotDescription(
        idx=0, is_last=True, cam_idx=0,
        visual_desc=(
            "Close-up of <Hero> and dark radial tension lines converge inward across the frame "
            "with a plain light background."
        ),
        variation_type="small",
        variation_reason="Holds against a plain white background with no location details and faint white sparkle marks.",
        ff_desc="<Hero> with small black sparkle marks at upper left on a plain white background.",
        lf_desc="Close-up on the plain white background.",
        motion_desc="<Hero> turns slightly against a plain light background.",
        ff_vis_char_idxs=[7, 7],
        lf_vis_char_idxs=[7, 7],
        audio_desc="[Speaker] Hero: 'Nani?'",
    )
    cleaned = AnimeDirectorGuard.lint_and_auto_correct_shot_description(sd, characters=chars)
    for field in (cleaned.visual_desc, cleaned.variation_reason, cleaned.ff_desc, cleaned.lf_desc, cleaned.motion_desc):
        low = field.lower()
        assert "plain light background" not in low
        assert "plain white background" not in low
        assert "sparkle marks" not in low
        assert "radial tension lines" not in low
    assert cleaned.ff_vis_char_idxs == [7]
    assert cleaned.lf_vis_char_idxs == [7]


def test_duration_cap_on_characterless_speedline_blur_transition():
    from interfaces.shot_description import ShotBriefDescription
    from agents.anime_director_guard import AnimeDirectorGuard

    shot = ShotBriefDescription(
        idx=0, is_last=False, cam_idx=0, transition_in="fade_black", target_duration_sec=5.0,
        visual_desc=(
            "A disorienting tilted view of the same city street crosswalk at sunset dissolving into dense white and gold vertical speed lines. "
            "A cursive 'Tea house' sign appears at the bottom. No characters are clearly visible."
        ),
        audio_desc="[Sound Effect] Muffled street noise.",
    )
    out = AnimeDirectorGuard.lint_and_auto_correct_shot_brief(shot, shot_idx=0, total_shots=4)
    assert out.target_duration_sec == 1.5


def test_purify_dynamic_camera_preserves_grammar_and_avoids_combat_prefixes_indoors():
    from interfaces.character import CharacterInScene
    from interfaces.shot_description import ShotDescription
    from agents.anime_director_guard import AnimeDirectorGuard, purify_and_enforce_dynamic_camera_angles

    # 1) Barefoot character crouching indoors must NOT trigger 'boots coiled in a low sprinter stance' or 'stone wall'
    indoor = "A medium shot inside <Seo Jiwoo>'s house doorway. <Seo Jiwoo> crouches barefoot holding two food bowls. <Kayden (Fat Orange Tabby Cat)> is not visible in this shot."
    out_indoor = purify_and_enforce_dynamic_camera_angles(indoor)
    assert "sprinter stance" not in out_indoor.lower()
    assert "combat boots" not in out_indoor.lower()
    assert "stone wall" not in out_indoor.lower()
    assert "two-shot" not in out_indoor.lower()
    assert "A inside" not in out_indoor

    # 2) 'at the cat's eye level' must not lose the subject leaving 'at the cat's,'
    cat_shot = "Close-up at the cat's eye level looking toward the doorway."
    out_cat = purify_and_enforce_dynamic_camera_angles(cat_shot)
    assert "at the cat's," not in out_cat
    assert "cat's height" in out_cat

    # 3) Solo repeated <Tag> must NOT become a two-shot, and ff_vis_char_idxs controls subject fallback
    chars = [
        CharacterInScene(idx=0, identifier_in_scene="Kayden (Human)", is_visible=True, static_features="man", dynamic_features="coat"),
        CharacterInScene(idx=2, identifier_in_scene="Seo Jiwoo", is_visible=True, static_features="teen", dynamic_features="white shirt"),
    ]
    sd = ShotDescription(
        idx=1, is_last=True, cam_idx=1,
        visual_desc="Inside <Seo Jiwoo>'s house doorway, <Seo Jiwoo> holds bowls and crouches. No other part of <Seo Jiwoo> is visible.",
        variation_type="small", variation_reason="Clean framing.",
        ff_desc="Standing upright in the house doorway holding bowls.",
        lf_desc="Holding bowls warmly in the doorway.",
        motion_desc="Camera holds steady.",
        ff_vis_char_idxs=[2], lf_vis_char_idxs=[2],
        audio_desc="[Speaker] Seo Jiwoo: 'Matase te.'",
    )
    cleaned_sd = AnimeDirectorGuard.lint_and_auto_correct_shot_description(sd, characters=chars)
    assert "two-shot" not in cleaned_sd.visual_desc.lower()
    assert "Kayden (Human)" not in cleaned_sd.ff_desc
    assert "stone wall" not in cleaned_sd.ff_desc.lower()
    assert "pavement" not in cleaned_sd.ff_desc.lower()


def test_enforce_pet_shop_and_store_owner():
    text = "Jiwoo is inside a pet shop looking at rustic brown wooden shelves with bright yellow star sparkles glowing around him."
    guarded = AnimeDirectorGuard.enforce_pet_shop_and_store_owner(text)
    assert "rustic brown wooden shelves" not in guarded.lower()
    assert "bright yellow star sparkles glowing" not in guarded.lower()
    assert "cool lavender-blue modular metal shelves" in guarded
    assert "[PET SHOP AESTHETICS & STORE OWNER]" in guarded
    assert "receding thinning dark brown hair" in guarded




