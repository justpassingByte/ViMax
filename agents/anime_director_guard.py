"""Anime Director Guard & Heuristic Linter (ViMax Anime Director Mindset).

Enforces professional anime directing grammar across all stages of production:
1. Context Jump Detection & Ma (Negative Space / Environmental Breathing)
2. Sakuga Action Grammar (Anticipation -> Frame Skip -> Empty Impact -> Momentum Dissipation)
3. Parametric Anti-Pattern Linting (Eliminates clumsy physical running in superhuman actions; generic templates)
4. Dramatic Camera Mapping (Enforces eye close-up, Dutch angle, bird's-eye, dynamic low-angle 3/4; bans eye-level monotony)
5. Spatial & Altitude Continuity (Swooping tilt-down tracking descent bridge between sky and ground)
6. Dynamic Duration Pacing (1.2s-2.0s for kinetic bursts, 2.5s-3.5s for dissipation, voice+0.5s for dialogue)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from interfaces import CharacterInScene, MangaBeat, ShotBriefDescription, ShotDescription

# Anti-patterns in action / rescue prompts that produce diffusion model deformities
BANNED_ACTION_MOTION_PATTERNS = [
    re.compile(r"\b(?:slowly\s+)?(?:running|jogging|sprinting)\b.*\b(?:carrying|holding|scooping)\b", re.IGNORECASE),
    re.compile(r"\b(?:carrying|holding|scooping)\b.*\b(?:slowly\s+)?(?:running|jogging|sprinting)\b", re.IGNORECASE),
    re.compile(r"\b(?:slow\s*motion|slowly)\s+(?:running|jogging|moving|sprinting)\b", re.IGNORECASE),
    re.compile(r"\b(?:clumsy|awkward)\s+(?:run|running|sprint)\b", re.IGNORECASE),
]

# High-speed action triggers requiring Sakuga grammar
SUPERHUMAN_SPEED_TRIGGERS = [
    "super-speed", "superspeed", "superhuman speed", "awakened", "lightning speed",
    "shunpo", "soru", "flash step", "teleport", "vanish", "blitz", "dash", "vụt qua",
    "siêu tốc", "tốc biến", "lướt đi", "biến mất", "cứu thoát", "né đòn",
    "touchdown", "tiếp đất", "aftermath", "rescue", "giải cứu", "landing", "impact"
]

SURFACE_KEYWORDS = [
    "cobblestones", "cobblestone", "asphalt", "pavement", "concrete", "stone", "gravel",
    "grass", "water", "wood", "rooftop", "metal tiles", "tiles", "ground", "floor", "sidewalk"
]

FORCE_FX_KEYWORDS = [
    "cyan electric sparks", "lightning arcs", "cyan lightning", "flames", "shockwave",
    "gusts of wind", "friction smoke and dust", "sonic wind rush", "air pressure wave",
    "kinetic force", "energy blast"
]

SPATIAL_ZONE_KEYWORDS = [
    ("alley", "the narrow alleyway"),
    ("hẻm", "the narrow alleyway"),
    ("crosswalk", "the street crosswalk"),
    ("intersection", "the busy intersection"),
    ("ngã tư", "the street intersection"),
    ("street", "the street level"),
    ("đường", "the street level"),
    ("courtyard", "the courtyard ground"),
    ("rooftop", "the lower rooftop"),
    ("ground", "the ground"),
]


def extract_context_subject(vis: str, characters: Optional[List[CharacterInScene]] = None) -> str:
    """Extract primary character or subject from text/characters list without hardcoding."""
    match = re.search(r"<([^>]+)>", vis)
    if match:
        return f"<{match.group(1)}>"
    if characters:
        for c in characters:
            if c.is_visible and c.identifier_in_scene and c.identifier_in_scene.lower() != "narrator":
                return f"<{c.identifier_in_scene}>"
    return "The subject"


def extract_context_surface(text: str) -> str:
    """Extract physical surface from text keywords."""
    text_l = (text or "").lower()
    for surf in SURFACE_KEYWORDS:
        if surf in text_l:
            return surf
    return "the ground"


def extract_context_force_fx(text: str) -> str:
    """Extract visual force FX from text keywords."""
    text_l = (text or "").lower()
    for fx in FORCE_FX_KEYWORDS:
        if fx in text_l:
            return fx
    return "atmospheric shockwave"


def extract_context_spatial_zone(text: str) -> str:
    """Extract target spatial destination for descent continuity."""
    text_l = (text or "").lower()
    for kw, zone in SPATIAL_ZONE_KEYWORDS:
        if kw in text_l:
            return zone
    return "the ground level"


class AnimeDirectorGuard:
    """Core intelligence engine enforcing anime directing grammar in ViMax."""

    @staticmethod
    def detect_context_jump(prev_scene_desc: str, curr_scene_desc: str) -> bool:
        """Detect whether transitioning between two scenes requires an environmental cushion beat (Ma)."""
        prev_lower = (prev_scene_desc or "").lower()
        curr_lower = (curr_scene_desc or "").lower()

        # Interior to exterior switch
        interior_kws = ["shop", "store", "inside", "room", "bedroom", "house", "indoor", "tiệm", "trong nhà", "phòng"]
        exterior_kws = ["street", "avenue", "road", "crosswalk", "alley", "outside", "ngoài phố", "đường phố", "ngã tư"]

        was_interior = any(kw in prev_lower for kw in interior_kws)
        now_exterior = any(kw in curr_lower for kw in exterior_kws)

        # Tone switch: calm/comedy to high peril/action
        calm_kws = ["peaceful", "cheerful", "comedy", "shopping", "laugh", "bình yên", "vui vẻ", "mua sắm"]
        peril_kws = ["accident", "danger", "peril", "rescue", "speeding", "truck", "suv", "tai nạn", "nguy hiểm", "giải cứu"]

        was_calm = any(kw in prev_lower for kw in calm_kws)
        now_peril = any(kw in curr_lower for kw in peril_kws)

        return (was_interior and now_exterior) or (was_calm and now_peril)

    @staticmethod
    def detect_altitude_spatial_jump(prev_desc: str, curr_desc: str) -> bool:
        """Detect abrupt vertical/spatial jumps (e.g. sky/clouds to ground/alley) requiring camera descent continuity."""
        prev_l = (prev_desc or "").lower()
        curr_l = (curr_desc or "").lower()

        # Clean prior synthetic bridge prefixes so we test genuine scene context
        prev_clean = re.sub(r"^(?:dynamic camera tilt-down tracking descent[^\:]*:\s*)+", "", prev_l)
        curr_clean = re.sub(r"^(?:dynamic camera tilt-down tracking descent[^\:]*:\s*)+", "", curr_l)

        sky_kws = ["sky", "clouds", "cloudy", "rooftops", "high-angle establishing", "bầu trời", "trên trời", "trên không"]
        ground_kws = ["alley", "ground", "asphalt", "pavement", "kneeling", "hẻm", "mặt đất", "quỳ", "street level", "sidewalk", "cobblestone"]

        was_sky = any(kw in prev_clean for kw in sky_kws)
        now_ground = any(kw in curr_clean for kw in ground_kws)

        # If previous shot was already at ground level, do not trigger altitude descent again!
        if any(kw in prev_clean for kw in ground_kws):
            was_sky = False

        has_descent_already = any(kw in curr_l for kw in ["descent", "tilt-down", "swoop", "downward", "hạ độ cao"])

        return was_sky and now_ground and not has_descent_already

    @classmethod
    def lint_and_auto_correct_shot_brief(
        cls,
        shot: ShotBriefDescription,
        shot_idx: int,
        total_shots: int,
        prev_shot: Optional[ShotBriefDescription] = None,
        characters: Optional[List[CharacterInScene]] = None,
    ) -> ShotBriefDescription:
        """Inspect and auto-correct a single ShotBriefDescription to adhere to Sakuga grammar."""
        vis = shot.visual_desc or ""
        aud = shot.audio_desc or ""
        dur = shot.target_duration_sec
        trans = shot.transition_in or "cut"

        subject = extract_context_subject(vis, characters)
        surface = extract_context_surface(vis)
        force_fx = extract_context_force_fx(vis)

        # 1. Eliminate banned clumsy running / carrying during high-speed actions
        for pat in BANNED_ACTION_MOTION_PATTERNS:
            if pat.search(vis):
                vis = re.sub(
                    pat,
                    f"blurs in a blinding supersonic velocity streak and {force_fx}, securing the target in a single Sakuga frame skip without breaking forward momentum",
                    vis,
                )
                if "[AnimeFX: FrameSkip]" not in vis:
                    vis = f"{vis} [AnimeFX: FrameSkip] [AnimeFX: SpeedStreak] [AnimeFX: Vanish]"
                if "[AnimeFX: FrameSkip]" not in aud:
                    aud = f"{aud} [AnimeFX: FrameSkip] [AnimeFX: SpeedStreak] [AnimeFX: Vanish]"
                dur = 1.5

        # 2. Superhuman speed / instinctive dash auto-tagging and dynamic duration calibration
        is_speed_action = any(kw in vis.lower() for kw in SUPERHUMAN_SPEED_TRIGGERS)
        if is_speed_action:
            if any(kw in vis.lower() for kw in ["launch", "sprinter", "crouch", "ignite", "tụ lực", "xuất phát", "instinct", "bản năng"]):
                # Phase 1: Pure Instinctive Sprint Push
                if "[AnimeFX: ActionBurst]" not in vis:
                    vis = f"{vis} [AnimeFX: ActionBurst] [AnimeFX: Chromatic]"
                dur = 1.5
            elif any(kw in vis.lower() for kw in ["streak", "blitz", "vanish", "frame skip", "vụt qua", "biến mất"]):
                # Phase 2: Supersonic Blitz / Frame Skip
                if "[AnimeFX: FrameSkip]" not in vis:
                    vis = f"{vis} [AnimeFX: FrameSkip] [AnimeFX: SpeedStreak] [AnimeFX: Vanish]"
                dur = 1.2
            elif any(kw in vis.lower() for kw in ["skid", "screech", "brakes", "smoke", "empty", "phanh", "trống rỗng"]):
                # Phase 3: Empty Impact / Shockwave
                if "[AnimeFX: SpeedLines]" not in vis:
                    vis = f"{vis} [AnimeFX: SpeedLines] [AnimeFX: ScreenShake]"
                dur = 2.0
            elif any(kw in vis.lower() for kw in ["touchdown", "aftermath", "cradling", "safe", "tiếp đất", "landing"]):
                # Phase 4 & 5: Kinetic Deceleration & Momentum Dissipation (Parametric template)
                if not any(kw in vis.lower() for kw in ["friction", "slide", "skid", "trượt", "ma sát", "sóng xung kích"]):
                    vis = f"Kinetic deceleration friction: {subject}'s boots slide hard across {surface} kicking up friction dust and smoke, body pivoting low to absorb momentum as {force_fx} dissipates, settling into recovery: {vis}"
                    if "[AnimeFX: ActionBurst]" not in vis:
                        vis = f"{vis} [AnimeFX: ActionBurst]"
                dur = 2.5

        # 3. Dramatic Camera Mapping (Enforce cinematic angles, ban eye-level monotony)
        DYNAMIC_ANGLE_PATTERNS = [
            "low-angle", "low angle", "worm's eye", "worm's-eye", "ground-level", "ground level",
            "overhead", "bird's-eye", "bird's eye", "top-down", "high angle",
            "close-up on eyes", "extreme close-up", "extreme face close-up",
            "dutch angle", "dutch-angle", "canted angle", "canted", "tilted horizon",
            "slanted perspective", "tracking shot"
        ]
        has_dynamic_angle = any(p in vis.lower() for p in DYNAMIC_ANGLE_PATTERNS)

        if not has_dynamic_angle:
            curr_vis_l = vis.lower()
            if any(kw in curr_vis_l for kw in ["panic", "wide eyes", "alarm", "hoảng loạn", "sợ hãi", "terror"]):
                vis = f"Extreme close-up on eyes, dramatic pupils dilated in sudden horror: {vis}"
            elif any(kw in curr_vis_l for kw in ["speed", "launch", "crouch", "tụ lực", "sprinter", "activation"]):
                vis = f"Dynamic low-angle 3/4 perspective looking up from {surface}: {vis}"
            elif any(kw in curr_vis_l for kw in ["danger", "approaching", "hurtling", "lao tới", "vehicle", "threat", "car", "truck"]):
                vis = f"Dramatic Dutch-angle slanted perspective, tilted 15-degree horizon: {vis}"
            elif any(kw in curr_vis_l for kw in ["victim", "frozen", "helpless", "chasing", "target"]):
                vis = f"High overhead bird's-eye perspective looking down at {surface}: {vis}"
            elif any(kw in curr_vis_l for kw in ["walk", "street", "corner", "avenue", "city"]):
                if shot_idx % 2 == 0:
                    vis = f"Atmospheric cinematic wide angle with deep vanishing point: {vis}"
                else:
                    vis = f"Low-angle tracking shot along {surface}: {vis}"

        # 4. Dialogue protection: Never abbreviate or speed-up shots with spoken dialogue
        has_dialogue = bool(re.search(r"\[(?:Speaker|Thought)\]", aud, re.IGNORECASE))
        if has_dialogue:
            if dur is not None and dur < 4.0:
                dur = None  # Preserve proper duration for speech clarity!

        return shot.model_copy(
            update={
                "visual_desc": vis.strip(),
                "audio_desc": aud.strip(),
                "target_duration_sec": dur,
                "transition_in": trans,
            }
        )

    @classmethod
    def lint_and_auto_correct_storyboard(
        cls,
        shots: List[ShotBriefDescription],
        characters: Optional[List[CharacterInScene]] = None,
    ) -> List[ShotBriefDescription]:
        """Lint and auto-correct an entire storyboard sequence before rendering."""
        corrected_shots: List[ShotBriefDescription] = []
        n_shots = len(shots)

        for i, shot in enumerate(shots):
            prev = corrected_shots[i - 1] if i > 0 else None
            fixed = cls.lint_and_auto_correct_shot_brief(
                shot=shot,
                shot_idx=i,
                total_shots=n_shots,
                prev_shot=prev,
                characters=characters,
            )
            # Smooth vertical / altitude jumps (e.g. sky streak down to ground alley)
            if prev is not None and cls.detect_altitude_spatial_jump(prev.visual_desc, fixed.visual_desc):
                lower_zone = extract_context_spatial_zone(fixed.visual_desc)
                # Update previous shot to swoop downward
                if "[Camera: Rapid tilt-down" not in prev.visual_desc:
                    prev_updated_vis = f"{prev.visual_desc} [Camera: Rapid tilt-down tracking shot swooping downward toward {lower_zone}]"
                    corrected_shots[i - 1] = prev.model_copy(update={"visual_desc": prev_updated_vis})
                # Update current shot to smoothly continue the descent
                curr_vis = f"Dynamic camera tilt-down tracking descent continuing rapidly from high altitude downward toward {lower_zone}: {fixed.visual_desc}"
                fixed = fixed.model_copy(update={"visual_desc": curr_vis})

            corrected_shots.append(fixed)

        return corrected_shots

    @classmethod
    def lint_and_auto_correct_shot_description(
        cls,
        sd: ShotDescription,
        characters: Optional[List[CharacterInScene]] = None,
    ) -> ShotDescription:
        """Ensure detailed visual and motion descriptions in ShotDescription conform to anime directing."""
        vis = sd.visual_desc or ""
        motion = sd.motion_desc or ""
        ff = sd.ff_desc or vis
        lf = sd.lf_desc or vis

        subject = extract_context_subject(vis, characters)
        surface = extract_context_surface(vis)
        force_fx = extract_context_force_fx(vis)

        # Scrub banned motion phrases from motion_desc, ff_desc, lf_desc
        for pat in BANNED_ACTION_MOTION_PATTERNS:
            if pat.search(motion):
                motion = re.sub(
                    pat,
                    f"blurs in a directional supersonic velocity rush and {force_fx}, securing the target in a single Sakuga frame skip",
                    motion,
                )
            if pat.search(ff):
                ff = re.sub(
                    pat,
                    f"launches forward on pure instinct, sudden blast of {force_fx} exploding backward from boots across {surface}",
                    ff,
                )
            if pat.search(lf):
                lf = re.sub(
                    pat,
                    f"vanishes into a blinding sonic speed streak, leaving empty {surface} behind",
                    lf,
                )

        return sd.model_copy(
            update={
                "visual_desc": vis.strip(),
                "motion_desc": motion.strip(),
                "ff_desc": ff.strip(),
                "lf_desc": lf.strip(),
            }
        )
