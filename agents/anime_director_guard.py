"""Anime Director Guard & Heuristic Linter (ViMax Anime Director Mindset).

Enforces professional anime directing grammar across all stages of production:
1. Context Jump Detection & Ma (Negative Space / Environmental Breathing)
2. 4-Phase Sakuga Action Grammar (Anticipation -> Frame Skip -> Empty Impact -> Touchdown)
3. Anti-Pattern Linting (Bans physical running/carrying in superhuman actions; bans duplicate victim clones)
4. Dramatic Camera Mapping (Enforces eye close-up, Dutch angle, bird's-eye, dynamic low-angle 3/4; bans consecutive eye-level monotony)
5. Dynamic Duration & Audio-Visual Sync (2.0s-3.5s for kinetic flashes, 6.0s-8.0s for dialogue/breathing)
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

# Anti-patterns where multiple generic victims are drawn in a solo danger beat
DUPLICATE_VICTIM_PATTERNS = [
    re.compile(r"\b(?:two|2|several|group of)\s+(?:boys|children|kids)\b.*\b(?:playing\s+soccer|kicking\s+ball)\b", re.IGNORECASE),
]

# High-speed action triggers requiring 4-phase Sakuga grammar
SUPERHUMAN_SPEED_TRIGGERS = [
    "super-speed", "superspeed", "superhuman speed", "awakened", "lightning speed",
    "shunpo", "soru", "flash step", "teleport", "vanish", "blitz", "dash", "vụt qua",
    "siêu tốc", "tốc biến", "lướt đi", "biến mất", "cứu thoát", "né đòn"
]


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

        # 1. Detect and eliminate banned physical running / carrying in rescue actions
        for pat in BANNED_ACTION_MOTION_PATTERNS:
            if pat.search(vis):
                # Auto-correct to Sakuga Frame Skip (pure sonic wind rush, zero lightning, zero clumsy running)
                vis = re.sub(
                    pat,
                    "blurs in a blinding supersonic wind rush and sonic air blur, scooping the victim in a single frame skip, vanishing instantly",
                    vis,
                )
                if "[AnimeFX: FrameSkip]" not in vis:
                    vis = f"{vis} [AnimeFX: FrameSkip] [AnimeFX: SpeedStreak] [AnimeFX: Vanish]"
                if "[AnimeFX: FrameSkip]" not in aud:
                    aud = f"{aud} [AnimeFX: FrameSkip] [AnimeFX: SpeedStreak] [AnimeFX: Vanish]"
                dur = 1.8

        # 2. Detect and eliminate duplicate victim clones in road hazard beats
        for pat in DUPLICATE_VICTIM_PATTERNS:
            if pat.search(vis):
                vis = re.sub(pat, "a single young boy chasing a runaway soccer ball into traffic", vis)

        # 3. Superhuman speed / instinctive dash auto-tagging and dynamic duration calibration
        is_speed_action = any(kw in vis.lower() for kw in SUPERHUMAN_SPEED_TRIGGERS)
        if is_speed_action:
            if any(kw in vis.lower() for kw in ["launch", "sprinter", "crouch", "ignite", "tụ lực", "xuất phát", "instinct", "bản năng"]):
                # Pha 1: Pure Instinctive Sprint Push (snappy, no elaborate dragon-ball charging)
                if "[AnimeFX: ActionBurst]" not in vis:
                    vis = f"{vis} [AnimeFX: ActionBurst] [AnimeFX: Chromatic]"
                dur = 1.8
            elif any(kw in vis.lower() for kw in ["streak", "blitz", "vanish", "frame skip", "vụt qua", "biến mất"]):
                # Pha 2: Supersonic Blitz / Frame Skip
                if "[AnimeFX: FrameSkip]" not in vis:
                    vis = f"{vis} [AnimeFX: FrameSkip] [AnimeFX: SpeedStreak] [AnimeFX: Vanish]"
                dur = 1.5
            elif any(kw in vis.lower() for kw in ["skid", "screech", "brakes", "smoke", "empty", "phanh", "trống rỗng"]):
                # Pha 3: Empty Impact / SUV Momentum
                if "[AnimeFX: SpeedLines]" not in vis:
                    vis = f"{vis} [AnimeFX: SpeedLines] [AnimeFX: ScreenShake]"
                dur = 2.5
            elif any(kw in vis.lower() for kw in ["touchdown", "aftermath", "cradling", "safe", "tiếp đất"]):
                # Pha 4: Touchdown & Aftermath
                dur = None  # Full duration for relief dialogue!

        # 4. Dramatic Camera Mapping (Enforce cinematic angles, ban eye-level monotony)
        DYNAMIC_ANGLE_PATTERNS = [
            "low-angle", "low angle", "worm's eye", "worm's-eye", "ground-level", "ground level",
            "overhead", "bird's-eye", "bird's eye", "top-down", "high angle",
            "close-up on eyes", "extreme close-up", "extreme face close-up",
            "dutch angle", "dutch-angle", "canted angle", "canted", "tilted horizon",
            "slanted perspective"
        ]
        has_dynamic_angle = any(p in vis.lower() for p in DYNAMIC_ANGLE_PATTERNS)

        if not has_dynamic_angle:
            curr_vis_l = vis.lower()
            if any(kw in curr_vis_l for kw in ["panic", "wide eyes", "alarm", "hoảng loạn", "sợ hãi", "terror"]):
                vis = f"Extreme close-up on eyes, dramatic pupils dilated in sudden horror: {vis}"
            elif any(kw in curr_vis_l for kw in ["speed", "launch", "crouch", "tụ lực", "sprinter", "activation"]):
                vis = f"Dynamic low-angle 3/4 perspective looking up from asphalt: {vis}"
            elif any(kw in curr_vis_l for kw in ["danger", "approaching", "hurtling", "lao tới", "suv", "truck", "car"]):
                vis = f"Dramatic Dutch-angle slanted perspective, tilted 15-degree horizon: {vis}"
            elif any(kw in curr_vis_l for kw in ["child", "boy", "victim", "frozen", "helpless", "chasing"]):
                vis = f"High overhead bird's-eye perspective looking down at asphalt: {vis}"
            elif any(kw in curr_vis_l for kw in ["walk", "street", "corner", "avenue", "city"]):
                if shot_idx % 2 == 0:
                    vis = f"Atmospheric cinematic wide angle with deep vanishing point: {vis}"
                else:
                    vis = f"Low-angle tracking shot along pavement: {vis}"

        # 5. Dialogue protection: Never abbreviate or speed-up shots with spoken dialogue
        has_dialogue = bool(re.search(r"\[(?:Speaker|Thought)\]", aud, re.IGNORECASE))
        if has_dialogue and dur is not None and dur < 5.0:
            dur = None  # Preserve full ~6.0s - 8.0s duration for speech clarity!

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

        # Scrub banned motion phrases from motion_desc, ff_desc, lf_desc
        for pat in BANNED_ACTION_MOTION_PATTERNS:
            if pat.search(motion):
                motion = re.sub(
                    pat,
                    "blurs in a directional supersonic wind rush and air pressure blur, vanishing from the road in a single frame skip",
                    motion,
                )
            if pat.search(ff):
                ff = re.sub(
                    pat,
                    "pushes off the curb on pure adrenaline instinct, sudden blast of wind and dust exploding backward from sneakers, zero lightning",
                    ff,
                )
            if pat.search(lf):
                lf = re.sub(
                    pat,
                    "vanishes into a blinding sonic speed streak, leaving empty road behind",
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
