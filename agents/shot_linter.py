"""Generic shot linter (series-agnostic).

Ports the *generic* directing-grammar cleanup that used to live in the old
series-specific guard — with NO character names, NO wardrobe/identity
injection, and NO forced camera-angle rewriting. All series-specific prompt
content now lives in ``series/<slug>/`` and is applied via ``utils.series_bible``.

Public helpers:
- ``strip_comic_artifacts(text)`` — remove manga layout artifacts / white voids.
- ``sanitize_motion_prompt(text)`` — soften gore wording that trips video safety filters.
- ``detect_context_jump(prev, curr)`` — interior→exterior / calm→peril cushion detector.
- ``audit_storyboard(shots, characters)`` — structural defects (clones, split panels, void bg, 4+ same cam_idx).
- ``lint_and_auto_correct_storyboard(shots, characters)`` / ``lint_and_auto_correct_shot_description(sd, characters)``.
"""

from __future__ import annotations

import re
from typing import List, Optional

from interfaces import CharacterInScene, ShotBriefDescription, ShotDescription

# Anti-patterns in action prompts that produce diffusion-model deformities.
BANNED_ACTION_MOTION_PATTERNS = [
    re.compile(r"\b(?:slowly\s+)?(?:running|jogging|sprinting)\b.*\b(?:carrying|holding|scooping)\b", re.IGNORECASE),
    re.compile(r"\b(?:carrying|holding|scooping)\b.*\b(?:slowly\s+)?(?:running|jogging|sprinting)\b", re.IGNORECASE),
    re.compile(r"\b(?:slow\s*motion|slowly)\s+(?:running|jogging|moving|sprinting)\b", re.IGNORECASE),
    re.compile(r"\b(?:clumsy|awkward)\s+(?:run|running|sprint)\b", re.IGNORECASE),
]

SUPERHUMAN_SPEED_TRIGGERS = [
    "super-speed", "superspeed", "superhuman speed", "awakened", "lightning speed",
    "shunpo", "soru", "flash step", "teleport", "vanish", "blitz", "dash", "vụt qua",
    "siêu tốc", "tốc biến", "lướt đi", "biến mất", "cứu thoát", "né đòn",
    "touchdown", "tiếp đất", "aftermath", "rescue", "giải cứu", "landing", "impact",
]

FORCE_FX_KEYWORDS = [
    "cyan electric sparks", "lightning arcs", "cyan lightning", "flames", "shockwave",
    "gusts of wind", "friction smoke and dust", "sonic wind rush", "air pressure wave",
    "kinetic force", "energy blast",
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
    """Primary on-screen subject, derived only from the text / roster (no hardcoded names)."""
    if re.search(r"\bno\s+characters?\s+(?:is|are)?\s*(?:clearly\s+)?visible\b", vis or "", re.IGNORECASE):
        return "the scene"
    behind_match = re.search(
        r"(?:from\s+behind|past)\s+(?:<([^>]+)>|([A-Z][a-zA-Z0-9_\s\(\)]+?)(?=\s+(?:in|looking|who|with|\.|\,)))",
        vis or "",
    )
    if behind_match:
        subj = behind_match.group(1) or behind_match.group(2)
        return f"<{subj.strip()}>"
    match = re.search(r"<([^>]+)>", vis or "")
    if match:
        return f"<{match.group(1)}>"
    if characters:
        for c in characters:
            if c.is_visible and c.identifier_in_scene and c.identifier_in_scene.lower() != "narrator":
                return f"<{c.identifier_in_scene}>"
    return "The subject"


def extract_context_surface(text: str) -> str:
    text_l = (text or "").lower()
    if "asphalt" in text_l or "pavement" in text_l:
        return "the asphalt pavement"
    if "cobblestone" in text_l:
        return "the cobblestones"
    if "sidewalk" in text_l or "crosswalk" in text_l:
        return "the street sidewalk"
    if "concrete" in text_l:
        return "the concrete ground"
    if "rooftop" in text_l:
        return "the rooftop"
    if "gravel" in text_l:
        return "the gravel ground"
    if "floor" in text_l:
        return "the floor"
    return "the ground"


def extract_context_force_fx(text: str) -> str:
    text_l = (text or "").lower()
    for fx in FORCE_FX_KEYWORDS:
        if fx in text_l:
            return fx
    return "atmospheric shockwave"


def extract_context_spatial_zone(text: str) -> str:
    text_l = (text or "").lower()
    for kw, zone in SPATIAL_ZONE_KEYWORDS:
        if kw in text_l:
            return zone
    return "the ground level"


def strip_comic_artifacts(text: str) -> str:
    """Strip manga layout artifacts, void backgrounds, and sparkle/tension-line chrome."""
    if not text:
        return ""
    updated = text
    updated = re.sub(
        r"\band\s+dark\s+radial\s+tension\s+lines\s+converge\s+inward\s+across\s+the\s+frame\b",
        "with dramatic shadowed vignette framing around the subject",
        updated, flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\bwith\s+(?:small|faint)\s+(?:black|white)\s+sparkle\s+marks(?:\s+(?:at|in)\s+[^,\.;]+)?",
        "with subtle warm atmospheric bokeh", updated, flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\b(?:small\s+|faint\s+)?(?:black|white)\s+sparkle\s+marks(?:\s+(?:at|in)\s+[^,\.;]+)?",
        "subtle warm atmospheric bokeh", updated, flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\bagainst\s+(?:a|the)\s+plain\s+(?:white|light)\s+background(?:\s+with\s+no\s+location\s+details)?\b",
        "against a softly blurred scene background", updated, flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\bon\s+(?:a|the)\s+plain\s+(?:white|light)\s+background(?:\s+with\s+no\s+location\s+details)?\b",
        "in a softly blurred scene environment", updated, flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\b(?:with|in|over)\s+(?:a|the)?\s*plain\s+(?:white|light)\s+background(?:\s+with\s+no\s+location\s+details)?\b",
        "with a softly blurred scene background", updated, flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\b(?:a|the)?\s*plain\s+(?:white|light)\s+background(?:\s+with\s+no\s+location\s+details)?\b",
        "a softly blurred scene background", updated, flags=re.IGNORECASE,
    )
    return re.sub(r"  +", " ", updated).strip()


def sanitize_motion_prompt(motion_desc: str) -> str:
    """Soften gore wording in a motion prompt so video diffusion safety filters don't refuse it.

    Visual wounds already come from the reference keyframe; the motion prompt only
    needs camera + character kinetics.
    """
    if not motion_desc:
        return ""
    sanitized = motion_desc
    sanitized = re.sub(r'[\.,;]?\s*(?:blood|sweat)\s+streams?\s+continue\s+down.*?(?:and\s+)?(?=cyan\s+electrical)', ', while ', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'[\.,;]?\s*(?:blood|sweat)\s+streams?\s+continue\s+down[^\.,;]*[\.,;]?', '.', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'\b(?:blood|sweat)\s+streams?[^\.,;]*', '', sanitized, flags=re.IGNORECASE)

    composite_replacements = [
        (r"\baround\s+the\s+(?:bleeding\s+)?wounds(?:\s+under\s+his\s+left\s+hand)?\b", "around his torso"),
        (r"\bpressing\s+tightly\s+over\s+his\s+chest\s+wounds\b", "pressing tightly over his chest for support"),
        (r"\bpressing\s+tightly\s+over\s+the\s+bleeding\s+(?:battle\s+)?wounds\b", "pressing tightly against his chest"),
        (r"\bclutching\s+his\s+bleeding\s+chest\b", "clutching his chest for support"),
        (r"\bclutched\s+over\s+his\s+bleeding\s+chest\b", "clutched over his chest"),
        (r"\bbleeding\s+(?:chest\s+)?wounds\b", "chest"),
        (r"\bbleeding\s+battle\s+wounds\b", "chest"),
        (r"\bbleeding\s+bare\s+chest\b", "bare chest"),
        (r"\bbleeding\s+chest\b", "chest"),
        (r"\bblood\s+marks?\s+left\s+on\b", "marks left on"),
        (r"\bblood\s+marks?\b", "shadow marks"),
        (r"\bblood\s+smear\s+trailing(?:\s+behind)?\b", "shadows trailing"),
        (r"\bblood\s+smear\b", "grimace"),
        (r"\bblood\s+dripping\b", "sweat dripping"),
        (r"\bblood\s+trail\b", "shadow trail"),
        (r"\bblood\s+stains?\b", "shadows"),
        (r"\bdeep\s+dark-crimson\s+lacerations\b", "battle marks"),
        (r"\bdeep\s+dark-crimson\s+cuts\b", "battle marks"),
        (r"\bdark-crimson\s+cuts\b", "battle marks"),
        (r"\bmarked\s+bare\s+chest,\s*torso\s+and\s+abdomen\b", "torso"),
        (r"\bbare\s+(?:exhausted|marked|wounded)\s+chest\b", "chest"),
        (r"\bmarked\s+with\s+[^\,]*\s*marked\s*,?", "with"),
    ]
    for pattern, repl in composite_replacements:
        sanitized = re.sub(pattern, repl, sanitized, flags=re.IGNORECASE)

    word_scrubbers = [
        (r"\bbloodied\b", "dusty"),
        (r"\bbloody\b", "marked"),
        (r"\bbleeding\b", "exhausted"),
        (r"\bblood\b", "shadow"),
        (r"\blacerat(?:ed|ions?)\b", "marked"),
        (r"\bwounded\b", "exhausted"),
        (r"\bwounds?\b", "torso"),
        (r"\b(?:deep\s+)?dark-crimson\b", "battle"),
        (r"\bopen\s+cuts?\b", "marks"),
        (r"\bgore\b", "debris"),
        (r"\bgory\b", "dramatic"),
    ]
    for pattern, repl in word_scrubbers:
        sanitized = re.sub(pattern, repl, sanitized, flags=re.IGNORECASE)

    sanitized = re.sub(r"\s*,\s*,", ",", sanitized)
    sanitized = re.sub(r"\s{2,}", " ", sanitized).strip()
    return sanitized


def detect_context_jump(prev_scene_desc: str, curr_scene_desc: str) -> bool:
    """Interior→exterior or calm→peril switch that warrants an environmental cushion (Ma)."""
    prev_lower = (prev_scene_desc or "").lower()
    curr_lower = (curr_scene_desc or "").lower()

    interior_kws = ["shop", "store", "inside", "room", "bedroom", "house", "indoor", "tiệm", "trong nhà", "phòng"]
    exterior_kws = ["street", "avenue", "road", "crosswalk", "alley", "outside", "ngoài phố", "đường phố", "ngã tư"]
    was_interior = any(kw in prev_lower for kw in interior_kws)
    now_exterior = any(kw in curr_lower for kw in exterior_kws)

    calm_kws = ["peaceful", "cheerful", "comedy", "shopping", "laugh", "bình yên", "vui vẻ", "mua sắm"]
    peril_kws = ["accident", "danger", "peril", "rescue", "speeding", "truck", "suv", "tai nạn", "nguy hiểm", "giải cứu"]
    was_calm = any(kw in prev_lower for kw in calm_kws)
    now_peril = any(kw in curr_lower for kw in peril_kws)

    return (was_interior and now_exterior) or (was_calm and now_peril)


def detect_altitude_spatial_jump(prev_desc: str, curr_desc: str) -> bool:
    """Abrupt sky→ground vertical jump requiring a camera-descent bridge."""
    prev_l = (prev_desc or "").lower()
    curr_l = (curr_desc or "").lower()
    prev_clean = re.sub(r"^(?:dynamic camera tilt-down tracking descent[^\:]*:\s*)+", "", prev_l)
    curr_clean = re.sub(r"^(?:dynamic camera tilt-down tracking descent[^\:]*:\s*)+", "", curr_l)

    sky_kws = ["sky", "clouds", "cloudy", "rooftops", "high-angle establishing", "bầu trời", "trên trời", "trên không"]
    ground_kws = ["alley", "ground", "asphalt", "pavement", "kneeling", "hẻm", "mặt đất", "quỳ", "street level", "sidewalk", "cobblestone"]
    was_sky = any(kw in prev_clean for kw in sky_kws)
    now_ground = any(kw in curr_clean for kw in ground_kws)
    if any(kw in prev_clean for kw in ground_kws):
        was_sky = False
    has_descent_already = any(kw in curr_l for kw in ["descent", "tilt-down", "swoop", "downward", "hạ độ cao"])
    return was_sky and now_ground and not has_descent_already


def audit_storyboard(
    shots: List[ShotBriefDescription],
    characters: Optional[List[CharacterInScene]] = None,
) -> List[str]:
    """Structural comic-panel / cloning / repetition defects (series-agnostic)."""
    defects: List[str] = []
    consecutive_cam = 1
    prev_cam: Optional[int] = None

    for idx, shot in enumerate(shots):
        vis = shot.visual_desc or ""
        vis_l = vis.lower()
        aud = shot.audio_desc or ""
        has_dialogue = bool(re.search(r"\[(?:Speaker|Thought|Inner\s*Monologue)\]", aud, re.IGNORECASE))

        if re.search(r"\b(?:two\s+overlapping\s+poses|second\s+(?:smaller\s+)?pose\s+of|overlapping\s+poses\s+of)\b", vis_l):
            defects.append(f"Shot {idx}: clones multiple poses of one character in a single frame; split into separate single-pose sequential shots.")
        if (re.search(r"\bat\s+the\s+top\s+in\s+the\s+distance[^.]*<[^>]+>", vis, re.IGNORECASE)
                and re.search(r"\bin\s+the\s+(?:large\s+)?foreground[^.]*<[^>]+>", vis, re.IGNORECASE)):
            defects.append(f"Shot {idx}: shows the same character(s) twice (distance + foreground) in one action frame; keep one continuous moment.")
        if re.search(r"\b(?:two\s+(?:small\s+)?(?:close-up\s+)?panels\s+placed\s+side\s+by\s+side|in\s+the\s+left\s+panel\b.*\bin\s+the\s+right\s+panel)\b", vis_l):
            defects.append(f"Shot {idx}: uses split side-by-side comic inset panels; rewrite as sequential single-view anime shots.")
        if re.search(r"\bplain\s+(?:white|light)\s+background\b", vis_l) and (shot.transition_in or "cut") != "white_flash":
            defects.append(f"Shot {idx}: uses a stark blank white/light void background; replace with softly blurred location lighting.")
        if re.search(r"(?:^|\s)'s\b", vis) or re.search(r"\bpart\s+of\s+is\s+visible\b", vis_l):
            defects.append(f"Shot {idx}: has a dropped <Character> tag / broken grammar; restore the explicit character name.")
        if (not has_dialogue and shot.target_duration_sec is not None and shot.target_duration_sec > 1.5
                and any(kw in vis_l for kw in ["speed lines", "motion blur", "disorienting tilted", "dissolving into"])):
            defects.append(f"Shot {idx}: characterless speed/blur transition lingers for {shot.target_duration_sec}s; keep under 1.5s.")

        if prev_cam is not None and shot.cam_idx == prev_cam:
            consecutive_cam += 1
            if consecutive_cam == 4:
                defects.append(f"Shots {idx - 3}..{idx}: repeat the same static cam_idx={shot.cam_idx} 4+ times consecutively; cut to a close-up or reverse angle.")
        else:
            consecutive_cam = 1
        prev_cam = shot.cam_idx

    return defects


def _tag_speed_and_lightning(vis: str, aud: str, dur, has_dialogue: bool):
    """Generic Sakuga auto-tagging + lightning geometry (no character names)."""
    force_fx = extract_context_force_fx(vis)
    for pat in BANNED_ACTION_MOTION_PATTERNS:
        if pat.search(vis):
            vis = re.sub(pat, f"blurs in a blinding supersonic velocity streak and {force_fx}, securing the target in a single Sakuga frame skip without breaking forward momentum", vis)
            if "[AnimeFX: FrameSkip]" not in vis:
                vis = f"{vis} [AnimeFX: FrameSkip] [AnimeFX: SpeedStreak] [AnimeFX: Vanish]"
            if "[AnimeFX: FrameSkip]" not in aud:
                aud = f"{aud} [AnimeFX: FrameSkip] [AnimeFX: SpeedStreak] [AnimeFX: Vanish]"
            dur = 1.5

    is_speed_action = any(kw in vis.lower() for kw in SUPERHUMAN_SPEED_TRIGGERS)
    if is_speed_action:
        if any(kw in vis.lower() for kw in ["launch", "sprinter", "crouch", "ignite", "tụ lực", "xuất phát", "instinct", "bản năng"]):
            if "[AnimeFX: ActionBurst]" not in vis:
                vis = f"{vis} [AnimeFX: ActionBurst] [AnimeFX: Chromatic]"
            dur = 1.5
        elif any(kw in vis.lower() for kw in ["streak", "blitz", "vanish", "frame skip", "vụt qua", "biến mất"]):
            if "[AnimeFX: FrameSkip]" not in vis:
                vis = f"{vis} [AnimeFX: FrameSkip] [AnimeFX: SpeedStreak] [AnimeFX: Vanish]"
            dur = 1.2
        elif any(kw in vis.lower() for kw in ["skid", "screech", "brakes", "smoke", "empty", "phanh", "trống rỗng"]):
            if "[AnimeFX: SpeedLines]" not in vis:
                vis = f"{vis} [AnimeFX: SpeedLines] [AnimeFX: ScreenShake]"
            dur = 2.0
        elif any(kw in vis.lower() for kw in ["touchdown", "aftermath", "cradling", "safe", "tiếp đất", "landing"]):
            if "[AnimeFX: ActionBurst]" not in vis:
                vis = f"{vis} [AnimeFX: ActionBurst]"
            dur = 2.0

    has_active_lightning = any(kw in vis.lower() for kw in ["lightning", "tia sét", "sét", "thunderbolt", "electric bolt"])
    has_passive_wisps = bool(re.search(r"\b(?:faint|residual|fading|dissolv\w*|flicker\w*|aftermath|wisps)\b", vis.lower()))
    is_lightning = has_active_lightning or ("spark" in vis.lower() and not has_passive_wisps)
    if is_lightning:
        vis = re.sub(
            r"\b(?:thin|smooth|straight|constant)?\s*(?:diagonal\s+)?(?:cyan-white|cyan|blue)?\s*(?:energy\s+)?(?:streak|beam|line)\b",
            "violent, jagged cyan-blue electric lightning bolt with sharp branching tendrils and high-voltage crackle",
            vis, flags=re.IGNORECASE,
        )
        if not has_dialogue:
            if any(kw in vis.lower() for kw in ["impact", "detonat", "slam", "crash", "va chạm", "nổ"]):
                dur = 2.0
            elif any(kw in vis.lower() for kw in ["sky", "establishing", "plung", "descent", "tracking"]):
                dur = 1.0 if "sky" in vis.lower() else 1.5
    return vis, aud, dur


def lint_and_auto_correct_shot_brief(
    shot: ShotBriefDescription,
    shot_idx: int,
    total_shots: int,
    prev_shot: Optional[ShotBriefDescription] = None,
    characters: Optional[List[CharacterInScene]] = None,
) -> ShotBriefDescription:
    """Generic auto-correct for one ShotBriefDescription (no character/wardrobe injection)."""
    vis = shot.visual_desc or ""
    aud = shot.audio_desc or ""
    dur = shot.target_duration_sec
    trans = shot.transition_in or "cut"

    has_dialogue = bool(re.search(r"\[(?:Speaker|Thought)\]", aud, re.IGNORECASE))
    vis, aud, dur = _tag_speed_and_lightning(vis, aud, dur, has_dialogue)
    vis = strip_comic_artifacts(vis)

    # Soft discovery transition after an intense prior beat.
    if prev_shot is not None and trans == "cut":
        prev_vis_l = (prev_shot.visual_desc or "").lower()
        prev_aud_l = (prev_shot.audio_desc or "").lower()
        curr_vis_l = vis.lower()
        was_intense = (
            any(kw in prev_vis_l for kw in ["monologue", "teeth", "bitter", "agony", "stagger", "limp", "walk", "bước", "hẻm", "alley"])
            or bool(re.search(r"\[(?:Speaker|Sound Effect)\]", prev_aud_l, re.IGNORECASE))
        )
        is_discovery = bool(re.search(r"\b(?:spot|spots|spotted|notice|notices|noticed|over-the-shoulder|pov|nhìn\s+thấy|in\s+the\s+distance)\b", curr_vis_l))
        if was_intense and is_discovery:
            trans = "crossfade"

    # Dialogue duration protection vs. capping characterless blur transitions.
    has_dialogue = bool(re.search(r"\[(?:Speaker|Thought|Inner\s*Monologue)\]", aud, re.IGNORECASE))
    if has_dialogue:
        if dur is not None and dur < 4.0:
            dur = None
    else:
        is_blur_speed_transition = any(kw in vis.lower() for kw in ["speed lines", "speedline", "motion blur", "disorienting tilted", "dissolving into", "speed streak"])
        if dur is not None and dur > 1.5 and is_blur_speed_transition:
            dur = 1.5

    return shot.model_copy(update={
        "visual_desc": vis.strip(),
        "audio_desc": aud.strip(),
        "target_duration_sec": dur,
        "transition_in": trans,
    })


def lint_and_auto_correct_storyboard(
    shots: List[ShotBriefDescription],
    characters: Optional[List[CharacterInScene]] = None,
) -> List[ShotBriefDescription]:
    """Generic auto-correct for a whole storyboard: per-shot cleanup + altitude/anti-stagnation smoothing."""
    corrected_shots: List[ShotBriefDescription] = []
    n_shots = len(shots)
    for i, shot in enumerate(shots):
        prev = corrected_shots[i - 1] if i > 0 else None
        fixed = lint_and_auto_correct_shot_brief(shot=shot, shot_idx=i, total_shots=n_shots, prev_shot=prev, characters=characters)

        if (prev is not None and prev.cam_idx != fixed.cam_idx and (fixed.transition_in or "cut") == "cut"
                and detect_altitude_spatial_jump(prev.visual_desc, fixed.visual_desc)):
            lower_zone = extract_context_spatial_zone(fixed.visual_desc)
            if "[Camera: Rapid tilt-down" not in prev.visual_desc:
                corrected_shots[i - 1] = prev.model_copy(update={"visual_desc": f"{prev.visual_desc} [Camera: Rapid tilt-down tracking shot swooping downward toward {lower_zone}]"})
            fixed = fixed.model_copy(update={"visual_desc": f"Dynamic camera tilt-down tracking descent continuing rapidly from high altitude downward toward {lower_zone}: {fixed.visual_desc}"})

        if prev is not None:
            prev_vis_l = prev.visual_desc.lower()
            curr_vis_l = fixed.visual_desc.lower()
            is_prev_panting = any(kw in prev_vis_l for kw in ["panting", "coughing", "heaving", "breath puff", "thở dốc", "ho sặc"])
            is_curr_panting = any(kw in curr_vis_l for kw in ["panting", "coughing", "heaving", "breath puff", "thở dốc", "ho sặc"])
            if is_prev_panting and is_curr_panting:
                if not re.search(r"\b(?:glance|look\s+(?:at|down|over|around|toward)|spot|stand|stagger|đứng|nhìn|bước|shifting)\b", curr_vis_l):
                    fixed = fixed.model_copy(update={"visual_desc": f"Progressive physical reaction: Shifting from the previous cough, {extract_context_subject(fixed.visual_desc, characters)} grimaces through the pain, lifting his chin alertly: {fixed.visual_desc}"})

        corrected_shots.append(fixed)
    return corrected_shots


def lint_and_auto_correct_shot_description(
    sd: ShotDescription,
    characters: Optional[List[CharacterInScene]] = None,
) -> ShotDescription:
    """Generic auto-correct for a decomposed ShotDescription (ff/lf/motion), no identity injection."""
    vis = strip_comic_artifacts(sd.visual_desc or "")
    ff = strip_comic_artifacts(sd.ff_desc or sd.visual_desc or "")
    lf = strip_comic_artifacts(sd.lf_desc or sd.visual_desc or "")
    motion = sd.motion_desc or ""
    variation_reason = strip_comic_artifacts(getattr(sd, "variation_reason", "") or "")
    force_fx = extract_context_force_fx(vis)
    surface = extract_context_surface(vis)

    # Normalise then soften the motion prompt.
    motion = re.sub(r"^\s*[,;:\.\-–—]+\s*", "", motion).strip()
    motion = strip_comic_artifacts(motion)
    motion = sanitize_motion_prompt(motion)

    # Lightning geometry on keyframes (generic).
    if any(kw in vis.lower() for kw in ["lightning", "tia sét", "sét", "thunderbolt", "electric bolt"]):
        for field_name, val in (("ff", ff), ("lf", lf)):
            new_val = re.sub(
                r"\b(?:thin|smooth|straight|constant)?\s*(?:diagonal\s+)?(?:cyan-white|cyan|blue)?\s*(?:energy\s+)?(?:streak|beam|line)\b",
                "violent, jagged cyan-blue electric lightning bolt with sharp branching tendrils and high-voltage crackle",
                val, flags=re.IGNORECASE,
            )
            if field_name == "ff":
                ff = new_val
            else:
                lf = new_val

    for pat in BANNED_ACTION_MOTION_PATTERNS:
        if pat.search(motion):
            motion = re.sub(pat, f"blurs in a directional supersonic velocity rush and {force_fx}, securing the target in a single Sakuga frame skip", motion)
        if pat.search(ff):
            ff = re.sub(pat, f"launches forward on pure instinct, sudden blast of {force_fx} exploding backward from boots across {surface}", ff)
        if pat.search(lf):
            lf = re.sub(pat, f"vanishes into a blinding sonic speed streak, leaving empty {surface} behind", lf)

    return sd.model_copy(update={
        "visual_desc": vis.strip(),
        "variation_reason": variation_reason.strip(),
        "motion_desc": motion.strip(),
        "ff_desc": ff.strip(),
        "lf_desc": lf.strip(),
        "ff_vis_char_idxs": list(dict.fromkeys(sd.ff_vis_char_idxs or [])),
        "lf_vis_char_idxs": list(dict.fromkeys(sd.lf_vis_char_idxs or [])),
    })
