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

# Banned camera angle patterns that cause diffusion models to generate flat eye-level / sitcom framings or slanted diagonal postures
BANNED_EYE_LEVEL_PATTERNS = [
    re.compile(r"\bmedium\s+shot\s+at\s+eye\s+level\b", re.IGNORECASE),
    re.compile(r"\b(?:shot\s+at\s+)?eye\s*[-–—]?\s*level(?:\s+shot)?\b", re.IGNORECASE),
    re.compile(r"\bat\s+eye\s*[-–—]?\s*level\b", re.IGNORECASE),
    re.compile(r"\bmedium\s+3/4(?:\s+low-angle)?\s+shot\b", re.IGNORECASE),
    re.compile(r"\bmedium\s+3/4\s+shot\b", re.IGNORECASE),
    re.compile(r"\bmedium\s+3/4\b", re.IGNORECASE),
    re.compile(r"\bmedium\s+shot\b", re.IGNORECASE),
    re.compile(r"\bmedium\s+close-?up(?:\s+shot)?\b", re.IGNORECASE),
    re.compile(r"\bwaist-up(?:\s+shot)?\b", re.IGNORECASE),
    re.compile(r"\bflat\s+(?:camera\s+)?angle\b", re.IGNORECASE),
    re.compile(r"\bneutral\s+angle\b", re.IGNORECASE),
    re.compile(r"\bwith combat boots planted in the lower foreground[,\.]?\b", re.IGNORECASE),
    re.compile(r"Extreme low-angle shot from ground level looking up from [^\.]+\.\s*Camera placed low near the pavement aiming upward at [^,]+,\s*(?:with combat boots planted in the lower foreground,)?\s*,?\s*vertical walls rising straight into the dark sky,\s*and body hunched forward in deep exhaustion and pain\s*", re.IGNORECASE),
    re.compile(r"\(\s*\)"),
]

GENUINE_DYNAMIC_KEYWORDS = [
    "dutch-angle", "dutch angle", "canted angle", "tilted angle", "dynamic diagonal",
    "worm's eye", "worm's-eye", "steep low-angle", "steep low angle",
    "extreme low-angle", "extreme low angle", "low-angle shot from ground level",
    "ground-level", "ground level", "pavement-level", "pavement level",
    "low-angle", "low angle", "view from below",
    "overhead", "bird's-eye", "bird's eye", "top-down",
    "close-up on eyes", "extreme close-up", "extreme face close-up",
    "tracking shot", "plunging downward", "swooping downward",
    "over-the-shoulder", "pov shot",
    "two-shot", "extreme wide establishing", "wide establishing",
    "cinematic wide angle", "high-angle", "high angle"
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
    if re.search(r"\bno\s+characters?\s+(?:is|are)?\s*(?:clearly\s+)?visible\b", vis or "", re.IGNORECASE):
        return "the scene"
    behind_match = re.search(r"(?:from\s+behind|past)\s+(?:<([^>]+)>|([A-Z][a-zA-Z0-9_\s\(\)]+?)(?=\s+(?:in|looking|who|with|\.|\,)))", vis)
    if behind_match:
        subj = behind_match.group(1) or behind_match.group(2)
        return f"<{subj.strip()}>"
    match = re.search(r"<([^>]+)>", vis)
    if "<Kayden (Human)>" in vis or "Kayden (Human)" in vis:
        return "<Kayden (Human)>"
    if match:
        return f"<{match.group(1)}>"
    if characters:
        for c in characters:
            if c.is_visible and c.identifier_in_scene and c.identifier_in_scene.lower() != "narrator":
                return f"<{c.identifier_in_scene}>"
    return "The subject"


def extract_context_surface(text: str) -> str:
    """Extract physical ground/surface from text keywords."""
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


def _extract_visible_character_tags(cleaned: str) -> List[str]:
    """Return distinct visible <Character> tags, ignoring clauses like '<Name> is not visible'."""
    masked = re.sub(r"<[^>]+>\s+(?:is|are)\s+not\s+visible\b", "", cleaned, flags=re.IGNORECASE)
    masked = re.sub(r"\bno\s+other\s+(?:part\s+of\s+<[^>]+>|characters?)[^.]*\.", "", masked, flags=re.IGNORECASE)
    tags = [t.strip() for t in re.findall(r"<([^>]+)>", masked) if t.strip()]
    return list(dict.fromkeys(tags))


def purify_and_enforce_dynamic_camera_angles(
    text: str,
    shot_idx: int = 0,
    surface: str = "the ground",
    has_dialogue: bool = False,
    subject: str = "The character",
) -> str:
    """Enforce dynamic anime camera angles without destroying sentence structure or injecting clashing alley combat boilerplate."""
    if not text:
        return text

    # Strip any previously injected director guard camera prefixes to avoid compounding
    cleaned = re.sub(
        r"^(?:(?:Dramatic|Extreme|Low-angle|High-angle|Dynamic|Ground-level|Wide cinematic)\s+"
        r"(?:tracking shot|close-up framing|shot looking steeply up|low-angle perspective|two-shot)[^\:]*:\s*)+",
        "",
        text,
    )

    # 1. Grammar-preserving inline upgrades (replace shot/angle terms in-place instead of deleting words mid-sentence)
    had_banned = False
    inline_replacements = [
        (r"\bmedium\s+close-?up(?:\s+shot)?\b", "tight close-up"),
        (r"\bmedium\s+3/4(?:\s+low-angle)?\s+shot\b", "dynamic low-angle 3/4 shot"),
        (r"\bmedium\s+3/4\b", "dynamic low-angle 3/4"),
        (r"\bmedium\s+shot\s+at\s+eye\s*[-–—]?\s*level\b", "dynamic low-angle shot"),
        (r"\bmedium\s+shot\b", "dynamic shot"),
        (r"\bshot\s+at\s+eye\s*[-–—]?\s*level\b", "low-angle shot"),
        (r"\bat\s+(?:the\s+)?([a-zA-Z0-9_]+'s)\s+eye\s*[-–—]?\s*level\b", r"low-angle at \1 height"),
        (r"\bat\s+eye\s*[-–—]?\s*level\b", "from a slight low angle"),
        (r"\beye\s*[-–—]?\s*level(?:\s+shot)?\b", "slight low-angle"),
        (r"\bwaist-up(?:\s+shot)?\b", "angled upper-body framing"),
        (r"\bflat\s+(?:camera\s+)?angle\b", "dynamic camera angle"),
        (r"\bneutral\s+angle\b", "angled camera view"),
    ]
    for pat_str, repl in inline_replacements:
        cleaned, n_sub = re.subn(pat_str, repl, cleaned, flags=re.IGNORECASE)
        if n_sub:
            had_banned = True

    for pat in BANNED_EYE_LEVEL_PATTERNS:
        if pat.search(cleaned):
            had_banned = True
            cleaned = pat.sub("", cleaned)

    # Fix indefinite article mismatch ("an tight", "an static", "a extreme") & spacing/punctuation
    cleaned = re.sub(r"\bAn\s+(?=(?:tight|dynamic|low-angle|high-angle|static|slight|wide)\b)", "A ", cleaned)
    cleaned = re.sub(r"\ban\s+(?=(?:tight|dynamic|low-angle|high-angle|static|slight|wide)\b)", "a ", cleaned)
    cleaned = re.sub(r"\bA\s+(?=(?:angled|extreme|overhead|over-the-shoulder)\b)", "An ", cleaned)
    cleaned = re.sub(r"\ba\s+(?=(?:angled|extreme|overhead|over-the-shoulder)\b)", "an ", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"^\s*[,;:\.\-–—]+\s*", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

    # 2. Check if text already contains genuine dynamic angle keywords
    has_dynamic = any(kw in cleaned.lower() for kw in GENUINE_DYNAMIC_KEYWORDS)
    if has_dynamic:
        # Also allow specialized panic-eyes upgrade if original specifically said "wide eyes" + panic/alarm & had no close-up yet
        if not (had_banned and any(kw in cleaned.lower() for kw in ["panic", "wide eyes", "alarm", "hoảng loạn", "sợ hãi", "terror"])):
            return cleaned

    # 3. Determine concise, context-appropriate camera framing ONLY if no dynamic angle exists yet
    cleaned_l = cleaned.lower()
    is_indoor_calm = any(
        kw in cleaned_l
        for kw in [
            "pet shop", "shop", "store", "shelf wall", "cat snack", "cat food",
            "indoor", "interior", "supermarket", "room", "bedroom", "living room",
            "aisle", "house", "doorway", "hallway", "apartment", "home",
            "cushion", "rug", "carpet", "sofa", "couch", "table", "bowl",
            "milk", "kitchen", "intercom", "barefoot", "sofa",
        ]
    )
    has_real_cat = bool(
        re.search(r"\bcat\b(?!\s+(?:snack|snacks|food|treat|treats|toy|toys|illustration|illustrations|sign|product|products)|-icon)", cleaned_l)
    )
    has_exterior_stone_wall = (
        bool(re.search(r"\b(?:stone|brick|alley|cracked|exterior)\s+wall\b", cleaned_l))
        or (bool(re.search(r"\bwall\b", cleaned_l)) and not is_indoor_calm and any(k in cleaned_l for k in ["alley", "asphalt", "pavement", "street", "night"]))
    )
    visible_tags = _extract_visible_character_tags(cleaned)
    is_multi_character = (
        bool(re.search(r"\btwo-?shot\b|\bboth\s+(?:figures|characters)\b", cleaned_l))
        or len(visible_tags) >= 2
    )
    no_character = (
        bool(re.search(r"\bno\s+characters?\s+(?:is|are)?\s*(?:clearly\s+)?visible\b", cleaned_l))
        or subject == "the scene"
    )
    is_upright_motion = (
        any(kw in cleaned_l for kw in ["stagger", "limp", "bước", "lê bước", "đứng dậy"])
        or bool(re.search(r"\b(?:push(?:es|ing)?\s+upright|struggl(?:es?|ing)\s+to\s+stand|pushes?\s+himself\s+up)\b", cleaned_l))
    )
    is_kinetic_speed = any(kw in cleaned_l for kw in ["supersonic", "lightning speed", "sprinter", "activation", "blitz", "dash", "launch"])

    prefix = ""
    is_prologue_battle = ("kayden" in subject.lower() or "kayden" in cleaned_l or "warrior" in cleaned_l) and (
        has_exterior_stone_wall or any(kw in cleaned_l for kw in ["alley", "night", "kneel", "cough", "clutch", "wound", "bleed"])
    )
    if no_character:
        if any(kw in cleaned_l for kw in ["sky", "clouds", "cloudy", "rooftops", "bầu trời"]):
            prefix = "High-angle wide establishing shot looking across"
        elif any(kw in cleaned_l for kw in ["alley", "street", "houses", "wall", "ground", "pavement"]) and not is_indoor_calm:
            prefix = f"Wide low-angle establishing view across {surface}"
        else:
            prefix = "Wide cinematic establishing shot"
    elif any(kw in cleaned_l for kw in ["panic", "wide eyes", "alarm", "hoảng loạn", "sợ hãi", "terror"]):
        prefix = "Extreme close-up on eyes, dramatic pupils dilated in sudden horror"
    elif is_kinetic_speed and not is_indoor_calm:
        prefix = f"Dynamic low-angle perspective looking up from {surface}, boots coiled in a low sprinter stance"
    elif re.search(r"\b(?:danger|approaching|hurtling|lao tới|vehicle|threat|speeding|truck|suv|van)\b", cleaned_l) and not re.search(r"\bcardboard\b", cleaned_l):
        prefix = f"Extreme low-angle ground-level perspective looking steeply up from {surface} at the oncoming threat"
    elif any(kw in cleaned_l for kw in ["victim", "frozen", "helpless", "chasing", "target"]):
        prefix = f"High overhead bird's-eye perspective looking down at {surface}"
    elif any(kw in cleaned_l for kw in ["sky", "clouds", "cloudy", "rooftops", "bầu trời"]) and not any(kw in cleaned_l for kw in ["ground", "pavement", "asphalt", "alley", "kneel", "cough"]):
        prefix = "High-angle establishing shot looking over"
    elif is_indoor_calm:
        if any(kw in cleaned_l for kw in ["wide establishing", "aisle", "stocked shelf"]):
            prefix = "Wide cinematic low-angle indoor establishing shot"
        else:
            prefix = "Dramatic low-angle close-up framing inside the room"
    elif is_multi_character or any(kw in cleaned_l for kw in ["vortex", "transform", "biến hình", "xoáy"]):
        prefix = f"Ground-level two-shot looking steeply upward from near {surface}"
    elif re.search(r"\b(?:over-the-shoulder|pov|spot|notice)\b", cleaned_l) or (has_real_cat and "behind" in cleaned_l):
        prefix = f"Low-angle over-the-shoulder POV shot framed from near {surface} looking past {subject} toward the target"
    elif is_prologue_battle:
        if is_upright_motion:
            prefix = f"Low-angle tracking shot along {surface}, camera placed low near the pavement tracking {subject} pushing upright and staggering forward along the vertical stone wall"
        elif has_dialogue or any(kw in cleaned_l for kw in ["grimace", "teeth", "glare", "monologue", "kuso", "nghiến răng"]):
            prefix = "Dramatic low-angle close-up framing looking steeply up from chest level toward the jaw and face, with the vertical stone wall rising straight on the right"
        else:
            prefix = f"Low-angle shot looking steeply up from ground level at {surface}, camera angled upward at {subject} who hunches forward in deep exhaustion and pain, with the vertical stone wall rising straight into the night sky"
    elif has_exterior_stone_wall or any(kw in cleaned_l for kw in ["kneel", "cough", "brace", "clutch", "wound", "bleed", "struggle", "stagger", "quỳ", "ho", "dựa", "ôm ngực", "vết thương", "đứng dậy"]) or is_upright_motion:
        if is_upright_motion:
            prefix = f"Low-angle tracking shot along {surface}, camera placed low near the pavement tracking {subject} moving forward"
        elif has_dialogue or any(kw in cleaned_l for kw in ["grimace", "teeth", "glare", "monologue", "kuso", "nghiến răng"]):
            prefix = "Dramatic low-angle close-up framing looking steeply up from chest level toward the jaw and face"
        else:
            prefix = f"Dynamic low-angle shot looking steeply up from {surface} toward {subject}"
    elif any(kw in cleaned_l for kw in ["walk", "limp", "step", "drag", "bước", "đi", "lê bước", "street", "corner", "avenue"]):
        prefix = f"Low ground-level tracking shot along {surface}, camera placed low near the pavement tracking forward motion in dynamic linear perspective"
    elif has_dialogue or any(kw in cleaned_l for kw in ["grimace", "teeth", "glare", "monologue", "kuso", "nghiến răng"]):
        prefix = "Dramatic low-angle close-up framing looking steeply up from chest level toward the jaw and face, with vertical background perspective"
    elif re.search(r"\b(?:over-the-shoulder|pov|spot|notice|nhìn thấy)\b", cleaned_l) or has_real_cat or "mèo" in cleaned_l:
        prefix = f"Low-angle over-the-shoulder POV shot framed from near {surface} looking past {subject} toward the target"
    else:
        prefix = f"Dramatic low-angle worm's-eye perspective looking steeply up from {surface}"

    if re.match(r"^(?:in|on|at|inside|under|beside|along)\b", cleaned, re.IGNORECASE):
        return f"{prefix} {cleaned}".strip()
    return f"{prefix}: {cleaned}".strip()


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
                # Clean Sakuga Aftermath / Safe Recovery - No clumsy real-world friction
                if "[AnimeFX: ActionBurst]" not in vis:
                    vis = f"{vis} [AnimeFX: ActionBurst]"
                dur = 2.0

        has_dialogue = bool(re.search(r"\[(?:Speaker|Thought)\]", aud, re.IGNORECASE))

        # 2b. Lightning Sakuga Geometry & Snappy Pacing (Bans sluggish laser streaks)
        has_active_lightning = any(
            kw in vis.lower()
            for kw in ["lightning", "tia sét", "sét", "thunderbolt", "electric bolt"]
        )
        has_passive_wisps = bool(
            re.search(r"\b(?:faint|residual|fading|dissolv\w*|flicker\w*|aftermath|wisps)\b", vis.lower())
        )
        is_lightning = has_active_lightning or ("spark" in vis.lower() and not has_passive_wisps)
        if is_lightning:
            # Upgrade flat/smooth laser streak phrases into authentic jagged forked lightning
            vis = re.sub(
                r"\b(?:thin|smooth|straight|constant)?\s*(?:diagonal\s+)?(?:cyan-white|cyan|blue)?\s*(?:energy\s+)?(?:streak|beam|line)\b",
                "violent, jagged cyan-blue electric lightning bolt with sharp branching tendrils and high-voltage crackle",
                vis,
                flags=re.IGNORECASE,
            )
            # Upgrade generic thunder sound effects to Chidori 1000-birds high-frequency electrical screeches
            aud = re.sub(
                r"\[Sound\s*Effect\]\s*(?:Sudden\s+)?(?:high-voltage\s+)?thunder\s+crackle[^\.\!\n]*",
                "[Sound Effect] High-voltage Chidori 1000-birds screeching and chirping, piercing high-frequency ozone lightning crackle BZZZZ-TWEET-KZZZT!",
                aud,
                flags=re.IGNORECASE,
            )
            aud = re.sub(
                r"\[Sound\s*Effect\]\s*(?:ROARING-THUNDER-WHOOSH|High-speed\s+lightning\s+plunge)[^\.\!\n]*",
                "[Sound Effect] Chidori lightning plunge, piercing 1000-birds electrical screech and high-frequency ozone shockwave!",
                aud,
                flags=re.IGNORECASE,
            )
            aud = re.sub(
                r"\[Sound\s*Effect\]\s*(?:ELECTRICAL\s+EXPLOSION|\s*thunder\s+blast)[^\.\!\n]*",
                "[Sound Effect] Violent Chidori lightning detonation, high-voltage 1000-birds electric screech and sizzling electrical discharge!",
                aud,
                flags=re.IGNORECASE,
            )
            aud = re.sub(
                r"\[Sound\s*Effect\]\s*High-voltage\s+surge\s+humming,\s*electrical\s+vortex\s+roar\s+BZZZZT-FLASH!",
                "[Sound Effect] Chidori Raikiri 1000-birds electrical surge, intense chirping and screeching high-voltage lightning vortex roar BZZZZT-CHIRP-FLASH!",
                aud,
                flags=re.IGNORECASE,
            )
            if not has_dialogue:
                if any(kw in vis.lower() for kw in ["impact", "detonat", "slam", "crash", "va chạm", "nổ"]):
                    dur = 2.0
                elif any(kw in vis.lower() for kw in ["sky", "establishing", "plung", "descent", "tracking"]):
                    dur = 1.0 if "sky" in vis.lower() else 1.5

        # 2c. Spatial Anchor & Hand Consistency (Enforce anatomical & environmental anchor for stone wall combat)
        is_indoor_calm = any(
            kw in vis.lower()
            for kw in [
                "pet shop", "store", "tiệm", "cửa hàng", "shelf", "shelves",
                "kệ", "room", "bedroom", "living room", "couch", "sofa",
                "table", "indoor", "checkout"
            ]
        )
        is_stone_wall_combat = "kayden" in subject.lower() or "kayden" in vis.lower() or "warrior" in vis.lower() or ("stone wall" in vis.lower() and not is_indoor_calm and "shelf" not in vis.lower())
        if is_stone_wall_combat:
            wall_on_right = (
                bool(re.search(r"\b(?:stone|brick|concrete|cracked|exterior)\s+wall\b.*?\b(?:on the right|bên phải)\b", vis, re.IGNORECASE))
                or "stone wall on the right" in vis.lower()
            ) and "shelf" not in vis.lower()
            wall_on_left = (
                bool(re.search(r"\b(?:stone|brick|concrete|cracked|exterior)\s+wall\b.*?\b(?:on the left|bên trái)\b", vis, re.IGNORECASE))
                or "stone wall on the left" in vis.lower()
            ) and "shelf" not in vis.lower()

            if wall_on_right:
                # When wall is on the right, the right hand is anchored to the wall, left hand clutches wounds
                vis = re.sub(r"\b(?:extended\s+)?left\s+hand\s+(?:pressed|braced|placed|on)\b.*?\b(?:wall|stone)\b", "bloodied right hand firmly braced against the stone wall on the right", vis, flags=re.IGNORECASE)
                vis = re.sub(r"\bone\s+hand\s+(?:pressed|braced|on)\b.*?\b(?:wall|stone)\b", "bloodied right hand firmly braced against the stone wall on the right", vis, flags=re.IGNORECASE)
                if "clutch" not in vis.lower() and any(kw in vis.lower() for kw in ["wound", "bleed", "chest", "torso"]):
                    if "left hand" not in vis.lower() and not is_indoor_calm and "jiwoo" not in vis.lower() and "hojin" not in vis.lower():
                        vis = f"{vis}, his trembling left hand clutching his wounded bleeding chest"
            elif wall_on_left:
                vis = re.sub(r"\b(?:extended\s+)?right\s+hand\s+(?:pressed|braced|placed|on)\b.*?\b(?:wall|stone)\b", "bloodied left hand firmly braced against the stone wall on the left", vis, flags=re.IGNORECASE)

        # 2d. Character Battle Injury & Wardrobe Continuity Enforcement (Kayden / Warrior only)
        if "kayden" in subject.lower() or "kayden" in vis.lower() or "warrior" in vis.lower():
            vis = cls.enforce_character_injury_continuity(vis)
        vis = cls.enforce_character_facial_integrity(vis)
        vis = cls.enforce_gaze_direction_continuity(vis)

        # 2e. Physical Recovery & Agony Pacing Protection (Bans rushing/skipping heavy collapse & coughing)
        is_recovery_pain = any(kw in vis.lower() for kw in ["kneel", "cough", "collapse", "skid", "quỳ", "ho sặc", "gục", "pain", "battered", "heaving", "stagger", "limp", "lê lết", "drag", "step by step"])
        if is_recovery_pain and not is_speed_action and not has_dialogue:
            if dur is not None and dur < 5.0:
                dur = 5.0  # Give proper extended breathing space for physical recovery and staggering struggle!

        # 2f. Cinematic Transition from Intense Monologue/Pain/Staggering Walk into Discovery (e.g. spotting target/cat in distance)
        if prev_shot is not None and trans == "cut":
            prev_vis_l = (prev_shot.visual_desc or "").lower()
            prev_aud_l = (prev_shot.audio_desc or "").lower()
            curr_vis_l = vis.lower()
            was_monologue_pain = (
                any(kw in prev_vis_l for kw in ["monologue", "kuso", "bleeding", "wound", "teeth", "bitter", "agony", "stagger", "limp", "walk", "bước", "hẻm", "alley"])
                or bool(re.search(r"\[(?:Speaker|Sound Effect)\]", prev_aud_l, re.IGNORECASE))
            )
            is_discovery = bool(
                re.search(
                    r"\b(?:spot|spots|spotted|notice|notices|noticed|cats?|mèo|over-the-shoulder|pov|nhìn\s+thấy|in\s+the\s+distance)\b",
                    curr_vis_l,
                )
            ) and not bool(re.search(r"\bcat-icon\b", curr_vis_l))
            if was_monologue_pain and is_discovery:
                trans = "crossfade"

        # 3. Dramatic Camera Mapping (Enforce cinematic angles, ban eye-level monotony)
        vis = purify_and_enforce_dynamic_camera_angles(
            text=vis,
            shot_idx=shot_idx,
            surface=surface,
            has_dialogue=has_dialogue,
            subject=subject,
        )
        vis = cls.sanitize_comic_panel_artifacts(vis)

        # 4. Dialogue protection: Never abbreviate or speed-up shots with spoken dialogue
        has_dialogue = bool(re.search(r"\[(?:Speaker|Thought|Inner\s*Monologue)\]", aud, re.IGNORECASE))
        if has_dialogue:
            if dur is not None and dur < 4.0:
                dur = None  # Preserve proper duration for speech clarity and dynamic Ma cushion!
        else:
            # Cap overly long characterless speedline/motion-blur transitions
            vis_l_check = vis.lower()
            is_blur_speed_transition = any(
                kw in vis_l_check
                for kw in [
                    "speed lines",
                    "speedline",
                    "motion blur",
                    "disorienting tilted",
                    "dissolving into",
                    "speed streak",
                ]
            )
            if dur is not None and dur > 1.5 and is_blur_speed_transition:
                dur = 1.5

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
            # Smooth vertical / altitude jumps only on direct camera cuts (not same cam_idx holds or explicit fade/crossfade transitions)
            if (
                prev is not None
                and prev.cam_idx != fixed.cam_idx
                and (fixed.transition_in or "cut") == "cut"
                and cls.detect_altitude_spatial_jump(prev.visual_desc, fixed.visual_desc)
            ):
                lower_zone = extract_context_spatial_zone(fixed.visual_desc)
                # Update previous shot to swoop downward
                if "[Camera: Rapid tilt-down" not in prev.visual_desc:
                    prev_updated_vis = f"{prev.visual_desc} [Camera: Rapid tilt-down tracking shot swooping downward toward {lower_zone}]"
                    corrected_shots[i - 1] = prev.model_copy(update={"visual_desc": prev_updated_vis})
                # Update current shot to smoothly continue the descent
                curr_vis = f"Dynamic camera tilt-down tracking descent continuing rapidly from high altitude downward toward {lower_zone}: {fixed.visual_desc}"
                fixed = fixed.model_copy(update={"visual_desc": curr_vis})

            # Anti-Stagnation Rule: Prevent consecutive shots from repeating the exact same breath/pant cycle
            if prev is not None:
                prev_vis_l = prev.visual_desc.lower()
                curr_vis_l = fixed.visual_desc.lower()
                is_prev_panting = any(kw in prev_vis_l for kw in ["panting", "coughing", "heaving", "breath puff", "thở dốc", "ho sặc"])
                is_curr_panting = any(kw in curr_vis_l for kw in ["panting", "coughing", "heaving", "breath puff", "thở dốc", "ho sặc"])

                if is_prev_panting and is_curr_panting:
                    if not re.search(r"\b(?:glance|look\s+(?:at|down|over|around|toward)|spot|stand|stagger|đứng|nhìn|bước|shifting)\b", curr_vis_l):
                        curr_updated = f"Progressive physical reaction: Shifting from the previous cough, {extract_context_subject(fixed.visual_desc, characters)} grimaces through the pain, lifting his chin alertly: {fixed.visual_desc}"
                        fixed = fixed.model_copy(update={"visual_desc": curr_updated})

            corrected_shots.append(fixed)

        return corrected_shots

    @classmethod
    def sanitize_comic_panel_artifacts(cls, text: str) -> str:
        """Strip manga layout artifacts, void backgrounds, phantom duplicates, and uniform drift."""
        if not text:
            return ""
        updated = text
        updated = re.sub(
            r"\band\s+dark\s+radial\s+tension\s+lines\s+converge\s+inward\s+across\s+the\s+frame\b",
            "with dramatic shadowed vignette framing around the subject",
            updated,
            flags=re.IGNORECASE,
        )
        updated = re.sub(
            r"\bwith\s+(?:small|faint)\s+(?:black|white)\s+sparkle\s+marks(?:\s+(?:at|in)\s+[^,\.;]+)?",
            "with subtle warm atmospheric bokeh",
            updated,
            flags=re.IGNORECASE,
        )
        updated = re.sub(
            r"\b(?:small\s+|faint\s+)?(?:black|white)\s+sparkle\s+marks(?:\s+(?:at|in)\s+[^,\.;]+)?",
            "subtle warm atmospheric bokeh",
            updated,
            flags=re.IGNORECASE,
        )
        updated = re.sub(
            r"\bagainst\s+(?:a|the)\s+plain\s+(?:white|light)\s+background(?:\s+with\s+no\s+location\s+details)?\b",
            "against a softly blurred scene background",
            updated,
            flags=re.IGNORECASE,
        )
        updated = re.sub(
            r"\bon\s+(?:a|the)\s+plain\s+(?:white|light)\s+background(?:\s+with\s+no\s+location\s+details)?\b",
            "in a softly blurred scene environment",
            updated,
            flags=re.IGNORECASE,
        )
        updated = re.sub(
            r"\b(?:with|in|over)\s+(?:a|the)?\s*plain\s+(?:white|light)\s+background(?:\s+with\s+no\s+location\s+details)?\b",
            "with a softly blurred scene background",
            updated,
            flags=re.IGNORECASE,
        )
        updated = re.sub(
            r"\b(?:a|the)?\s*plain\s+(?:white|light)\s+background(?:\s+with\s+no\s+location\s+details)?\b",
            "a softly blurred scene background",
            updated,
            flags=re.IGNORECASE,
        )
        return re.sub(r"  +", " ", updated).strip()

    @classmethod
    def audit_storyboard_defects(
        cls,
        shots: List[ShotBriefDescription],
        characters: Optional[List[CharacterInScene]] = None,
    ) -> List[str]:
        """Detect structural comic-panel, cloning, repetition, and pacing defects in a storyboard."""
        defects: List[str] = []
        consecutive_cam = 1
        prev_cam: Optional[int] = None

        for idx, shot in enumerate(shots):
            vis = shot.visual_desc or ""
            vis_l = vis.lower()
            aud = shot.audio_desc or ""
            has_dialogue = bool(re.search(r"\[(?:Speaker|Thought|Inner\s*Monologue)\]", aud, re.IGNORECASE))

            if re.search(
                r"\b(?:two\s+overlapping\s+poses|second\s+(?:smaller\s+)?pose\s+of|overlapping\s+poses\s+of)\b",
                vis_l,
            ):
                defects.append(
                    f"Shot {idx}: clones multiple poses of one character in a single frame; split into separate single-pose sequential shots."
                )

            if (
                re.search(r"\bat\s+the\s+top\s+in\s+the\s+distance[^.]*<[^>]+>", vis, re.IGNORECASE)
                and re.search(r"\bin\s+the\s+(?:large\s+)?foreground[^.]*<[^>]+>", vis, re.IGNORECASE)
            ):
                defects.append(
                    f"Shot {idx}: shows the same character(s) twice (distance + foreground) in one action frame; keep one continuous moment."
                )

            if re.search(
                r"\b(?:two\s+(?:small\s+)?(?:close-up\s+)?panels\s+placed\s+side\s+by\s+side|in\s+the\s+left\s+panel\b.*\bin\s+the\s+right\s+panel)\b",
                vis_l,
            ):
                defects.append(
                    f"Shot {idx}: uses split side-by-side comic inset panels; rewrite as sequential single-view anime shots."
                )

            if re.search(r"\bplain\s+(?:white|light)\s+background\b", vis_l) and (shot.transition_in or "cut") != "white_flash":
                defects.append(
                    f"Shot {idx}: uses a stark blank white/light void background; replace with softly blurred location lighting."
                )

            if (
                re.search(r"(?:^|\s)'s\b", vis)
                or re.search(r"\bpart\s+of\s+is\s+visible\b", vis_l)
            ):
                defects.append(
                    f"Shot {idx}: has a dropped <Character> tag / broken grammar (e.g. \"'s\" or \"of is visible\"); restore the explicit character name."
                )

            if (
                not has_dialogue
                and shot.target_duration_sec is not None
                and shot.target_duration_sec > 1.5
                and any(kw in vis_l for kw in ["speed lines", "motion blur", "disorienting tilted", "dissolving into"])
            ):
                defects.append(
                    f"Shot {idx}: characterless speed/blur transition lingers for {shot.target_duration_sec}s; keep under 1.5s."
                )

            if prev_cam is not None and shot.cam_idx == prev_cam:
                consecutive_cam += 1
                if consecutive_cam == 4:
                    defects.append(
                        f"Shots {idx - 3}..{idx}: repeat the same static cam_idx={shot.cam_idx} 4+ times consecutively; cut to a close-up or reverse angle."
                    )
            else:
                consecutive_cam = 1
            prev_cam = shot.cam_idx

        return defects

    @classmethod
    def enforce_character_injury_continuity(cls, text: str) -> str:
        """Enforce character battle injury, shredded wardrobe, and cyan spark continuity across shots.
        
        Specifically prevents rear and over-the-shoulder views from resetting wounded characters 
        to clean, undamaged coats or unblemished skin, and ensures bare chest wounds are vividly 
        described with deep bleeding lacerations across all angles.
        """
        if not text:
            return ""
        
        updated = text
        clean_coat_pats = [
            r"\b(?:the\s+)?shoulder\s+and\s+upper\s+back\s+of\s+his\s+(?:ragged\s+)?dark\s+(?:charcoal(?:-slate)?\s+)?coat\b",
            r"\b(?:the\s+)?upper\s+back\s+and\s+shoulder\s+of\s+his\s+(?:ragged\s+)?dark\s+(?:charcoal(?:-slate)?\s+)?coat\b",
            r"\b(?:the\s+)?shoulder\s+of\s+his\s+(?:ragged\s+)?dark\s+(?:charcoal(?:-slate)?\s+)?coat\b",
            r"\b(?:the\s+)?upper\s+back\s+of\s+his\s+(?:ragged\s+)?dark\s+(?:charcoal(?:-slate)?\s+)?coat\b",
            r"\b(?:the\s+)?back\s+of\s+his\s+(?:ragged\s+)?dark\s+(?:charcoal(?:-slate)?\s+)?coat\b",
        ]
        replacement = (
            "the shoulder and upper back of his canonical matte medium-dark slate-grey hooded trench coat "
            "(muted charcoal-ash grey in shadow, shifting cleanly toward cool light grey along directional and rim-lit "
            "shoulders, hood folds, and outer sleeve planes—noticeably lighter grey than his black inner turtleneck and never pitch black), "
            "draped covering both shoulders and upper back with the hood resting down around his collar, and matching long "
            "slate-grey sleeves down to his wrists with subtle fraying, dusty crimson scuffs, and faint residual cyan electrical sparks"
        )
        if "canonical matte medium-dark slate-grey hooded trench coat" not in updated.lower():
            for pat in clean_coat_pats:
                if re.search(pat, updated, re.IGNORECASE):
                    updated = re.sub(pat, replacement, updated, flags=re.IGNORECASE)
                    break
        
        # Upgrade weak or faint wound descriptions on chest/torso to severe bleeding battle damage
        weak_wounds_pats = [
            (
                r"\b(?:dark-crimson\s+cuts\s+and\s+)?faintly\s+glowing\s+cyan\s+scratches\s+across\s+(?:his\s+)?(?:chest\s+and\s+arms|chest)\b",
                "deep dark-crimson bleeding battle lacerations and blood streams dripping down his muscular bare chest and abs, with bright crackling cyan electrical sparks along the open wounds",
            ),
            (
                r"\bdark-crimson\s+cuts\s+and\s+faintly\s+glowing\s+cyan\s+scratches\b",
                "severe deep dark-crimson bleeding battle lacerations and blood streams dripping down his muscular bare chest and abs, with bright crackling cyan electrical sparks along the wounds",
            ),
            (
                r"\bdeep\s+dark-crimson\s+cuts\s+and\s+faint\s+glowing\s+cyan\s+scratches\b",
                "severe deep dark-crimson bleeding battle lacerations and blood streams dripping down his muscular bare chest and abs, with bright crackling cyan electrical sparks along the wounds",
            ),
            (
                r"\bfaintly\s+glowing\s+cyan\s+scratches\b",
                "deep dark-crimson bleeding lacerations and crackling cyan sparks",
            ),
        ]
        for pat, repl in weak_wounds_pats:
            if re.search(pat, updated, re.IGNORECASE):
                updated = re.sub(pat, repl, updated, flags=re.IGNORECASE)
        
        updated = cls.enforce_character_facial_integrity(updated)
        return updated

    @classmethod
    def enforce_character_facial_integrity(cls, text: str) -> str:
        """Enforce character facial features and ban facial hair, stubble, beard, and aging drift.
        
        Kayden Break is canonically an extremely handsome, youthful Korean anime protagonist (~early 20s)
        with smooth porcelain pale skin, sharp refined clean-shaven jawline, and strictly NO facial hair,
        NO beard, NO mustache, NO stubble, and NO goatee.
        """
        if not text:
            return ""

        updated = text
        text_l = text.lower()
        text_without_cat_kayden = re.sub(
            r"<?\bkayden\s*\([^)]*(?:cat|tabby|feline|mèo)[^)]*\)>?",
            "",
            text_l,
        )
        has_kayden = (
            "kayden" in text_without_cat_kayden
            or "the tall man" in text_l
            or "the young adult man" in text_l
            or "protagonist" in text_l
        )

        if has_kayden:
            # Remove prior injected facial integrity blocks (including partially stripped ones) so re-linting is idempotent
            updated = re.sub(
                r"\s*Face\s*&\s*jawline:\s*Extremely handsome youthful clean-shaven Korean anime face.*?100%\s*smooth\s*hairless\s*face\s*and\s*chin\.",
                "",
                updated,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip()

            # Check if this shot has NO visible face (back-view, insert shot of hand/foot/object, head bowed, face hidden)
            is_no_face = bool(re.search(
                r"\b(seen from behind|from behind|back view|rear view|behind|head bowed|back turned|facing away|facing wall|over-the-shoulder|back of|looking down|obscured face|face obscured|insert shot|tightly focused on.*?hand|focused on.*?hand|grounded hand|close-up on.*?hand|hand pressed|knuckles|foot|feet|boots)\b",
                text_l,
            ))

            if is_no_face:
                # Do NOT append facial rule! Enforce framing and strictly forbid hallucinating a face!
                no_face_mandate = (
                    "[STRICT FRAMING & NO-FACE-REVEAL MANDATE]: Pure body/insert/rear framing without facial view. "
                    "Strictly DO NOT tilt camera, DO NOT turn character around, and DO NOT hallucinate or reveal any human face. "
                    "Maintain exact composition and angle from Image 0 without inventing unseen facial features."
                )
                updated = f"{updated} {no_face_mandate}".strip()
                return updated.strip()

            # 1. Strip any hallucinated facial hair or beard terms
            beard_patterns = [
                r"\b(?:with\s+a\s+)?(?:beard|stubble|mustache|facial\s+hair|goatee|unshaven|scruff|chinstrap|sideburns?|5\s*o'clock\s*shadow)\b",
                r"\b(?:dark\s+)?shading\s+along\s+(?:the\s+)?jawline\b",
                r"\bradiating\s+stubble\b",
                r"\bchin\s+shadow\b",
            ]
            for pat in beard_patterns:
                updated = re.sub(pat, "", updated, flags=re.IGNORECASE)

            # Check if battle injuries are present
            has_injuries = bool(re.search(
                r"\b(wound|wounded|injury|injured|bleed|bleeding|blood|bloodied|bruise|bruised|scraped|scratched|cut|cuts|slashed|battle wear|battle damage|torn)\b",
                text_l,
            ))

            # Clean up mouth blood dripping onto chin to prevent AI from drawing a dark chin goatee
            updated = re.sub(
                r"\b(?:blood\s+smear\s+on\s+chin|blood\s+(?:dripping|streaming|trailing)\s+down\s+(?:his\s+)?chin)\b",
                "a single small neat trickle of dark crimson blood at the corner of his lip, with chin completely hairless and smooth",
                updated,
                flags=re.IGNORECASE,
            )

            # 2. Upgrade aging / gaunt / haggard descriptors into handsome youthful descriptors while preserving battle wear
            if not has_injuries:
                aging_replacements = [
                    (r"\b(?:pale\s+)?exhausted\s+face\b", "smooth clean-shaven pale handsome face"),
                    (r"\bgaunt\s+(?:exhausted\s+)?face\b", "sharp handsome anime face"),
                    (r"\bgaunt\s+appearance\b", "sharp youthful anime features"),
                    (r"\bhollow\s+(?:dark\s+)?eye\s*sockets?\b", "sharp intense dark eyes"),
                    (r"\bhollow\s+eyes\b", "sharp intense dark eyes"),
                    (r"\bsunken\s+cheeks\b", "smooth porcelain pale cheeks"),
                    (r"\bhaggard\s+face\b", "resolute handsome face"),
                ]
            else:
                aging_replacements = [
                    (r"\bgaunt\s+(?:exhausted\s+)?face\b", "sharp handsome anime face with battle scrapes preserved"),
                    (r"\bgaunt\s+appearance\b", "sharp youthful anime features"),
                    (r"\bhollow\s+(?:dark\s+)?eye\s*sockets?\b", "sharp intense dark eyes"),
                    (r"\bhollow\s+eyes\b", "sharp intense dark eyes"),
                    (r"\bhaggard\s+face\b", "resolute handsome face with battle scrapes"),
                ]
            for pat, repl in aging_replacements:
                updated = re.sub(pat, repl, updated, flags=re.IGNORECASE)

            # 3. Append single canonical facial integrity rule
            facial_skin = "sharp refined clean-shaven anime jawline (with all battle cuts and bloodstains strictly preserved)" if has_injuries else "smooth porcelain pale skin, sharp refined anime jawline"
            facial_rule = (
                f"Face & jawline: Extremely handsome youthful clean-shaven Korean anime face, {facial_skin}, "
                "with strictly NO facial hair, NO beard, NO mustache, NO stubble, NO goatee, "
                "NO 5 o'clock shadow, 100% hairless face and chin."
            )
            updated = f"{updated} {facial_rule}".strip()

        return updated.strip()

    @classmethod
    def enforce_gaze_direction_continuity(cls, text: str) -> str:
        """Enforce character eye gaze direction and eliminate awkward fourth-wall breaking (looking at camera).
        
        Specifically ensures that when a character is observing a target (e.g. cat in the distance) or 
        towering over a target, their head and eyes are aimed strictly at the target, NEVER staring at the camera.
        """
        if not text:
            return ""
        
        updated = text
        text_l = text.lower()
        text_without_cat_kayden = re.sub(
            r"<?\bkayden\s*\([^)]*(?:cat|tabby|feline|mèo)[^)]*\)>?",
            "",
            text_l,
        )
        has_human = "kayden (human)" in text_l or "<kayden (human)>" in text_l or "kayden" in text_without_cat_kayden
        has_cat = "cat" in text_l or "mèo" in text_l or "tabby" in text_l or "neko" in text_l
        
        if has_human and has_cat:
            # Over-the-shoulder / spotting in distance context (Shot 6)
            if any(kw in text_l for kw in ["over-the-shoulder", "distance", "observes", "spot", "notice", "xa"]):
                if "strictly no looking at the camera" not in updated.lower():
                    updated = re.sub(r"\bbody\s+facing\s+forward\s+down\s+the\s+alley\s+toward\s+the\s+background\b", "body angled away from camera looking deep down the alley toward the distant cat", updated, flags=re.IGNORECASE)
                    updated = re.sub(r"\bface\s+(?:facing|turned\s+toward)\s+(?:the\s+)?camera\b", "face turned away from camera looking at the cat in the distance", updated, flags=re.IGNORECASE)
                    gaze_instruction = (
                        "Gaze & camera staging: Framed from behind Kayden's shoulder with the back of his head, messy bun, and torn coat shoulder in the side foreground facing away from the camera. "
                        "Kayden's head is turned facing directly down the alley toward the cat in the distance, his eyes completely locked onto the cat (strictly NO looking at the camera, zero fourth-wall eye contact)."
                    )
                    updated = f"{updated}. {gaze_instruction}"

            # Two-shot / towering over cat context (Shot 7)
            elif any(kw in text_l for kw in ["towering", "tower", "stands directly over", "two-shot", "vortex", "seal"]):
                if "strictly no looking at the camera" not in updated.lower():
                    updated = re.sub(r"\b(?:head\s+raised|facing\s+forward\s+and\s+downward)\b", "head tilted downward toward the cat", updated, flags=re.IGNORECASE)
                    gaze_instruction = (
                        "Gaze & eye direction: Kayden's head is tilted downward, his sharp dark eyes burning with intensity and locked directly down on the chubby cat at his feet (strictly NO looking at the camera or viewer, zero fourth-wall eye contact). "
                        "The chubby cat's head is tilted upward, with round wide startled eyes locked directly up at Kayden."
                    )
                    updated = f"{updated}. {gaze_instruction}"
        
        return updated

    @classmethod
    def enforce_anime_style(cls, text: str) -> str:
        """Enforce authentic Korean manhwa webtoon art style and palette matching the original Eleceed comic.
        Prevents diffusion models from drifting into photorealism, 3D CGI, live-action,
        or realistic animal textures, while preserving original webtoon color grading.
        """
        if not text:
            return ""
        if "[STYLE & COLOR PALETTE - ELECEED WEBTOON]" in text:
            return text

        anime_block = (
            "\n[STYLE & COLOR PALETTE - ELECEED WEBTOON]: Render cleanly in the authentic art style of the original Korean manhwa Eleceed. "
            "Faithfully preserve the character facial likeness, hairstyle, skin tone, and clothing colors directly from the character turnaround portraits (Image 0). "
            "Clean manhwa line art, soft natural shading, strictly faithful to the webtoon's original color palette and warm lighting. "
            "Cats and animals must be drawn in cute stylized manhwa art matching turnaround portraits, strictly NEVER photorealistic animals, NEVER realistic 3D fur textures. "
            "Strictly NO photorealism, NO 3D CGI rendering, NO realistic camera photograph textures."
        )
        return f"{text.rstrip()} {anime_block}"

    @classmethod
    def enforce_jiwoo_wardrobe(cls, text: str, is_at_home: bool = False) -> str:
        """Enforce Seo Jiwoo's canonical two-phase wardrobe continuity.
        
        Phase 1 (Outside / School / Street / Rescue - Scenes 5..17):
            Jiwoo is on his way home from school wearing his complete school uniform:
            school uniform blazer strictly matching his turnaround reference portrait (Image 0) in color and tone,
            worn over a crisp white collared shirt with a loose light blue-gray tie, dark trousers, and light blue-gray backpack.
            His facial likeness must strictly match Image 0.
        
        Phase 2 (At home / Yard feeding cats / Inside - Scenes 18..53):
            Jiwoo has arrived home and REMOVED his school blazer; he is wearing ONLY
            his crisp white collared shirt with a loose light blue-gray tie and dark charcoal trousers.
            Strictly NO blazer, NO suit jacket.
        """
        if not text:
            return ""

        text_l = text.lower()
        if "jiwoo" not in text_l and "seo jiwoo" not in text_l:
            return text

        updated = text

        if is_at_home:
            # Home phase: remove blazer/jacket if mentioned, enforce white shirt only
            if "[WARDROBE AT HOME]" not in updated:
                home_wardrobe_rule = (
                    "\n[WARDROBE AT HOME]: Seo Jiwoo has arrived home and REMOVED his school uniform blazer; "
                    "he is wearing ONLY his crisp white collared shirt with a loose light blue-gray tie and dark trousers. "
                    "Even if reference portraits show a blazer, do NOT render any blazer or jacket; render him in white shirt sleeves only. "
                    "Face and hair of Seo Jiwoo must strictly match his character turnaround: youthful rounded chin, short light ash-brown hair with bangs, bright expressive brown eyes."
                )
                updated = f"{updated.rstrip()} {home_wardrobe_rule}"
        else:
            # Outside phase: enforce school uniform blazer matching Image 0
            if "[WARDROBE SCHOOL UNIFORM]" not in updated:
                outside_wardrobe_rule = (
                    "\n[WARDROBE SCHOOL UNIFORM]: Seo Jiwoo is on his way home from school wearing his canonical "
                    "school uniform blazer strictly matching his turnaround portrait in color, tone, and design, "
                    "worn over a crisp white collared shirt with a loose light blue-gray tie, dark trousers, "
                    "and light blue-gray backpack straps. Do not alter the jacket color or strip his blazer. "
                    "Face and hair of Seo Jiwoo must strictly match his character turnaround: youthful rounded chin, short light ash-brown hair with bangs over forehead, bright expressive brown eyes, subtle soft cheek blush."
                )
                updated = f"{updated.rstrip()} {outside_wardrobe_rule}"

        return updated

    @classmethod
    def enforce_hojin_and_mother_wardrobe(cls, text: str) -> str:
        """Enforce canonical wardrobe and facial consistency for Hojin and Hojin's Mother."""
        if not text:
            return ""

        updated = text
        text_l = text.lower()

        if "hojin" in text_l and "[WARDROBE HOJIN]" not in updated:
            hojin_rule = (
                "\n[WARDROBE HOJIN]: Hojin (young boy, ~6-7 years old) must strictly match his turnaround portrait (Image 0/Image 1): "
                "soft rounded face, chubby cheeks, neat dark brown hair with bangs, warm dark eyes. He is wearing his canonical "
                "bright yellow jacket over a white t-shirt, blue denim jeans, and dark navy sneakers. Strictly NO striped shirt, "
                "NO cartoon chibi dot eyes, NO caricature expressions."
            )
            updated = f"{updated.rstrip()} {hojin_rule}"

        if ("mother" in text_l or "hojin's mother" in text_l) and "[WARDROBE HOJIN'S MOTHER]" not in updated:
            mother_rule = (
                "\n[WARDROBE HOJIN'S MOTHER]: Hojin's mother must strictly match her turnaround portrait: elegant Korean woman in her early 30s, "
                "shoulder-length straight dark brown hair with bangs, gentle features. She is wearing her canonical black cardigan over a light lavender/mauve "
                "t-shirt, blue jeans, and white sneakers. Strictly NO beige trench coat, NO hair bun."
            )
            updated = f"{updated.rstrip()} {mother_rule}"

        return updated

    @classmethod
    def enforce_pet_shop_and_store_owner(cls, text: str) -> str:
        """Enforce canonical aesthetics for pet shop interior and Store Owner.
        
        Guarantees:
        1. Pet shop shelves are cool lavender-blue / light gray modular metal shelves (gondola shelving)
           with clean pastel cat food pouches and kibble packages under bright, crisp white convenience store lighting.
           Strictly NO rustic brown wooden shelves, NO cluttered hanging feathers/plushies, NO shoujo fairy-tale yellow floating stars/sparkles or glowing bokeh.
        2. Store Owner is a friendly middle-aged Korean man (40s to 50s) with receding thinning dark brown hair, gentle laugh lines
           around smiling squinting eyes, round nose, wearing a light grayish-blue sweater/pullover over a collared shirt.
           Strictly NO young boy face, NO thick fluffy spike hair, NO miniature stiff doll posture.
        """
        if not text:
            return ""

        text_l = text.lower()
        is_pet_shop = any(kw in text_l for kw in ["pet shop", "cat food", "cat treat", "cat snack", "pet store", "shelf wall", "store owner"])
        if not is_pet_shop:
            return text

        updated = text
        # Remove any rustic wood or yellow sparkle mentions
        updated = re.sub(r"\b(?:rustic\s+)?brown\s+wood(?:en)?\s+shelves\b", "clean cool lavender-blue modular metal shelves", updated, flags=re.IGNORECASE)
        updated = re.sub(r"\b(?:bright\s+)?yellow\s+star\s+sparkles(?:\s+glowing\s+around[^\.,;]*)?[\.,;]?", "", updated, flags=re.IGNORECASE)
        updated = re.sub(r"\b(?:warm\s+atmospheric\s+)?bokeh(?:\s+lights?)?\b", "clean store lighting", updated, flags=re.IGNORECASE)

        if "[PET SHOP AESTHETICS & STORE OWNER]" not in updated:
            pet_shop_rule = (
                "\n[PET SHOP AESTHETICS & STORE OWNER]: "
                "Interior Shelving: Modern Korean pet shop / convenience store with cool lavender-blue and light gray modular metal gondola shelving, "
                "neatly stocked with pastel-colored cat food pouches (light purple, teal, pink, white kibble bags) and cans under bright crisp white fluorescent lighting. "
                "Strictly NO brown wooden shelves, NO cluttered toy wands, NO floating yellow shoujo star sparkles. "
                "Store Owner (Character 3): Friendly middle-aged Korean man (late 40s to 50s) matching his turnaround portrait: "
                "receding thinning dark brown hair, gentle smile lines around warm smiling eyes, round nose, wearing a light grayish-blue sweater over a collared shirt. "
                "Strictly NO teenage boy face, NO thick fluffy spike hair, NO stiff miniature doll-like posture."
            )
            updated = f"{updated.rstrip()} {pet_shop_rule}"

        return updated

    @classmethod
    def sanitize_motion_prompt_safety(cls, motion_desc: str) -> str:
        """Sanitize motion prompt to avoid triggering video diffusion safety filters (gore/violence refusals).
        
        Diffusion models already have visual wounds from reference keyframes. The motion prompt 
        only needs to describe camera and character kinetics (breathing, trembling, clenching jaw,
        glowing sparks) without triggering automated text safety/gore filters.
        """
        if not motion_desc:
            return ""
        
        sanitized = motion_desc
        # Phase 1: Strip stream clauses
        sanitized = re.sub(r'[\.,;]?\s*(?:blood|sweat)\s+streams?\s+continue\s+down.*?(?:and\s+)?(?=cyan\s+electrical)', ', while ', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'[\.,;]?\s*(?:blood|sweat)\s+streams?\s+continue\s+down[^\.,;]*[\.,;]?', '.', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'\b(?:blood|sweat)\s+streams?[^\.,;]*', '', sanitized, flags=re.IGNORECASE)

        # Phase 2: Specific composite phrase replacements
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

        # Phase 3: Blanket scrubbing of any residual gore / violence refusal trigger words
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

        # Clean up repeated spaces or leftover punctuation
        sanitized = re.sub(r"\s*,\s*,", ",", sanitized)
        sanitized = re.sub(r"\s{2,}", " ", sanitized).strip()
        return sanitized


    @classmethod
    def enforce_kayden_cat_bandages(cls, text: str, is_inside_house: bool = False) -> str:
        """Enforce canonical medical bandages on Kayden cat after rescue in house scenes."""
        if not text:
            return ""
        updated = text
        text_l = text.lower()
        if is_inside_house and ("kayden" in text_l and "cat" in text_l) and "[MANDATORY MEDICAL BANDAGES]" not in updated:
            bandage_rule = (
                "\n[MANDATORY MEDICAL BANDAGES]: Kayden (Fat Orange Tabby Cat) has clean white medical gauze bandages "
                "neatly wrapped around his plump chubby torso and body, and a small white gauze bandage patch on his forehead/ear, "
                "applied by Jiwoo to treat his severe battle wounds. Maintain these white medical bandages consistently across all frames."
            )
            updated = f"{updated.rstrip()} {bandage_rule}"
        return updated

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
        variation_reason = cls.sanitize_comic_panel_artifacts(getattr(sd, "variation_reason", "") or "")

        vis_char_pool = characters
        if characters and sd.ff_vis_char_idxs:
            filtered = [c for c in characters if c.idx in set(sd.ff_vis_char_idxs)]
            if filtered:
                vis_char_pool = filtered
        subject = extract_context_subject(vis, vis_char_pool)
        surface = extract_context_surface(vis)
        force_fx = extract_context_force_fx(vis)

        # Enforce dynamic camera angles across detailed keyframe descriptions (ban eye-level / medium shots)
        has_dialogue = bool(re.search(r"\[(?:Speaker|Thought)\]", sd.audio_desc or "", re.IGNORECASE))
        vis = purify_and_enforce_dynamic_camera_angles(
            text=vis,
            shot_idx=sd.idx,
            surface=surface,
            has_dialogue=has_dialogue,
            subject=subject,
        )
        ff = purify_and_enforce_dynamic_camera_angles(
            text=ff,
            shot_idx=sd.idx,
            surface=surface,
            has_dialogue=False,
            subject=subject,
        )
        lf = purify_and_enforce_dynamic_camera_angles(
            text=lf,
            shot_idx=sd.idx,
            surface=surface,
            has_dialogue=False,
            subject=subject,
        )

        # Enforce character battle injury and shredded wardrobe continuity across all frames (Kayden only)
        vis = cls.sanitize_comic_panel_artifacts(vis)
        ff = cls.sanitize_comic_panel_artifacts(ff)
        lf = cls.sanitize_comic_panel_artifacts(lf)
        is_kayden_battle = "kayden" in subject.lower() or any("kayden" in x.lower() for x in [vis, ff, lf])
        if is_kayden_battle:
            vis = cls.enforce_character_injury_continuity(vis)
            ff = cls.enforce_character_injury_continuity(ff)
            lf = cls.enforce_character_injury_continuity(lf)
        vis = cls.enforce_gaze_direction_continuity(vis)
        ff = cls.enforce_gaze_direction_continuity(ff)
        lf = cls.enforce_gaze_direction_continuity(lf)

        for pat in BANNED_EYE_LEVEL_PATTERNS:
            if pat.search(motion):
                motion = pat.sub("", motion).strip()
        motion = re.sub(r"^\s*[,;:\.\-–—]+\s*", "", motion).strip()
        motion = cls.sanitize_comic_panel_artifacts(motion)
        motion = cls.sanitize_motion_prompt_safety(motion)

        ff_idxs = list(dict.fromkeys(sd.ff_vis_char_idxs or []))
        lf_idxs = list(dict.fromkeys(sd.lf_vis_char_idxs or []))

        # Enforce spatial hand consistency across detailed keyframe descriptions (Kayden only)
        if is_kayden_battle:
            wall_on_right = bool(re.search(r"\b(?:wall|tường)\b.*\b(?:on the right|bên phải)\b", vis, re.IGNORECASE)) or "stone wall on the right" in vis.lower()
            if wall_on_right:
                ff = re.sub(r"\b(?:extended\s+)?left\s+hand\s+(?:pressed|braced|placed|on)\b.*?\b(?:wall|stone)\b", "bloodied right hand firmly braced against the stone wall on the right", ff, flags=re.IGNORECASE)
                lf = re.sub(r"\b(?:extended\s+)?left\s+hand\s+(?:pressed|braced|placed|on)\b.*?\b(?:wall|stone)\b", "bloodied right hand firmly braced against the stone wall on the right", lf, flags=re.IGNORECASE)
            ff = re.sub(r"\bone\s+hand\s+(?:pressed|braced|on)\b.*?\b(?:wall|stone)\b", "bloodied right hand firmly braced against the stone wall on the right", ff, flags=re.IGNORECASE)
            lf = re.sub(r"\b(?:extended\s+)?left\s+hand\s+(?:pressed|braced|placed|on)\b.*?\b(?:wall|stone)\b", "bloodied right hand firmly braced against the stone wall on the right", lf, flags=re.IGNORECASE)
            lf = re.sub(r"\bone\s+hand\s+(?:pressed|braced|on)\b.*?\b(?:wall|stone)\b", "bloodied right hand firmly braced against the stone wall on the right", lf, flags=re.IGNORECASE)
            motion = re.sub(r"\b(?:extended\s+)?left\s+hand\s+(?:pressed|braced|placed|on)\b.*?\b(?:wall|stone)\b", "right hand firmly braced against the stone wall on the right", motion, flags=re.IGNORECASE)
            motion = re.sub(r"\bone\s+hand\s+(?:pressed|braced|on)\b.*?\b(?:wall|stone)\b", "right hand firmly braced against the stone wall on the right", motion, flags=re.IGNORECASE)

        # Enforce lightning geometry in keyframe descriptions
        is_lightning = any(kw in vis.lower() for kw in ["lightning", "tia sét", "sét", "thunderbolt", "electric bolt"])
        if is_lightning:
            ff = re.sub(
                r"\b(?:thin|smooth|straight|constant)?\s*(?:diagonal\s+)?(?:cyan-white|cyan|blue)?\s*(?:energy\s+)?(?:streak|beam|line)\b",
                "violent, jagged cyan-blue electric lightning bolt with sharp branching tendrils and high-voltage crackle",
                ff,
                flags=re.IGNORECASE,
            )
            lf = re.sub(
                r"\b(?:thin|smooth|straight|constant)?\s*(?:diagonal\s+)?(?:cyan-white|cyan|blue)?\s*(?:energy\s+)?(?:streak|beam|line)\b",
                "violent, jagged cyan-blue electric lightning bolt with sharp branching tendrils and high-voltage crackle",
                lf,
                flags=re.IGNORECASE,
            )

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
                "variation_reason": variation_reason.strip(),
                "motion_desc": motion.strip(),
                "ff_desc": ff.strip(),
                "lf_desc": lf.strip(),
                "ff_vis_char_idxs": ff_idxs,
                "lf_vis_char_idxs": lf_idxs,
            }
        )
