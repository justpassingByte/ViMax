from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal, Tuple

_VALID_TRANSITIONS = {"cut", "crossfade", "fade_black", "white_flash", "whip_pan"}


def _normalize_transition_in(value: object) -> str:
    """Normalize transition_in variants to supported canonical literals."""
    if value is None:
        return "cut"
    raw = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if raw in _VALID_TRANSITIONS:
        return raw
    alias_map = {
        "dissolve": "crossfade",
        "fade": "crossfade",
        "fade_in": "fade_black",
        "fade_to_black": "fade_black",
        "black_fade": "fade_black",
        "flash": "white_flash",
        "flash_white": "white_flash",
        "swish_pan": "whip_pan",
        "snap_pan": "whip_pan",
        "hard_cut": "cut",
    }
    return alias_map.get(raw, "cut")


class ShotBriefDescription(BaseModel):
    idx: int = Field(
        description="The index of the shot in the sequence, starting from 0.",
        examples=[0, 1, 2],
    )
    is_last: bool = Field(
        description="Whether this is the last shot. If True, the story of the script has ended and no more shots will be planned after this one.",
        examples=[False, True],
    )

    # visual
    cam_idx: int = Field(
        description="The index of the camera in the scene.",
        examples=[0, 1, 2],
    )
    visual_desc: str = Field(
        description='''A vivid and detailed visual description of the shot that convey rich visual information through text. The character identifiers in the description must match those in the character list and be enclosed in angle brackets (e.g., <Alice>, <Bob>). All visible characters should be described.
        If there is a conversation, summarize or quote the line using single quotes '...' (avoid raw nested double quotes inside JSON strings, e.g., <SLING> says: 'Gear retracted. Flaps transitioning.').
        ''',
        examples=[
            "An over-the-shoulder shot at eye level, positioned behind <Alice>. The foreground, including <Alice>'s shoulder and head, is softly blurred, directing focus onto <Bob>'s face. <Bob>'s subtle reactions—shifting from surprise to delight—are clearly visible. The supermarket background is gently blurred with cool fluorescent lighting.",
        ]
    )


    # audio
    audio_desc: str = Field(
        description="A detailed description of the audio in the shot. If the script includes [Speaker]/[Thought] lines or (Vietsub: ...), preserve them verbatim using single quotes if needed.",
        examples=[
            "[Sound Effect] Ambient sound (supermarket background noise, shopping cart wheels rolling)",
            "[Speaker] Alice (Happy): 'Hello, how are you?' (Vietsub: Chào bạn, bạn khỏe không?)",
            None,
        ],
    )
    transition_in: Literal["cut", "crossfade", "fade_black", "white_flash", "whip_pan"] = Field(
        default="cut",
        description="Editorial transition from the previous shot into this shot: 'cut' for continuous action or sharp impacts, 'crossfade' for gentle emotional/flashback/closing dissolves, 'fade_black' for clear time skips or location shifts, 'white_flash' for power awakenings, 'whip_pan' for dynamic swish pan angle shifts.",
        examples=["cut", "fade_black", "crossfade", "white_flash", "whip_pan"],
    )

    @field_validator("transition_in", mode="before")
    @classmethod
    def _validate_transition_in(cls, v: object) -> str:
        return _normalize_transition_in(v)
    continuity_from_prev: bool = Field(
        default=False,
        description="Set True when this shot immediately continues the spatial momentum, battle condition, lighting, and screen direction from the previous shot's ending frame.",
        examples=[False, True],
    )
    target_duration_sec: Optional[float] = Field(
        default=None,
        description="Optional editorial duration (in seconds) for post-render pacing of silent anime action beats (e.g., 1.2-3.0s). Leave None for spoken dialogue shots to preserve full audio duration.",
        examples=[None, 4.0, 3.5],
    )
    # Rule-based shot planner fields (phase 08). Optional; old data reads fine.
    beat_ids: List[str] = Field(default_factory=list)
    beat_type: str = ""
    shot_size: str = ""
    camera_angle: str = ""
    camera_movement: str = ""
    fx: List[str] = Field(default_factory=list)
    characters: List[str] = Field(default_factory=list)

    def __str__(self):
        s = f"Shot {self.idx}:\n"
        s += f"Camera Index: {self.cam_idx}\n"
        s += f"Visual: {self.visual_desc}\n"
        if self.audio_desc:
            s += f"Audio: {self.audio_desc}"
        return s


class ShotDescription(BaseModel):
    idx: int = Field(
        description="The index of the shot in the sequence, starting from 0."
    )
    is_last: bool = Field(
        description="Whether this is the last shot in the sequence. If True, no more shots will be planned after this one."
    )

    # visual
    cam_idx: int = Field(
        description="The index of the camera in the scene.",
        examples=[0, 1, 2],
    )
    visual_desc: str = Field(
        description='''A vivid and detailed visual description of the shot that convey rich visual information through text. The character identifiers in the description must match those in the character list and be enclosed in angle brackets (e.g., <Alice>, <Bob>).
        If there is dialogue, quote it using single quotes '...' rather than raw double quotes inside JSON strings (e.g., <SLING> says: 'Gear retracted.'). ''',
        examples=[
            "An over-the-shoulder shot at eye level, positioned behind <Alice>. The foreground, including <Alice>'s shoulder and head, is softly blurred, directing focus onto <Bob>'s face. <Bob>'s subtle reactions—shifting from surprise to delight—are clearly visible. The supermarket background is gently blurred with cool fluorescent lighting.",
        ]
    )
    variation_type: Literal["large", "medium", "small"] = Field(
        description="Indicates the degree of change in the shot's content.",
        examples=["large", "medium", "small"],
    )
    variation_reason: str = Field(
        description="The reason for the variation type of the shot.",
        examples=[
            "This is a transition shot where the content of the first frame and the last frame differs dramatically. So the variation type is large.",
            "Compared to the first frame, a new character appears in the last frame, and there are no significant changes in the composition. So the variation type is medium.",
            "Compared to the first frame, there are only minor changes in the composition. So the variation type is small.",
            "This shot only shows Alice speaking and the changes in her facial expressions, thus the variation type is small.",
        ],
    )

    ff_desc: str = Field(
        description="The first frame of the shot.",
        examples=[
            "Medium shot of a supermarket aisle at eye level. Bob(a tall man wearing a blue shirt and jeans) is positioned on the right side of the frame, captured in profile and facing right, while Alice(a young woman with short hair, wearing a green dress) is on the left, shown pushing a shopping cart with her gaze lowered toward the ground. They are arranged in a front-to-back spatial relationship. Shelves line both sides of the frame, and cool-toned fluorescent lighting from above washes over the scene. The vibrant colors of product packaging contrast with the metallic gray of the shopping cart, all contained within a stable, horizontally balanced composition.",
            "Extreme long shot. Aerial view from hundreds of meters above the ground. The boundless golden desert resembles undulating frozen waves, occupying the vast majority of the frame. At the very center of the image, a tiny, solitary explorer appears only as a faint dark speck, dragging a long, lonely trail of footprints behind him, stretching all the way to the edge of the frame.",
            "Medium shot at eye level angle. Designer A(with a beard, wearing a white suit) leans forward passionately, speaking emphatically. Product Manager B(with a beard, wearing a white T-shirt) sits with crossed arms, looking skeptical. Between them, Development Engineer C(brown hair, wearing a blue T-shirt) appears anxious, glancing between the two. Project Manager D(curly hair, wearing a red T-shirt) prepares to mediate, focusing on a whiteboard. Bright overhead lighting highlights their expressions, with a blurred whiteboard and glass wall in the background.",
            "A low-angle close-up shot captures the figure from below, framing him from the chest up. His face appears resolute and commanding, his eyes piercing as he speaks passionately. Flecks of saliva are visible, emphasizing his intensity. The overcast sky breaks with occasional light, casting him as a heroic, almost monumental figure against the gloom.",
            "An extremely close-up of an old, motionless pocket watch. Soft light highlights scratches on its brass case and the enamel dial with Roman numerals. The second hand remains fixed at 'VIII', casting a sharp shadow. A wrinkled finger gently touches the glass surface, evoking a tangible sense of stillness and time.",
            "An over-the-shoulder shot at eye level, positioned behind Character A(red hair, wearing a white T-shirt). The foreground, including A's shoulder and head, is softly blurred, directing focus onto Character B(with a beard, wearing a white T-shirt)'s face. B's subtle reactions—shifting from surprise to confusion, then to a glimmer of understanding—are clearly visible. The café background is gently blurred with warm lighting.",
        ]
    )
    ff_vis_char_idxs: List[int] = Field(
        default=[],
        description="The indices of the characters in the first frame.",
        examples=[
            [0, 1],
            [0],
            [],
        ],
    )
    lf_desc: str = Field(
        description="The last frame of the shot.",
    )
    lf_vis_char_idxs: List[int] = Field(
        default=[],
        description="The indices of the characters in the last frame.",
    )
    motion_desc: str = Field(
        description='''The motion description of the shot.
        If there is dialogue or narration, use single quotes '...' rather than raw double quotes inside JSON strings (e.g., SLING says: 'Gear retracted.' or Narration: 'Everything looks quiet.'). ''',
    )

    # audio
    audio_desc: str = Field(
        description="A detailed description of the audio in the shot, preserving any [Speaker]/[Thought] and (Vietsub: ...) lines.",
        examples=[
            "[Sound Effect] Ambient sound (supermarket background noise, shopping cart wheels rolling)",
            "[Speaker] Alice (Happy): 'Hello, how are you?' (Vietsub: Chào bạn, bạn khỏe không?)",
            None,
        ],
    )
    transition_in: Literal["cut", "crossfade", "fade_black", "white_flash", "whip_pan"] = Field(
        default="cut",
        description="Editorial transition from the previous shot into this shot: 'cut', 'crossfade', 'fade_black', 'white_flash', or 'whip_pan'.",
        examples=["cut", "fade_black", "crossfade", "white_flash", "whip_pan"],
    )

    @field_validator("transition_in", mode="before")
    @classmethod
    def _validate_transition_in(cls, v: object) -> str:
        return _normalize_transition_in(v)
    continuity_from_prev: bool = Field(
        default=False,
        description="Whether this shot directly anchors its opening visual state to the previous shot's ending frame.",
        examples=[False, True],
    )
    target_duration_sec: Optional[float] = Field(
        default=None,
        description="Optional post-render target duration (in seconds) for non-dialogue action pacing while keeping both opening and ending keyframes intact.",
        examples=[None, 4.0, 3.5],
    )
