from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from .character import CharacterInScene


class MangaDialogueLine(BaseModel):
    speaker: str = Field(
        default="Narrator",
        description="Canonical character name matching CharacterInScene or 'Narrator'.",
    )
    line: str = Field(
        default="",
        description="Spoken line or inner monologue kept concise (<= 12-15 words) for clean anime pacing.",
    )
    line_type: Literal["spoken", "thought", "narration"] = Field(
        default="spoken",
        description="Whether this line is spoken out loud ('spoken'), internal thought ('thought'), or caption/narration ('narration').",
    )
    emotion: str = Field(
        default="",
        description="Brief delivery emotion or tone, e.g. 'Tense', 'Shocked', 'Sarcastic'.",
    )
    vietsub: str = Field(
        default="",
        description="Natural Vietnamese translation faithful to the exact comic bubble.",
    )
    source_text: str = Field(
        default="",
        description="Verbatim original text inside the comic speech/thought bubble, before any rewrite or translation.",
    )


class MangaCharacterObservation(BaseModel):
    name: str = Field(
        ...,
        description=(
            "Character name including distinct physical form if the character transforms, "
            "e.g. 'Kayden (Human)' vs 'Kayden (Fat Orange Tabby Cat)'."
        ),
    )
    species_or_form: str = Field(
        default="human",
        description="Physical form/species, e.g. 'human', 'cat (four-legged quadruped)', 'dog (four-legged quadruped)'.",
    )
    appearance_summary: str = Field(
        default="",
        description="Concrete visible anatomy, hair color/style, eye color, fur stripes, scars, and posture seen directly on the page.",
    )
    outfit_summary: str = Field(
        default="",
        description="Signature outfit and exact colors seen directly on the page (or 'Unclothed tabby fur on all four paws' for animals).",
    )
    page_idx: int = Field(
        default=0,
        description="0-based chapter page index matching the [Page N] image header.",
    )
    crop_box_norm: List[int] = Field(
        default_factory=lambda: [0, 0, 1000, 1000],
        description="Normalized [ymin, xmin, ymax, xmax] bounding box in 0..1000 scale isolating this character cleanly.",
    )
    confidence: float = Field(
        default=1.0,
        description="Visual clarity/suitability score (0.0 to 1.0) as a clean character seed crop.",
    )


class MangaBeat(BaseModel):
    beat_id: str = Field(
        default="",
        description="Unique beat identifier assigned deterministically during extraction/planning.",
    )
    chapter: int = Field(default=1, description="1-based chapter number.")
    page_idx: int = Field(
        default=0,
        description="0-based chapter page index where this comic panel primarily appears.",
    )
    crop_box_norm: List[int] = Field(
        default_factory=lambda: [0, 0, 1000, 1000],
        description="Normalized [ymin, xmin, ymax, xmax] panel bounding box in 0..1000 scale on page_idx.",
    )
    setting: str = Field(
        default="",
        description="Location, lighting, weather, and atmosphere directly visible in the panel.",
    )
    action_desc: str = Field(
        default="",
        description="Concrete visual action, camera angle/framing, character poses, facing directions, and expressions.",
    )
    visual_details: str = Field(
        default="",
        description="Specific visual details directly visible: hair/fur/clothing colors, battle scuffs, lighting FX.",
    )
    characters_present: List[str] = Field(
        default_factory=list,
        description="Canonical character names visible in this beat.",
    )
    dialogues: List[MangaDialogueLine] = Field(
        default_factory=list,
        description="Dialogue, internal thought, and narration bubbles appearing in this beat.",
    )
    sfx: List[str] = Field(
        default_factory=list,
        description="Sound effects depicted on the panel (e.g. 'SPPPPPT!', 'CRACK!', 'MEOW').",
    )
    mood: str = Field(default="", description="Emotional tone / pacing mood of this beat.")
    is_scene_break: bool = Field(
        default=False,
        description="True when location, time of day, or distinct story scene changes cleanly at the start of this beat.",
    )
    anchor_image_path: str = Field(
        default="",
        description="Resolved file path to the cropped panel anchor image for this beat.",
    )


class MangaChunkExtraction(BaseModel):
    chapter: int = Field(default=1)
    start_page_idx: int = Field(default=0)
    end_page_idx: int = Field(default=0)
    characters_observed: List[MangaCharacterObservation] = Field(default_factory=list)
    beats: List[MangaBeat] = Field(default_factory=list)
    chunk_tail_summary: str = Field(
        default="",
        description="Short continuity note summarizing any unresolved motion or split speech bubble at the bottom of the last page.",
    )


class ShotPanelAnchor(BaseModel):
    shot_idx: int
    beat_ids: List[str] = Field(default_factory=list)
    panel_image_paths: List[str] = Field(default_factory=list)
    description: str = Field(
        default=(
            "COMIC PANEL COMPOSITION & POSE REFERENCE "
            "(redraw faithfully into borderless textless 16:9 anime film style, "
            "REMOVE ALL speech bubbles, SFX lettering, and comic borders)"
        )
    )


class MangaEpisodeScene(BaseModel):
    scene_idx: int
    chapter: int = Field(default=1)
    title: str = Field(default="")
    setting: str = Field(default="")
    beat_ids: List[str] = Field(default_factory=list)
    script_text: str = Field(default="")
    panel_anchors_by_beat: Dict[str, str] = Field(default_factory=dict)


class MangaEpisodePlan(BaseModel):
    episode_title: str = Field(default="")
    style: str = Field(default="")
    chapters: List[int] = Field(default_factory=list)
    characters: List[CharacterInScene] = Field(default_factory=list)
    character_seed_refs: Dict[str, List[str]] = Field(default_factory=dict)
    scenes: List[MangaEpisodeScene] = Field(default_factory=list)
    total_beats: int = Field(default=0)


class MangaSeriesEpisodeEntry(BaseModel):
    episode_idx: int = Field(..., description="1-based episode index within the series.")
    episode_title: str = Field(default="")
    chapters: List[int] = Field(default_factory=list)
    working_dir: str = Field(default="")
    scene_count: int = Field(default=0)
    total_beats: int = Field(default=0)
    character_names: List[str] = Field(default_factory=list)
    status: Literal["planned", "portraits", "frames", "rendered", "error"] = Field(default="planned")
    final_video_path: Optional[str] = Field(default=None)
    subtitle_paths: List[str] = Field(default_factory=list)
    error: Optional[str] = Field(default=None)


class MangaSeriesPlan(BaseModel):
    series_title: str = Field(default="")
    style: str = Field(default="")
    total_chapters: int = Field(default=0)
    chapters_per_episode: int = Field(default=1)
    episodes: List[MangaSeriesEpisodeEntry] = Field(default_factory=list)
    series_characters_path: str = Field(default="")
    series_portraits_registry_path: str = Field(default="")
    completed_episodes: int = Field(default=0)
