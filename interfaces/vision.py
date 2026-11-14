from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field

ShotCategory = Literal["canon", "inferred", "creative"]
"""Provenance label for every planned shot.

- ``canon``: the shot directly re-presents a specific manga panel composition.
- ``inferred``: the shot reinterprets an event that the manga establishes
  (different camera angle on the same canonical moment).
- ``creative``: additional cinematic coverage (establishing shot, reaction
  shot, insert detail) that adds no new plot content.
"""


class CharacterBibleEntry(BaseModel):
    """Persistent character identity built from cross-panel observations.

    Keyed by the canonical ``identifier_in_scene`` used by the existing
    roster/alias system so all downstream agents share the same IDs.
    """

    character_key: str = Field(
        ...,
        description="Canonical character name matching CharacterInScene.identifier_in_scene (form-scoped, e.g. 'Kayden (Human)').",
    )
    aliases: List[str] = Field(
        default_factory=list,
        description="Raw names observed across panels that map to this identity.",
    )
    form_category: Literal["human", "feline", "canine", "animal", "other"] = Field(
        default="human",
        description="Physical form category separating distinct forms of the same base character.",
    )
    permanent_features: str = Field(
        default="",
        description="Identity features stable across panels: hair/fur, eyes, body proportions, scars, species.",
    )
    temporary_features: str = Field(
        default="",
        description="Outfit / accessory / injury state observed in this chapter (may change between scenes).",
    )
    reference_image_paths: List[str] = Field(
        default_factory=list,
        description="Seed crops or user-provided reference images usable as identity anchors.",
    )
    source_beat_ids: List[str] = Field(
        default_factory=list,
        description="Beat IDs where this character appears (evidence trail).",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Identity-merge confidence: how consistently observations were matched.",
    )


class EnvironmentBibleEntry(BaseModel):
    """Reconstructed location shared by multiple panels / beats."""

    environment_id: str = Field(
        ...,
        description="Stable ID such as 'env_001' shared across shots and scene graphs.",
    )
    name: str = Field(
        default="",
        description="Short human-readable location label, e.g. 'Rainy back alley at night'.",
    )
    env_type: str = Field(
        default="",
        description="Location archetype, e.g. 'urban alley', 'classroom', 'rooftop'.",
    )
    indoor: bool = Field(default=False, description="True when the location is interior.")
    lighting: str = Field(default="", description="Lighting conditions visible in the panels.")
    time_of_day: str = Field(
        default="unknown",
        description="Time of day when supported by visual evidence; 'unknown' otherwise.",
    )
    weather: str = Field(
        default="unknown",
        description="Weather when visually supported; 'unknown' otherwise.",
    )
    key_props: List[str] = Field(
        default_factory=list,
        description="Recurring environmental objects worth insert shots (sign, dumpster, window...).",
    )
    foreground_desc: str = Field(default="", description="Typical near-camera layer content.")
    midground_desc: str = Field(default="", description="Typical mid-layer content.")
    background_desc: str = Field(default="", description="Typical far-layer content.")
    source_beat_ids: List[str] = Field(
        default_factory=list,
        description="Beats that witness this environment (evidence trail).",
    )
    reference_image_paths: List[str] = Field(
        default_factory=list,
        description="Panel anchor crops that best represent the environment.",
    )


class SpatialEntity(BaseModel):
    entity_id: str = Field(
        ...,
        description="Entity key within one beat's scene graph: a character key, 'camera', or prop label.",
    )
    entity_type: Literal["character", "object", "environment"] = Field(default="character")
    label: str = Field(default="", description="Human-readable label, e.g. 'Kayden (Human)' or 'dumpster'.")


class SpatialRelation(BaseModel):
    subject: str = Field(..., description="Entity id of the relation subject.")
    relation: str = Field(
        ...,
        description="Spatial verb, e.g. 'left_of', 'right_of', 'behind', 'in_front_of', 'holding', 'facing', 'above', 'below'.",
    )
    obj: str = Field(..., description="Entity id of the relation object.")
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)


class BeatSceneGraph(BaseModel):
    """Inferred 2D spatial layout of one manga beat.

    Manga panels are drawings, not measurements: positions are stored as
    qualitative relations with confidence, never as 3D coordinates.
    """

    beat_id: str = Field(..., description="Beat this graph describes.")
    environment_id: str = Field(
        default="",
        description="EnvironmentBibleEntry.environment_id this beat belongs to.",
    )
    entities: List[SpatialEntity] = Field(default_factory=list)
    relations: List[SpatialRelation] = Field(default_factory=list)
    camera_position: str = Field(
        default="",
        description="Author's implied camera position, e.g. 'behind Kayden (Human), low angle'.",
    )
    source_page_idx: int = Field(default=0)


class ChapterEvent(BaseModel):
    event_id: str = Field(..., description="Stable event id such as 'ev_001'.")
    description: str = Field(..., description="What happens, grounded in the panels.")
    event_type: Literal["action", "dialogue", "reaction", "establishing", "transition", "other"] = Field(
        default="action",
    )
    beat_ids: List[str] = Field(default_factory=list, description="Supporting beats (evidence trail).")
    character_keys: List[str] = Field(default_factory=list)
    environment_id: str = Field(default="")
    order: int = Field(default=0, description="Chronological position within the chapter.")
    emotional_tone: str = Field(default="")


class IdentityDedupPair(BaseModel):
    """A pair of character bible entries that may refer to the same person."""

    a: str = Field(..., description="First character_key.")
    b: str = Field(..., description="Second character_key.")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Embedding cosine similarity of permanent features.")
    suspected: bool = Field(..., description="True when similarity crosses the dedup threshold.")
    never_cooccur: bool = Field(
        default=False,
        description="No shared source beats: the two names never appear together in any panel.",
    )


class IdentityDedupReport(BaseModel):
    """Report-only duplicate-identity analysis over the character bible."""

    status: Literal["ok", "skipped"] = Field(default="skipped")
    embedder: str = Field(default="", description="Which embedder ran ('mock', provider id, ...).")
    threshold: float = Field(default=0.82)
    reason: str = Field(default="", description="Why the analysis was skipped, when applicable.")
    pairs: List[IdentityDedupPair] = Field(default_factory=list, description="Sorted by similarity desc; capped.")

    def suspected_pairs(self) -> List[IdentityDedupPair]:
        return [p for p in self.pairs if p.suspected]


class VisionWorldModel(BaseModel):
    """Chapter-level structured understanding produced by the Vision Layer."""

    chapters: List[int] = Field(default_factory=list)
    characters: List[CharacterBibleEntry] = Field(default_factory=list)
    environments: List[EnvironmentBibleEntry] = Field(default_factory=list)
    beat_scene_graphs: List[BeatSceneGraph] = Field(default_factory=list)
    events: List[ChapterEvent] = Field(default_factory=list)
    identity_dedup: Optional[IdentityDedupReport] = Field(
        default=None,
        description="Report-only suspected duplicate identities (e.g. same child described twice).",
    )
    notes: str = Field(
        default="",
        description="Free-form continuity notes carried from the reading pass.",
    )

    def character_by_key(self, key: str) -> Optional[CharacterBibleEntry]:
        for entry in self.characters:
            if entry.character_key == key:
                return entry
        return None

    def environment_by_id(self, env_id: str) -> Optional[EnvironmentBibleEntry]:
        for entry in self.environments:
            if entry.environment_id == env_id:
                return entry
        return None


class ExpansionShotDirective(BaseModel):
    """One additional shot the Cinematic Director wants for a beat."""

    shot_type: Literal[
        "establishing_wide",
        "aerial_reveal",
        "over_the_shoulder",
        "reaction_close_up",
        "insert_detail",
        "environment_b_roll",
        "alternate_angle",
        "return_to_panel",
    ] = Field(
        default="alternate_angle",
        description="Cinematic purpose of the additional coverage.",
    )
    category: ShotCategory = Field(
        default="creative",
        description="'inferred' when re-framing a canonical moment; 'creative' for added coverage.",
    )
    purpose: str = Field(default="", description="Why this shot improves the adaptation.")
    visual_hint: str = Field(
        default="",
        description="Concrete visual guidance grounded in the world model (characters, props, lighting).",
    )
    camera_angle: str = Field(default="", description="e.g. 'high_angle', 'low_angle_worms_eye', 'eye_level'.")
    camera_movement: str = Field(default="", description="e.g. 'slow_dolly_in', 'static', 'aerial_drift'.")
    framing: str = Field(default="", description="e.g. 'wide', 'medium', 'close_up', 'extreme_close_up'.")
    target_duration_sec: Optional[float] = Field(
        default=None,
        description="Optional pacing hint (1.2-7.0s) aligned with the storyboard artist's duration rules.",
    )
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class BeatExpansionDirective(BaseModel):
    beat_id: str = Field(..., description="Beat to expand.")
    preserve_panel_composition: bool = Field(
        default=True,
        description="True when the original manga panel must still be represented as its own shot.",
    )
    additional_shots: List[ExpansionShotDirective] = Field(default_factory=list)


class SceneExpansionPlan(BaseModel):
    """Per-scene cinematic expansion plan feeding the existing StoryboardArtist."""

    scene_idx: int = Field(default=0)
    beat_directives: List[BeatExpansionDirective] = Field(default_factory=list)
    world_context_block: str = Field(
        default="",
        description="Compact world-model text block injected into the storyboard prompt.",
    )
    notes: str = Field(default="")


class ShotProvenance(BaseModel):
    """Traceability record attached to every storyboard shot."""

    shot_idx: int = Field(...)
    scene_idx: int = Field(default=0)
    source_beat_ids: List[str] = Field(default_factory=list)
    category: ShotCategory = Field(default="creative")
    purpose: str = Field(default="")
    supporting_environment_id: str = Field(default="")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class VisualQAIssue(BaseModel):
    shot_idx: int = Field(...)
    aspect: Literal[
        "character_identity",
        "outfit_consistency",
        "character_count",
        "environment_consistency",
        "spatial_relation",
        "action_continuity",
        "camera_continuity",
        "narrative_alignment",
        "artifact",
        "missing_frame",
        "identity_dedup",
        "vlm_review",
    ] = Field(default="artifact")
    severity: Literal["info", "warning", "critical"] = Field(default="warning")
    description: str = Field(default="")
    suggestion: str = Field(default="")


class VisualQAReport(BaseModel):
    """Scene-level visual QA result comparing generated keyframes against the
    source panel anchors and the world model."""

    scene_idx: int = Field(default=0)
    reviewed_frames: int = Field(default=0)
    passed_shots: List[int] = Field(default_factory=list)
    failed_shots: List[int] = Field(default_factory=list)
    issues: List[VisualQAIssue] = Field(default_factory=list)
    summary: str = Field(default="")

    def shots_with_critical_issues(self) -> List[int]:
        return sorted({i.shot_idx for i in self.issues if i.severity == "critical"})
