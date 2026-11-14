"""Character profile schema (phase 05).

A profile gathers everything known about one character across chapters: forms,
permanent features, outfits (valid over beat ranges), transient states, source
crops, per-outfit embeddings, and a voice profile (filled in phase 07). It is
the single source of truth the reference-selection step reads for the right
outfit/state at a given beat.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class OutfitSheet(BaseModel):
    front: str = ""
    side: str = ""
    back: str = ""


class Outfit(BaseModel):
    outfit_id: str
    label: str = ""
    description: str = ""
    valid_from_beat: str = ""          # e.g. "ch01_b012"
    valid_to_beat: Optional[str] = None
    sheet: OutfitSheet = Field(default_factory=OutfitSheet)


class StateEntry(BaseModel):
    kind: Literal["injury", "item", "hair", "form", "other"] = "other"
    text: str = ""
    from_beat: str = ""
    to_beat: Optional[str] = None
    evidence_beat_ids: List[str] = Field(default_factory=list)


class SourceCrop(BaseModel):
    path: str
    beat_id: str = ""
    form_id: str = ""
    outfit_id: str = ""
    usable: bool = True


class VoiceProfile(BaseModel):
    mode: str = "native"               # native | native_fixed | native_with_reference
    description: str = ""
    reference_audio: str = ""
    engine: str = ""
    voice_id: str = ""
    language: str = "ja"


class CharacterProfile(BaseModel):
    character_id: str
    display_name: str = ""
    aliases: List[str] = Field(default_factory=list)
    forms: List[dict] = Field(default_factory=list)       # {form_id, label, species}
    permanent_features: str = ""
    outfits: List[Outfit] = Field(default_factory=list)
    states: List[StateEntry] = Field(default_factory=list)
    source_crops: List[SourceCrop] = Field(default_factory=list)
    embedding_paths: Dict[str, str] = Field(default_factory=dict)  # "<form_id>/<outfit_id>" -> .npy
    voice: VoiceProfile = Field(default_factory=VoiceProfile)
    prompt_notes: str = ""
    cooccur_beat_ids: List[str] = Field(default_factory=list)  # beats this character shared with others
    status: Literal["draft", "approved", "locked"] = "draft"
    version: int = 1
