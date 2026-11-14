"""Chapter-level World Model builder for the Manga Vision Layer.

Consumes the per-chunk ``MangaChunkExtraction`` output of ``MangaReader`` and
builds a ``VisionWorldModel``:

- ``CharacterBibleEntry`` list (deterministic merge of the existing character
  roster + cross-panel observations; no extra LLM cost).
- ``EnvironmentBibleEntry`` list (VLM analysis pass, deduplicated with a
  deterministic token-similarity merge so repeated locations share one ID).
- ``BeatSceneGraph`` per beat (qualitative spatial relations with confidence —
  manga panels are drawings, so nothing claims 3D precision).
- ``ChapterEvent`` timeline (grounded in beat IDs).

All VLM calls are batched, text-first (compact beat summaries, not full-page
images) and cached on disk for reproducibility and resumability.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from interfaces import (
    CharacterBibleEntry,
    ChapterEvent,
    CharacterInScene,
    BeatSceneGraph,
    EnvironmentBibleEntry,
    MangaBeat,
    MangaChunkExtraction,
    SpatialEntity,
    SpatialRelation,
    VisionWorldModel,
)
from utils.retry import after_func
from utils.robust_json_parser import TrailingCommaTolerantPydanticOutputParser as PydanticOutputParser

_BEATS_PER_ANALYSIS_BATCH = 10
_ENV_MERGE_TOKEN_JACCARD = 0.55

_ALLOWED_RELATIONS = {
    "left_of",
    "right_of",
    "above",
    "below",
    "behind",
    "in_front_of",
    "facing",
    "holding",
    "touching",
    "standing_on",
    "sitting_on",
    "near",
    "inside",
    "outside",
    "between",
}

_ALLOWED_EVENT_TYPES = {"action", "dialogue", "reaction", "establishing", "transition", "other"}

_system_prompt_template_analyze = """
[Role]
You are a manga continuity archivist building a structured WORLD MODEL from an
already-extracted chapter beat list.

[Task]
Given a sequential list of canon beats (setting, action, characters, dialogue,
SFX) extracted from manga panels, produce STRICTLY grounded analysis:

1. `environments`: distinct physical LOCATIONS. Multiple beats set at the same
   location must share ONE environment entry (same `key`). Record only what
   the beats literally state or visually imply. `time_of_day` / `weather` stay
   "unknown" when unsupported.
2. `beat_scene_graphs`: for EVERY provided beat, a qualitative spatial layout:
   entities (characters from the roster, notable objects, environment itself),
   pairwise spatial relations, and the implied camera position. Relations MUST
   use the allowed relation vocabulary. Manga panels are drawings: express
   positions qualitatively with confidence values, NEVER as measurements.
3. `events`: chronological chapter-level events. Each event lists the beat_ids
   that evidence it. Do NOT invent plot content not present in the beats.

[Hard Rules]
- Reference characters ONLY by the canonical names given in the input roster
  (including form suffixes like "(Human)"/"(Cat)").
- `beat_id` values must be copied verbatim from the input.
- Never invent new characters, names, or locations beyond what beats state.
- Relations confidence: 0.9+ when explicitly drawn, 0.5-0.8 when inferred.

[Output Format]
Return ONLY valid JSON matching:
{format_instructions}
"""

_human_prompt_template_analyze = """
CHARACTER ROSTER (canonical names, use verbatim):
{characters_str}

BEATS ({beat_count} beats, in reading order):
{beats_str}

Produce the world model JSON for ONLY these beats:
"""


class _RawEnvironmentCandidate(BaseModel):
    key: str = Field(default="", description="Stable short key chosen by the model for this location.")
    name: str = Field(default="")
    env_type: str = Field(default="")
    indoor: bool = Field(default=False)
    lighting: str = Field(default="")
    time_of_day: str = Field(default="unknown")
    weather: str = Field(default="unknown")
    key_props: List[str] = Field(default_factory=list)
    foreground_desc: str = Field(default="")
    midground_desc: str = Field(default="")
    background_desc: str = Field(default="")
    beat_ids: List[str] = Field(default_factory=list)


class _RawEntity(BaseModel):
    entity_id: str = Field(default="")
    entity_type: str = Field(default="character")
    label: str = Field(default="")


class _RawRelation(BaseModel):
    subject: str = Field(default="")
    relation: str = Field(default="near")
    obj: str = Field(default="")
    confidence: float = Field(default=0.6)


class _RawSceneGraph(BaseModel):
    beat_id: str = Field(default="")
    environment_key: str = Field(default="")
    entities: List[_RawEntity] = Field(default_factory=list)
    relations: List[_RawRelation] = Field(default_factory=list)
    camera_position: str = Field(default="")


class _RawEvent(BaseModel):
    description: str = Field(default="")
    event_type: str = Field(default="action")
    beat_ids: List[str] = Field(default_factory=list)
    character_keys: List[str] = Field(default_factory=list)
    environment_key: str = Field(default="")
    emotional_tone: str = Field(default="")


class _RawWorldAnalysis(BaseModel):
    environments: List[_RawEnvironmentCandidate] = Field(default_factory=list)
    beat_scene_graphs: List[_RawSceneGraph] = Field(default_factory=list)
    events: List[_RawEvent] = Field(default_factory=list)


def _token_set(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9]+", str(text or "").lower()) if len(tok) >= 3}


def _token_jaccard(a: str, b: str) -> float:
    ta, tb = _token_set(a), _token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _safe_relation(rel: str) -> str:
    cleaned = re.sub(r"[^a-z_]", "", str(rel or "").strip().lower().replace(" ", "_"))
    if cleaned in _ALLOWED_RELATIONS:
        return cleaned
    # Common synonyms -> allowed vocabulary (best effort, deterministic)
    synonyms = {
        "on": "standing_on",
        "in": "inside",
        "next_to": "near",
        "beside": "near",
        "under": "below",
        "over": "above",
        "in_front": "in_front_of",
        "front": "in_front_of",
        "back": "behind",
        "left": "left_of",
        "right": "right_of",
    }
    return synonyms.get(cleaned, "near")


def _env_similarity(candidate: _RawEnvironmentCandidate, entry: EnvironmentBibleEntry) -> float:
    parts = (
        _token_jaccard(candidate.name, entry.name) * 2.0
        + _token_jaccard(candidate.name, entry.env_type)
        + _token_jaccard(candidate.env_type, entry.env_type)
    )
    overlap_beats = len(set(candidate.beat_ids) & set(entry.source_beat_ids))
    return parts + (1.0 if overlap_beats else 0.0)


class WorldModelBuilder:
    """Builds a cached, chapter-level ``VisionWorldModel`` from reading chunks."""

    def __init__(self, chat_model: Runnable) -> None:
        self.chat_model = chat_model

    # ------------------------------------------------------------------
    # Character bible (deterministic, no VLM cost)
    # ------------------------------------------------------------------
    def build_character_bible(
        self,
        *,
        chunks: Sequence[MangaChunkExtraction],
        characters: Sequence[CharacterInScene],
        character_seed_refs: Optional[Dict[str, List[str]]] = None,
    ) -> List[CharacterBibleEntry]:
        roster_by_name = {c.identifier_in_scene: c for c in characters}
        alias_to_canonical: Dict[str, str] = {}
        for name in roster_by_name:
            alias_to_canonical[name.lower()] = name

        grouped: Dict[str, Dict[str, Any]] = {}
        for chunk in chunks:
            for obs in chunk.characters_observed:
                raw = " ".join((obs.name or "").split()).strip()
                if not raw:
                    continue
                canonical = alias_to_canonical.get(raw.lower()) or raw
                entry = grouped.setdefault(
                    canonical,
                    {"aliases": set(), "permanent": [], "temporary": [], "beats": set(), "pages": set()},
                )
                if raw != canonical:
                    entry["aliases"].add(raw)
                if obs.appearance_summary:
                    entry["permanent"].append(obs.appearance_summary.strip())
                if obs.outfit_summary:
                    entry["temporary"].append(obs.outfit_summary.strip())
                entry["pages"].add(obs.page_idx)

        # Spoken-only speakers (e.g. Narrator) may not have observations.
        for name, char in roster_by_name.items():
            grouped.setdefault(
                name,
                {
                    "aliases": set(),
                    "permanent": [char.static_features] if char.static_features else [],
                    "temporary": [char.dynamic_features] if char.dynamic_features else [],
                    "beats": set(),
                    "pages": set(),
                },
            )

        bible: List[CharacterBibleEntry] = []
        for name, info in grouped.items():
            roster = roster_by_name.get(name)
            obs_count = len(info["permanent"]) + len(info["temporary"])
            confidence = min(1.0, 0.35 + 0.2 * obs_count) if obs_count else 0.35
            bible.append(
                CharacterBibleEntry(
                    character_key=name,
                    aliases=sorted(info["aliases"]),
                    permanent_features="; ".join(dict.fromkeys(info["permanent"]))[:500],
                    temporary_features="; ".join(dict.fromkeys(info["temporary"]))[:350],
                    reference_image_paths=list((character_seed_refs or {}).get(name) or []),
                    source_beat_ids=sorted(info["beats"]),
                    confidence=confidence,
                )
            )
        bible.sort(key=lambda e: e.character_key.lower())
        return bible

    # ------------------------------------------------------------------
    # VLM analysis passes (batched + cached)
    # ------------------------------------------------------------------
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=6),
        after=after_func,
        reraise=True,
    )
    async def _invoke_analysis(self, *, characters: Sequence[CharacterInScene], beats: Sequence[MangaBeat]) -> _RawWorldAnalysis:
        parser = PydanticOutputParser(pydantic_object=_RawWorldAnalysis)
        characters_str = "\n".join(
            f"- {c.identifier_in_scene} (visible={c.is_visible}; static={c.static_features})"
            for c in characters
        )
        beats_str = self._format_beats(beats)
        messages = [
            SystemMessage(
                content=_system_prompt_template_analyze.format(format_instructions=parser.get_format_instructions())
            ),
            HumanMessage(
                content=_human_prompt_template_analyze.format(
                    characters_str=characters_str,
                    beat_count=len(beats),
                    beats_str=beats_str,
                )
            ),
        ]
        chain = self.chat_model | parser
        raw: _RawWorldAnalysis = await chain.ainvoke(messages)
        return raw

    @staticmethod
    def _format_beats(beats: Sequence[MangaBeat]) -> str:
        lines: List[str] = []
        for beat in beats:
            dlg = " | ".join(f"{d.speaker}: '{d.line}' ({d.line_type})" for d in beat.dialogues)
            lines.append(
                f"[{beat.beat_id}] p{beat.page_idx} scene_break={beat.is_scene_break} "
                f"setting='{beat.setting}' chars={list(beat.characters_present)} "
                f"action='{beat.action_desc}' details='{beat.visual_details}' sfx={list(beat.sfx)} "
                f"dialogues=[{dlg}]"
            )
        return "\n".join(lines)

    async def build_world_model(
        self,
        *,
        chunks: Sequence[MangaChunkExtraction],
        characters: Sequence[CharacterInScene],
        character_seed_refs: Optional[Dict[str, List[str]]] = None,
        working_dir: str | Path,
        progress=None,
        identity_dedup_embedder: Optional[Any] = None,
        identity_dedup_threshold: float = 0.82,
    ) -> VisionWorldModel:
        """Assemble the chapter-level world model (VLM passes cached on disk)."""
        out_dir = Path(working_dir) / "vision_world"
        out_dir.mkdir(parents=True, exist_ok=True)

        all_beats: List[MangaBeat] = []
        chapters: List[int] = []
        for chunk in chunks:
            chapters.append(chunk.chapter)
            all_beats.extend(chunk.beats)

        analysis: _RawWorldAnalysis = _RawWorldAnalysis()
        batches = [all_beats[i : i + _BEATS_PER_ANALYSIS_BATCH] for i in range(0, len(all_beats), _BEATS_PER_ANALYSIS_BATCH)]
        for batch_idx, batch in enumerate(batches):
            if not batch:
                continue
            cache_path = out_dir / f"analysis_batch_{batch_idx:03d}.json"
            if cache_path.is_file():
                try:
                    cached = _RawWorldAnalysis.model_validate(json.loads(cache_path.read_text(encoding="utf-8")))
                    if cached.beat_scene_graphs or cached.environments or cached.events:
                        analysis.environments.extend(cached.environments)
                        analysis.beat_scene_graphs.extend(cached.beat_scene_graphs)
                        analysis.events.extend(cached.events)
                        continue
                except (OSError, ValueError):
                    pass
            if progress is not None:
                progress(
                    "vision_world_analysis",
                    f"Analyzing beats {batch[0].beat_id}..{batch[-1].beat_id} (batch {batch_idx + 1}/{len(batches)})",
                    {"batch_idx": batch_idx, "beat_count": len(batch)},
                )
            raw = await self._invoke_analysis(characters=characters, beats=batch)
            analysis.environments.extend(raw.environments)
            analysis.beat_scene_graphs.extend(raw.beat_scene_graphs)
            analysis.events.extend(raw.events)
            cache_path.write_text(raw.model_dump_json(indent=2), encoding="utf-8")

        known_beat_ids = {b.beat_id for b in all_beats}
        environments, env_key_to_id = self._merge_environments(analysis.environments, all_beats, out_dir)
        graphs = self._normalize_scene_graphs(analysis.beat_scene_graphs, known_beat_ids, env_key_to_id)
        events = self._normalize_events(analysis.events, known_beat_ids, env_key_to_id)

        bible = self.build_character_bible(
            chunks=chunks,
            characters=characters,
            character_seed_refs=character_seed_refs,
        )

        # Backfill beat evidence into the character bible from beats themselves.
        beat_id_lookup = {b.beat_id: b for b in all_beats}
        for entry in bible:
            evidence: List[str] = []
            for beat_id, beat in beat_id_lookup.items():
                if any(
                    entry.character_key.lower() in str(p).lower() or str(p).lower() in entry.character_key.lower()
                    for p in beat.characters_present
                ):
                    evidence.append(beat_id)
            entry.source_beat_ids = sorted(set(entry.source_beat_ids) | set(evidence))

        model = VisionWorldModel(
            chapters=sorted(set(chapters)),
            characters=bible,
            environments=environments,
            beat_scene_graphs=graphs,
            events=events,
            notes="\n".join(dict.fromkeys(c.chunk_tail_summary for c in chunks if c.chunk_tail_summary)),
        )

        # Report-only duplicate-identity analysis (e.g. the same child the
        # reading pass called both "Hojin" and "Young Boy with Soccer Ball").
        if identity_dedup_embedder is not None:
            from agents.identity_dedup import detect_duplicate_identities

            try:
                dedup = detect_duplicate_identities(
                    bible,
                    embedder=identity_dedup_embedder,
                    threshold=identity_dedup_threshold,
                )
            except Exception:  # noqa: BLE001 - report-only, never fatal
                dedup = None
            model.identity_dedup = dedup
            if dedup is not None:
                try:
                    (out_dir / "identity_dedup.json").write_text(
                        dedup.model_dump_json(indent=2),
                        encoding="utf-8",
                    )
                except OSError:
                    pass
                if progress is not None:
                    suspected = len(dedup.suspected_pairs())
                    progress(
                        "vision_identity_dedup",
                        (
                            f"Identity dedup ({dedup.embedder}): "
                            f"{suspected} suspected duplicate pair(s)"
                            if dedup.status == "ok"
                            else f"Identity dedup skipped: {dedup.reason}"
                        ),
                        {"suspected": suspected, "status": dedup.status},
                    )

        (out_dir / "world_model.json").write_text(model.model_dump_json(indent=2), encoding="utf-8")
        return model

    # ------------------------------------------------------------------
    # Normalization / merging helpers (deterministic)
    # ------------------------------------------------------------------
    def _merge_environments(
        self,
        candidates: Sequence[_RawEnvironmentCandidate],
        all_beats: Sequence[MangaBeat],
        out_dir: Path,
    ) -> Tuple[List[EnvironmentBibleEntry], Dict[str, str]]:
        beat_anchor_by_id: Dict[str, str] = {}
        known_beat_ids = {b.beat_id for b in all_beats}
        for beat in all_beats:
            if beat.anchor_image_path:
                beat_anchor_by_id[beat.beat_id] = beat.anchor_image_path

        merged: List[EnvironmentBibleEntry] = []
        key_to_id: Dict[str, str] = {}

        def _candidate_from_env_key(candidate: _RawEnvironmentCandidate, env_id: str, name: str) -> EnvironmentBibleEntry:
            # Evidence trail: every chapter beat witnessing this environment;
            # reference images only from beats that actually have panel crops.
            witnessed = [b for b in candidate.beat_ids if b in known_beat_ids]
            refs = [beat_anchor_by_id[b] for b in witnessed if b in beat_anchor_by_id][:2]
            return EnvironmentBibleEntry(
                environment_id=env_id,
                name=name or candidate.name or "Unknown location",
                env_type=candidate.env_type,
                indoor=bool(candidate.indoor),
                lighting=candidate.lighting,
                time_of_day=candidate.time_of_day or "unknown",
                weather=candidate.weather or "unknown",
                key_props=[p for p in candidate.key_props if str(p).strip()][:8],
                foreground_desc=candidate.foreground_desc,
                midground_desc=candidate.midground_desc,
                background_desc=candidate.background_desc,
                source_beat_ids=witnessed,
                reference_image_paths=refs,
            )

        for candidate in candidates:
            key = " ".join(str(candidate.key or candidate.name or "").split()).lower()
            if not key:
                continue
            matched: Optional[EnvironmentBibleEntry] = None
            if key in key_to_id:
                target_id = key_to_id[key]
                matched = next((e for e in merged if e.environment_id == target_id), None)
            else:
                best_score = 0.0
                best_entry: Optional[EnvironmentBibleEntry] = None
                for entry in merged:
                    score = _env_similarity(candidate, entry)
                    if score > best_score:
                        best_score = score
                        best_entry = entry
                # Name/setting similarity above threshold OR shared beats -> same location.
                if best_entry is not None and (
                    best_score >= 2.0 or (_token_jaccard(candidate.name, best_entry.name) >= _ENV_MERGE_TOKEN_JACCARD)
                ):
                    matched = best_entry

            if matched is None:
                env_id = f"env_{len(merged) + 1:03d}"
                entry = _candidate_from_env_key(candidate, env_id, candidate.name)
                merged.append(entry)
                key_to_id[key] = env_id
            else:
                key_to_id[key] = matched.environment_id
                for beat_id in candidate.beat_ids:
                    if beat_id in known_beat_ids and beat_id not in matched.source_beat_ids:
                        matched.source_beat_ids.append(beat_id)
                if candidate.key and candidate.key not in matched.key_props and len(candidate.key) <= 32:
                    pass  # keys are internal labels, not props
                if candidate.foreground_desc and not matched.foreground_desc:
                    matched.foreground_desc = candidate.foreground_desc
                if candidate.midground_desc and not matched.midground_desc:
                    matched.midground_desc = candidate.midground_desc
                if candidate.background_desc and not matched.background_desc:
                    matched.background_desc = candidate.background_desc
                for prop in candidate.key_props:
                    if prop and prop not in matched.key_props and len(matched.key_props) < 8:
                        matched.key_props.append(prop)

        merged.sort(key=lambda e: e.environment_id)
        return merged, key_to_id

    def _normalize_scene_graphs(
        self,
        raw_graphs: Sequence[_RawSceneGraph],
        known_beat_ids: set[str],
        env_key_to_id: Dict[str, str],
    ) -> List[BeatSceneGraph]:
        graphs: List[BeatSceneGraph] = []
        for raw in raw_graphs:
            beat_id = str(raw.beat_id or "").strip()
            if beat_id not in known_beat_ids:
                continue
            entities: List[SpatialEntity] = []
            entity_ids: set[str] = set()
            for ent in raw.entities:
                eid = " ".join(str(ent.entity_id or ent.label or "").split())
                if not eid or eid in entity_ids:
                    continue
                entity_ids.add(eid)
                entities.append(
                    SpatialEntity(
                        entity_id=eid,
                        entity_type=ent.entity_type if ent.entity_type in {"character", "object", "environment"} else "character",
                        label=str(ent.label or eid),
                    )
                )
            relations: List[SpatialRelation] = []
            for rel in raw.relations:
                if rel.subject not in entity_ids or rel.obj not in entity_ids:
                    continue
                try:
                    conf = max(0.0, min(1.0, float(rel.confidence)))
                except (TypeError, ValueError):
                    conf = 0.6
                relations.append(
                    SpatialRelation(subject=rel.subject, relation=_safe_relation(rel.relation), obj=rel.obj, confidence=conf)
                )
            env_id = env_key_to_id.get(" ".join(str(raw.environment_key or "").split()).lower(), "")
            graphs.append(
                BeatSceneGraph(
                    beat_id=beat_id,
                    environment_id=env_id,
                    entities=entities,
                    relations=relations,
                    camera_position=str(raw.camera_position or ""),
                )
            )
        graphs.sort(key=lambda g: g.beat_id)
        return graphs

    def _normalize_events(
        self,
        raw_events: Sequence[_RawEvent],
        known_beat_ids: set[str],
        env_key_to_id: Dict[str, str],
    ) -> List[ChapterEvent]:
        events: List[ChapterEvent] = []
        for order, raw in enumerate(raw_events):
            beat_ids = [b for b in raw.beat_ids if b in known_beat_ids]
            description = str(raw.description or "").strip()
            if not description:
                continue
            events.append(
                ChapterEvent(
                    event_id=f"ev_{order + 1:03d}",
                    description=description,
                    event_type=raw.event_type if raw.event_type in _ALLOWED_EVENT_TYPES else "other",
                    beat_ids=beat_ids,
                    character_keys=[str(c) for c in raw.character_keys],
                    environment_id=env_key_to_id.get(" ".join(str(raw.environment_key or "").split()).lower(), ""),
                    order=order,
                    emotional_tone=str(raw.emotional_tone or ""),
                )
            )
        return events


def load_world_model(working_dir: str | Path) -> Optional[VisionWorldModel]:
    """Load a previously built world model, or None when absent/corrupt."""
    path = Path(working_dir) / "vision_world" / "world_model.json"
    if not path.is_file():
        return None
    try:
        return VisionWorldModel.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None
