"""Cinematic Director: manga-aware scene expansion for the Vision Layer.

Responsibilities (deliberately NARROW — it does not replace the existing
``StoryboardArtist``):

1. ``plan_scene_expansion``: decide WHERE additional cinematic coverage
   (establishing / reaction / insert / alternate-angle shots) genuinely improves
   the adaptation, grounded in the ``VisionWorldModel``. Every directive carries
   a ``category`` label (``canon`` / ``inferred`` / ``creative``) so generated
   shots never silently change the story.
2. ``format_world_context``: build a compact, deterministic world-context text
   block (character bible, environment bible, spatial relations, event
   timeline) injected into the existing storyboard prompt. This avoids sending
   full chapter images to every LLM call.
3. ``apply_expansion_to_script``: append directives + world context to the
   canon scene script in the exact ``[RefBeat: ...]``-tag format the existing
   ``StoryboardArtist`` and ``bind_storyboard_to_panel_anchors`` already
   understand, keeping backward compatibility.
4. ``build_shot_provenance``: per-shot traceability records (source beats +
   category) for downstream QA.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from interfaces import (
    BeatExpansionDirective,
    MangaBeat,
    SceneExpansionPlan,
    ShotBriefDescription,
    ShotProvenance,
    VisionWorldModel,
)
from utils.retry import after_func
from utils.robust_json_parser import TrailingCommaTolerantPydanticOutputParser as PydanticOutputParser

_REFBEAT_TAG_RE = re.compile(r"\s*\[(?:RefBeat|BEAT)\s*:\s*([^\]]+)\]\s*", re.IGNORECASE)

_MAX_ADDITIONAL_SHOTS_PER_BEAT = 3

_system_prompt_template_expand = """
[Role]
You are a cinematic director adapting manga chapters into an anime episode.

[Task]
You receive one scene's canon beats (extracted from manga panels) plus a world
model summary (character bible, environment bible, spatial relations, event
timeline). Keep the adaptation SIMPLE, CLEAN, and MINIMALIST: add additional
cinematic coverage ONLY where a bare transition would otherwise confuse the
viewer, and avoid stuffing redundant shots into short scenes.

[What you may add (use sparingly)]
- establishing_wide / aerial_reveal: show the environment BEFORE the action only
  if the location was not yet introduced, using the environment bible's literal
  description (architecture, props, lighting, time of day, weather). NEVER invent locations.
- reaction_close_up: a present character's facial reaction when no reaction
  framing exists nearby. No new dialogue content.
- insert_detail: one brief detail shot of a key prop/injury only when needed to
  clarify a beat, never back-to-back with a nearly identical wide pose.
- alternate_angle / over_the_shoulder: only to prevent back-to-back identical
  compositions (e.g. replacing a duplicate rear view with a clean side profile).
- environment_b_roll: quiet atmospheric coverage only when a major scene pause
  is genuinely missing. NEVER insert extra ground crash/explosion impacts after a
  sky streak if the next beat cuts smoothly into a calm aftermath scene.

[Hard Rules]
1. MINIMALIST RESTRAINT (NO BLOAT): Default to returning an empty
   `additional_shots` list (`[]`) on most beats! Add AT MOST 1 clarifying shot on
   a beat only when truly necessary; never pad every beat just to hit the cap of
   {max_shots_per_beat}.
2. NO CONSECUTIVE DUPLICATES: Never stack `establishing_wide`, `return_to_panel`,
   or repeated poses across adjacent beats that depict the same moment (e.g.
   consecutive panting/kneeling beats).
3. CATEGORY LABELS: shots that re-frame a canonical moment are "inferred";
   added coverage is "creative". Never label an invented plot event as canon.
4. NO new characters, NO new dialogue, NO new plot events, NO new locations.
   Creative shots may expand VISUAL coverage only.
5. `preserve_panel_composition` stays true unless the beat is purely a
   transition (empty scenery / SFX panel).
6. visual_hint must preserve world-model facts verbatim: character names, exact
   garment tonal values & light reactivity (e.g. medium slate-grey shifting to
   cool light grey under directional/rim light rather than pitch black),
   environment description, key_props, and lighting angle.
7. Every directive MUST set beat_id to a beat id given in the input.

[Output Format]
Return ONLY valid JSON matching:
{format_instructions}
"""

_human_prompt_template_expand = """
WORLD MODEL SUMMARY:
{world_context}

SCENE BEATS (canon, in reading order):
{beats_str}

Produce the expansion plan JSON for this scene:
"""


class SceneExpansionResponse(BaseModel):
    scene_idx: int = Field(default=0)
    beat_directives: List[BeatExpansionDirective] = Field(default_factory=list)
    notes: str = Field(default="")


# Shot types kept when the canon gate restricts cinematic expansion.
_CANON_ALLOWED_SHOT_TYPES = {
    "establishing_wide",
    "reaction_close_up",
    "insert_detail",
    "over_the_shoulder",
    "alternate_angle",
}


def _restrict_expansion_directives(directives, scene_beats):
    """Trim expansion directives to canon-safe coverage (phase 01, quality.canon_gate).

    - keep only re-framing / reaction / insert coverage (drop b_roll, aerial, return_to_panel),
    - allow ``establishing_wide`` only on the first beat or an ``is_scene_break`` beat,
    - cap total added shots at 25% of the scene's beats, dropping lowest confidence first.
    """
    if not scene_beats:
        return directives
    first_beat_id = scene_beats[0].beat_id
    break_beats = {b.beat_id for b in scene_beats if getattr(b, "is_scene_break", False)}
    budget = max(0, len(scene_beats) // 4)

    kept: List = []  # (confidence, directive_beat_id, shot)
    for directive in directives:
        filtered = []
        for shot in directive.additional_shots:
            if shot.shot_type not in _CANON_ALLOWED_SHOT_TYPES:
                continue
            if shot.shot_type == "establishing_wide" and directive.beat_id != first_beat_id and directive.beat_id not in break_beats:
                continue
            filtered.append(shot)
        directive.additional_shots = filtered
        for shot in filtered:
            kept.append((shot.confidence, directive.beat_id, shot))

    if len(kept) > budget:
        # keep the highest-confidence `budget` shots; clear the rest.
        kept.sort(key=lambda t: t[0], reverse=True)
        survivors = {id(t[2]) for t in kept[:budget]}
        for directive in directives:
            directive.additional_shots = [s for s in directive.additional_shots if id(s) in survivors]

    return [d for d in directives if d.additional_shots or d.preserve_panel_composition]


def extract_shot_beat_refs(text: str) -> List[str]:
    """Pull ``[RefBeat: id[,id2]]`` beat references out of a shot description."""
    beat_ids: List[str] = []
    for match in _REFBEAT_TAG_RE.finditer(str(text or "")):
        for token in re.split(r"[,;\s]+", match.group(1).strip()):
            token = token.strip()
            if token and token not in beat_ids:
                beat_ids.append(token)
    return beat_ids


def format_world_context(
    world_model: VisionWorldModel,
    scene_beats: Sequence[MangaBeat],
    *,
    max_chars: int = 2400,
) -> str:
    """Compact deterministic world-model text block for one scene.

    Text-only: crops/references travel separately via the existing panel
    anchor mechanism, keeping storyboard prompts small.
    """
    beat_ids = {b.beat_id for b in scene_beats}
    scene_graphs = {g.beat_id: g for g in world_model.beat_scene_graphs if g.beat_id in beat_ids}
    env_ids_in_scene = {g.environment_id for g in scene_graphs.values() if g.environment_id}
    environments = [e for e in world_model.environments if e.environment_id in env_ids_in_scene]
    events = [ev for ev in world_model.events if beat_ids & set(ev.beat_ids)]

    lines: List[str] = []

    lines.append("CHARACTER BIBLE (canonical names; use verbatim):")
    present_keys: set[str] = set()
    for beat in scene_beats:
        present_keys.update(str(c) for c in beat.characters_present)
    for entry in world_model.characters:
        if entry.character_key in present_keys or any(
            k.lower() in entry.character_key.lower() or entry.character_key.lower() in k.lower()
            for k in present_keys
        ):
            feats = entry.permanent_features or "(no permanent features recorded)"
            temp = f" outfit/state: {entry.temporary_features}" if entry.temporary_features else ""
            lines.append(f"- {entry.character_key}: {feats}{temp}")

    if environments:
        lines.append("ENVIRONMENT BIBLE (shared locations for this scene):")
        for env in environments:
            parts = [f"- {env.environment_id} '{env.name}'"]
            parts.append(f"type={env.env_type or 'unknown'}")
            parts.append("indoor" if env.indoor else "outdoor")
            if env.lighting:
                parts.append(f"lighting: {env.lighting}")
            if env.time_of_day != "unknown":
                parts.append(f"time: {env.time_of_day}")
            if env.weather != "unknown":
                parts.append(f"weather: {env.weather}")
            if env.key_props:
                parts.append(f"props: {', '.join(env.key_props[:6])}")
            lines.append(" | ".join(parts))
            zones = " / ".join(
                z for z in (env.foreground_desc, env.midground_desc, env.background_desc) if z
            )
            if zones:
                lines.append(f"  depth zones: {zones}")

    scene_graph_lines: List[str] = []
    for beat in scene_beats:
        graph = scene_graphs.get(beat.beat_id)
        if graph is None:
            continue
        rels = "; ".join(f"{r.subject} {r.relation} {r.obj}" for r in graph.relations[:6])
        entry = f"- [{beat.beat_id}] camera: {graph.camera_position or 'unspecified'}"
        if rels:
            entry += f" | spatial: {rels}"
        scene_graph_lines.append(entry)
    if scene_graph_lines:
        lines.append("SPATIAL SCENE GRAPHS (qualitative, from the panels):")
        lines.extend(scene_graph_lines)

    if events:
        lines.append("EVENT TIMELINE (this scene):")
        for ev in events:
            tone = f" ({ev.emotional_tone})" if ev.emotional_tone else ""
            lines.append(f"- {ev.description}{tone}")

    block = "\n".join(lines).strip()
    if max_chars > 0 and len(block) > max_chars:
        block = block[: max_chars - 3].rstrip() + "..."
    return block


class CinematicDirector:
    """Plans per-scene cinematic expansion grounded in the world model."""

    def __init__(self, chat_model: Runnable) -> None:
        self.chat_model = chat_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=6),
        after=after_func,
        reraise=True,
    )
    async def _invoke_expansion(
        self,
        *,
        scene_idx: int,
        scene_beats: Sequence[MangaBeat],
        world_context: str,
    ) -> SceneExpansionResponse:
        parser = PydanticOutputParser(pydantic_object=SceneExpansionResponse)
        beats_str = WorldModelBeatsFormatter.format_beats(scene_beats)
        messages = [
            SystemMessage(
                content=_system_prompt_template_expand.format(
                    format_instructions=parser.get_format_instructions(),
                    max_shots_per_beat=_MAX_ADDITIONAL_SHOTS_PER_BEAT,
                )
            ),
            HumanMessage(
                content=_human_prompt_template_expand.format(world_context=world_context, beats_str=beats_str)
            ),
        ]
        chain = self.chat_model | parser
        raw: SceneExpansionResponse = await chain.ainvoke(messages)
        return raw

    async def plan_scene_expansion(
        self,
        *,
        scene_idx: int,
        scene_beats: Sequence[MangaBeat],
        world_model: VisionWorldModel,
        working_dir: str | Path,
        max_context_chars: int = 2400,
        progress: Optional[Callable[[str, str, Dict[str, Any] | None], None]] = None,
        restrict_shot_types: bool = False,
    ) -> SceneExpansionPlan:
        out_dir = Path(working_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cache_path = out_dir / "expansion_plan.json"
        world_context = format_world_context(world_model, scene_beats, max_chars=max_context_chars)

        if cache_path.is_file():
            try:
                cached = SceneExpansionPlan.model_validate(json.loads(cache_path.read_text(encoding="utf-8")))
                if cached.world_context_block or cached.beat_directives:
                    return cached
            except (OSError, ValueError):
                pass

        if progress is not None:
            progress(
                "cinematic_expansion_plan",
                f"Planning cinematic expansion for scene {scene_idx}",
                {"scene_idx": scene_idx, "beat_count": len(scene_beats)},
            )
        raw = await self._invoke_expansion(
            scene_idx=scene_idx,
            scene_beats=scene_beats,
            world_context=world_context or "(no world model available)",
        )
        known_beat_ids = {b.beat_id for b in scene_beats}
        directives = [d for d in raw.beat_directives if d.beat_id in known_beat_ids]
        for directive in directives:
            directive.additional_shots = directive.additional_shots[:_MAX_ADDITIONAL_SHOTS_PER_BEAT]
        if restrict_shot_types:
            directives = _restrict_expansion_directives(directives, scene_beats)
        plan = SceneExpansionPlan(
            scene_idx=scene_idx,
            beat_directives=directives,
            world_context_block=world_context,
            notes=str(raw.notes or ""),
        )
        cache_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        return plan

    # ------------------------------------------------------------------
    # Script augmentation + provenance
    # ------------------------------------------------------------------
    @staticmethod
    def apply_expansion_to_script(script_text: str, plan: SceneExpansionPlan) -> str:
        """Append the expansion directives + world context to a scene script.

        The base canon script (beats, dialogues, Vietsub tags) is NEVER
        modified — directives are appended in the same textual format the
        existing storyboard artist consumes.
        """
        blocks: List[str] = []
        if plan.world_context_block:
            blocks.append(
                "WORLD CONTEXT (structured, from the Manga Vision Layer — treat as ground truth for environments, "
                "character features, and spatial layout):\n"
                f"{plan.world_context_block}"
            )
        directive_lines: List[str] = []
        for directive in plan.beat_directives:
            for shot in directive.additional_shots:
                cam = f", camera {shot.camera_angle or 'unspecified'}"
                move = f", movement {shot.camera_movement or 'static'}"
                dur = f", ~{shot.target_duration_sec:.1f}s" if shot.target_duration_sec else ""
                directive_lines.append(
                    f"[RefBeat: {directive.beat_id}] ADD {shot.shot_type} ({shot.category}){cam}{move}{dur}: "
                    f"{shot.visual_hint or shot.purpose}"
                )
            if directive.preserve_panel_composition and not directive.additional_shots:
                directive_lines.append(
                    f"[RefBeat: {directive.beat_id}] PRESERVE original panel composition as its own shot."
                )
        if directive_lines:
            blocks.append(
                "CINEMATIC EXPANSION DIRECTIVES (category labels are binding: 'inferred' shots re-frame canon "
                "moments, 'creative' shots add coverage without new plot content; never invent new characters, "
                "dialogue, or locations):\n" + "\n".join(directive_lines)
            )
        if not blocks:
            return script_text
        return script_text.rstrip() + "\n\n" + "\n\n".join(blocks) + "\n"

    @staticmethod
    def build_shot_provenance(
        *,
        storyboard: Sequence[ShotBriefDescription],
        scene_idx: int,
        known_beat_ids: Sequence[str],
        expansion_plan: Optional[SceneExpansionPlan] = None,
    ) -> List[ShotProvenance]:
        """Assign each shot a traceability record: source beats + category.

        Category rules:
        - explicit ``[RefBeat]`` tag      -> ``canon`` (panel composition reused)
        - tag + expansion directive match -> keeps the directive's category
        - no tag, text binds to a beat    -> ``inferred``
        - no tag, no binding              -> ``creative``
        """
        known = set(known_beat_ids)
        directive_by_beat: Dict[str, List[str]] = {}
        purposes: Dict[str, str] = {}
        if expansion_plan is not None:
            for directive in expansion_plan.beat_directives:
                for shot in directive.additional_shots:
                    directive_by_beat.setdefault(directive.beat_id, []).append(shot.shot_type)
                    purposes.setdefault(shot.shot_type, shot.purpose)

        provenance: List[ShotProvenance] = []
        for shot in storyboard:
            beat_refs = [
                bid for bid in extract_shot_beat_refs(f"{shot.visual_desc} {shot.audio_desc or ''}") if bid in known
            ]
            if beat_refs:
                category = "canon"
                # An explicit add-directive for this beat means the shot likely
                # realizes additional coverage; keep canon for tagged shots but
                # record purpose when the shot type matches a directive.
                for bid in beat_refs:
                    shot_types = directive_by_beat.get(bid) or []
                    if shot_types:
                        category = "inferred"
                        break
            else:
                category = "creative"
            purpose = ""
            if category != "canon":
                for shot_type in directive_by_beat.get(beat_refs[0] if beat_refs else "", []):
                    purpose = purposes.get(shot_type, "")
                    if purpose:
                        break
            provenance.append(
                ShotProvenance(
                    shot_idx=shot.idx,
                    scene_idx=scene_idx,
                    source_beat_ids=beat_refs,
                    category=category,  # type: ignore[arg-type]
                    purpose=purpose,
                    confidence=0.9 if beat_refs else 0.4,
                )
            )
        return provenance


class WorldModelBeatsFormatter:
    """Shared compact beat formatting for the director prompt."""

    @staticmethod
    def format_beats(beats: Sequence[MangaBeat]) -> str:
        lines: List[str] = []
        for beat in beats:
            dlg = " | ".join(f"{d.speaker}: '{d.line}'" for d in beat.dialogues)
            lines.append(
                f"[{beat.beat_id}] setting='{beat.setting}' chars={list(beat.characters_present)} "
                f"action='{beat.action_desc}' sfx={list(beat.sfx)} dialogues=[{dlg}]"
            )
        return "\n".join(lines)
