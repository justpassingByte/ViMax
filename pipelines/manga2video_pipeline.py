from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Literal

from moviepy import VideoFileClip

from agents import shot_linter
from agents.cinematic_director import CinematicDirector
from agents.manga_reader import MangaReader
from agents.storyboard_artist import StoryboardArtist
from agents.visual_qa import VisualQAReviewer
from agents.world_model_builder import WorldModelBuilder, load_world_model
from interfaces import (
    Camera,
    CharacterInScene,
    MangaBeat,
    MangaChunkExtraction,
    MangaEpisodePlan,
    MangaSeriesEpisodeEntry,
    MangaSeriesPlan,
    ShotBriefDescription,
    ShotDescription,
    VisualQAReport,
    VisionWorldModel,
)
from pipelines.script2video_pipeline import Script2VideoPipeline, _normalize_model_list, _resolve_workspace_root
from tools.manga_source import (
    ComicManifest,
    group_chapters_for_episodes,
    ingest_manga_sources,
    parse_chapter_range,
)
from utils.image_selection import DEFAULT_IMAGE_CANDIDATES, validate_image_candidate_count
from utils.video import concatenate_video_files


_REFBEAT_TAG_RE = re.compile(r"\s*\[(?:RefBeat|BEAT)\s*:\s*([^\]]+)\]\s*", re.IGNORECASE)
_BRACKET_WORD_RE = re.compile(r"[a-zA-Z0-9À-ỹ]+")


def _clean_refbeat_tags(text: str) -> Tuple[str, List[str]]:
    raw = str(text or "")
    beat_ids: List[str] = []
    for m in _REFBEAT_TAG_RE.finditer(raw):
        for token in re.split(r"[,;\s]+", m.group(1).strip()):
            tok = token.strip()
            if tok and tok not in beat_ids:
                beat_ids.append(tok)
    cleaned = _REFBEAT_TAG_RE.sub(" ", raw)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned, beat_ids


def _text_tokens(text: str) -> set[str]:
    return {tok.lower() for tok in _BRACKET_WORD_RE.findall(str(text or "")) if len(tok) >= 2}


def bind_storyboard_to_panel_anchors(
    storyboard: List[ShotBriefDescription],
    scene_beats: Sequence[MangaBeat],
    scene_dir: str | Path,
) -> Tuple[List[ShotBriefDescription], Dict[int, List[Tuple[str, str]]]]:
    """Bind each storyboard shot to 1-2 original comic panel crops and strip `[RefBeat: ...]` tags."""
    scene_root = Path(scene_dir)
    scene_root.mkdir(parents=True, exist_ok=True)
    beats_list = list(scene_beats)
    beat_by_id = {b.beat_id: b for b in beats_list}
    n_shots = max(1, len(storyboard))
    n_beats = max(1, len(beats_list))

    clean_shots: List[ShotBriefDescription] = []
    shot_panel_refs: Dict[int, List[Tuple[str, str]]] = {}
    refs_json_payload: Dict[str, List[Dict[str, str]]] = {}

    for shot_pos, shot in enumerate(storyboard):
        clean_vis, vis_tags = _clean_refbeat_tags(shot.visual_desc)
        clean_aud, aud_tags = _clean_refbeat_tags(shot.audio_desc)
        clean_shot = shot.model_copy(
            update={
                "visual_desc": clean_vis,
                "audio_desc": clean_aud,
            }
        )
        clean_shots.append(clean_shot)

        selected_beats: List[MangaBeat] = []
        seen_ids: set[str] = set()
        for bid in vis_tags + aud_tags:
            b = beat_by_id.get(bid)
            if b is not None and b.anchor_image_path and b.beat_id not in seen_ids:
                selected_beats.append(b)
                seen_ids.add(b.beat_id)
            if len(selected_beats) >= 2:
                break

        if not selected_beats and beats_list:
            shot_tokens = _text_tokens(f"{clean_vis} {clean_aud}")
            shot_rel = shot_pos / max(1, n_shots - 1) if n_shots > 1 else 0.0
            best_score = -1.0
            best_beat: Optional[MangaBeat] = None
            for b_pos, beat in enumerate(beats_list):
                if not beat.anchor_image_path:
                    continue
                beat_rel = b_pos / max(1, n_beats - 1) if n_beats > 1 else 0.0
                pos_score = max(0.0, 1.0 - abs(shot_rel - beat_rel)) * 2.5
                dlg_text = " ".join(f"{d.speaker} {d.line} {d.vietsub}" for d in beat.dialogues)
                beat_tokens = _text_tokens(f"{beat.setting} {beat.action_desc} {beat.visual_details} {dlg_text}")
                overlap = (len(shot_tokens & beat_tokens) / max(1, min(len(shot_tokens), len(beat_tokens)))) if shot_tokens and beat_tokens else 0.0
                char_overlap = sum(
                    1
                    for ch in beat.characters_present
                    if any(tok in f"{clean_vis} {clean_aud}".lower() for tok in _text_tokens(ch))
                )
                score = pos_score + overlap * 4.0 + min(2, char_overlap) * 1.5
                if score > best_score:
                    best_score = score
                    best_beat = beat
            if best_beat is not None:
                selected_beats.append(best_beat)

        has_speech = bool(re.search(r"\[(Speaker|Thought|Inner\s*Monologue)\]", clean_aud, re.I)) or bool(
            re.search(r"\b(?:says?|thinks?)\s*:", clean_vis, re.I)
        )
        has_vietsub = bool(re.search(r"Vietsub\s*:", clean_aud, re.I))
        if has_speech and not has_vietsub and selected_beats:
            canon_vietsubs = [
                d.vietsub.strip()
                for b in selected_beats
                for d in b.dialogues
                if getattr(d, "vietsub", None) and d.vietsub.strip()
            ]
            if canon_vietsubs:
                clean_aud = (
                    f"{clean_aud.rstrip()}\n(Vietsub: {' '.join(canon_vietsubs)})"
                    if clean_aud.strip()
                    else f"(Vietsub: {' '.join(canon_vietsubs)})"
                )
                clean_shots[-1] = clean_shots[-1].model_copy(update={"audio_desc": clean_aud})

        pairs: List[Tuple[str, str]] = []
        entries: List[Dict[str, str]] = []
        for beat in selected_beats[:2]:
            anchor = str(beat.anchor_image_path or "").strip()
            if not anchor:
                continue
            desc = (
                f"ORIGINAL COMIC PANEL COMPOSITION & POSE REFERENCE (Ch.{beat.chapter} p.{beat.page_idx}, {beat.beat_id}: {beat.action_desc}). "
                "Redraw cleanly as borderless textless 2D Japanese anime matching the character turnaround portraits; "
                "REMOVE all speech bubbles, panel gutters, and SFX text."
            )
            pairs.append((anchor, desc))
            entries.append({"path": anchor, "description": desc, "beat_id": beat.beat_id})

        if pairs:
            shot_panel_refs[clean_shot.idx] = pairs
            refs_json_payload[str(clean_shot.idx)] = entries

    (scene_root / "storyboard.json").write_text(
        json.dumps([s.model_dump() for s in clean_shots], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (scene_root / "shot_panel_refs.json").write_text(
        json.dumps(refs_json_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return clean_shots, shot_panel_refs


_SRT_TIME_RE = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{3})")


def _parse_srt_time(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600.0 + int(m) * 60.0 + int(s) + int(ms) / 1000.0


def _merge_scene_subtitles(
    scene_dirs: Sequence[Path],
    scene_durations: Sequence[float],
    dest_dir: Path,
    *,
    suffix: str,
    font: str,
    font_size: int,
) -> Tuple[Optional[str], Optional[str]]:
    from utils.audio import write_ass_subtitles, write_srt_subtitles

    merged_cues: List[Tuple[float, float, str]] = []
    offset = 0.0
    for scene_dir, scene_dur in zip(scene_dirs, scene_durations):
        srt_path = scene_dir / f"final_video_{suffix}.srt"
        if srt_path.is_file():
            raw = srt_path.read_text(encoding="utf-8")
            for block in re.split(r"\r?\n\r?\n+", raw.strip()):
                lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
                if len(lines) < 2:
                    continue
                time_line = lines[1] if "-->" in lines[1] else lines[0]
                text_lines = lines[2:] if "-->" in lines[1] else lines[1:]
                m = _SRT_TIME_RE.search(time_line)
                text = " ".join(text_lines).strip()
                if not m or not text:
                    continue
                t0 = _parse_srt_time(m.group(1), m.group(2), m.group(3), m.group(4))
                t1 = _parse_srt_time(m.group(5), m.group(6), m.group(7), m.group(8))
                start_abs = offset + max(0.0, t0)
                end_abs = offset + min(scene_dur, max(t0 + 0.1, t1)) if scene_dur > 0 else offset + t1
                merged_cues.append((start_abs, end_abs, text))
        offset += float(scene_dur)

    if not merged_cues:
        return None, None
    srt_out = write_srt_subtitles(merged_cues, str(dest_dir / f"final_video_{suffix}.srt"))
    ass_out = write_ass_subtitles(merged_cues, str(dest_dir / f"final_video_{suffix}.ass"), font=font, font_size=font_size)
    return srt_out, ass_out


def _emit_progress(
    progress: Callable[[str, str, Dict[str, Any] | None], None] | None,
    stage: str,
    message: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    if progress is not None:
        progress(stage, message, metadata or {})


def _scene_progress_scope(
    progress: Callable[[str, str, Dict[str, Any] | None], None] | None,
    scene_idx: int,
):
    if progress is None:
        return None

    def _emit(stage: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        payload = {"scene_idx": scene_idx}
        payload.update(metadata or {})
        progress(stage, message, payload)

    return _emit


def _align_chunks_to_episode_scenes(
    all_chunks: Sequence[MangaChunkExtraction],
    beats_by_scene: Dict[int, List[MangaBeat]],
) -> List[MangaChunkExtraction]:
    """Rebuild canonical per-chapter chunks using post-split beat IDs from ``beats_by_scene``.

    ``build_episode_scenes`` expands multi-dialogue panels into sequential beat IDs
    (``ch01_b000..ch01_b021``). Feeding raw unsplit chunks directly into
    ``WorldModelBuilder`` causes beat_id drift against ``scene_*/beat_map.json``.
    """
    ordered_beats: List[MangaBeat] = []
    for sc_idx in sorted(beats_by_scene):
        ordered_beats.extend(beats_by_scene[sc_idx])
    if not ordered_beats:
        return list(all_chunks)

    notes_by_ch: Dict[int, List[str]] = {}
    for ch in all_chunks:
        if ch.chunk_tail_summary:
            notes_by_ch.setdefault(ch.chapter, []).append(ch.chunk_tail_summary)

    beats_by_ch: Dict[int, List[MangaBeat]] = {}
    for beat in ordered_beats:
        beats_by_ch.setdefault(beat.chapter, []).append(beat)

    return [
        MangaChunkExtraction(
            chapter=ch_num,
            beats=ch_beats,
            chunk_tail_summary="\n".join(dict.fromkeys(notes_by_ch.get(ch_num, []))),
        )
        for ch_num, ch_beats in sorted(beats_by_ch.items())
    ]


async def _build_or_load_world_model(
    *,
    pipeline: "Manga2VideoPipeline",
    manifest: ComicManifest,
    all_chunks: List[MangaChunkExtraction],
    characters: List[CharacterInScene],
    character_seed_refs: Dict[str, List[str]],
    work_root: Path,
    progress: Callable[[str, str, Dict[str, Any] | None], None] | None,
) -> Optional[VisionWorldModel]:
    """Build (or reuse) the chapter-level world model when the vision layer is enabled.

    Returns None when the feature flag is off — the pipeline then behaves
    exactly as before this layer existed.
    """
    from agent_runtime.config import manga_vision_enabled

    workspace_root = _resolve_workspace_root(str(work_root))
    if not manga_vision_enabled(workspace_root):
        return None

    cached = load_world_model(work_root)
    if cached is not None and (cached.beat_scene_graphs or cached.environments or cached.events):
        _emit_progress(progress, "vision_world_cached", "Reusing cached manga vision world model")
        return cached

    if not all_chunks:
        return None
    _emit_progress(progress, "vision_world_build_start", "Building manga vision world model")
    from agents.identity_dedup import build_identity_dedup_embedder
    from agent_runtime.config import manga_vision_identity_dedup_threshold

    dedup_embedder, dedup_kind = build_identity_dedup_embedder(workspace_root)
    _emit_progress(
        progress,
        "vision_identity_dedup_embedder",
        f"Identity dedup embedder: {dedup_kind}",
        {"kind": dedup_kind},
    )
    builder = WorldModelBuilder(chat_model=pipeline.chat_model)
    world_model = await builder.build_world_model(
        chunks=all_chunks,
        characters=characters,
        character_seed_refs=character_seed_refs,
        working_dir=work_root,
        progress=progress,
        identity_dedup_embedder=dedup_embedder,
        identity_dedup_threshold=manga_vision_identity_dedup_threshold(workspace_root),
    )
    _emit_progress(
        progress,
        "vision_world_build_done",
        (
            f"World model ready: {len(world_model.characters)} characters, "
            f"{len(world_model.environments)} environments, "
            f"{len(world_model.beat_scene_graphs)} scene graphs, {len(world_model.events)} events"
        ),
        {
            "characters": len(world_model.characters),
            "environments": len(world_model.environments),
            "scene_graphs": len(world_model.beat_scene_graphs),
            "events": len(world_model.events),
        },
    )
    return world_model


async def _augment_scene_script_with_vision_async(
    *,
    scene_idx: int,
    scene_dir: Path,
    scene_beats: List[MangaBeat],
    world_model: Optional[VisionWorldModel],
    chat_model: Any,
    progress: Callable[[str, str, Dict[str, Any] | None], None] | None,
) -> str:
    """Async version used by ``plan_episode`` to build/apply the expansion plan."""
    from agent_runtime.config import manga_vision_enabled, manga_vision_max_context_chars, quality_canon_gate

    scene_script_path = scene_dir / "script.txt"
    scene_script = scene_script_path.read_text(encoding="utf-8")
    workspace_root = _resolve_workspace_root(str(scene_dir))
    if world_model is None or not manga_vision_enabled(workspace_root) or not scene_beats:
        return scene_script
    restrict_shot_types = quality_canon_gate(workspace_root)

    cached_plan = None
    expansion_path = scene_dir / "expansion_plan.json"
    if expansion_path.is_file():
        from interfaces import SceneExpansionPlan

        try:
            cached_plan = SceneExpansionPlan.model_validate(json.loads(expansion_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            cached_plan = None

    if cached_plan is None:
        director = CinematicDirector(chat_model=chat_model)
        cached_plan = await director.plan_scene_expansion(
            scene_idx=scene_idx,
            scene_beats=scene_beats,
            world_model=world_model,
            working_dir=scene_dir,
            max_context_chars=manga_vision_max_context_chars(workspace_root),
            progress=progress,
            restrict_shot_types=restrict_shot_types,
        )

    augmented = CinematicDirector.apply_expansion_to_script(scene_script, cached_plan)
    if augmented != scene_script:
        _emit_progress(
            progress,
            "cinematic_script_augmented",
            f"Scene {scene_idx} script augmented with world context & expansion directives",
            {"scene_idx": scene_idx, "beat_directives": len(cached_plan.beat_directives)},
        )
    return augmented


def _write_scene_shot_provenance(
    *,
    scene_idx: int,
    scene_dir: Path,
    clean_storyboard: List[ShotBriefDescription],
    scene_beats: List[MangaBeat],
    expansion_plan_json_path: Path | None,
) -> None:
    """Persist per-shot provenance (source beats + category) for visual QA."""
    known_beat_ids = [b.beat_id for b in scene_beats]
    plan = None
    if expansion_plan_json_path is not None and expansion_plan_json_path.is_file():
        from interfaces import SceneExpansionPlan

        try:
            plan = SceneExpansionPlan.model_validate(json.loads(expansion_plan_json_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            plan = None

    provenance = CinematicDirector.build_shot_provenance(
        storyboard=clean_storyboard,
        scene_idx=scene_idx,
        known_beat_ids=known_beat_ids,
        expansion_plan=plan,
    )

    # Fallback: when tags were already stripped (cached re-plan), recover beat
    # bindings from the shot_panel_refs.json written by the previous bind pass.
    refs_path = scene_dir / "shot_panel_refs.json"
    if refs_path.is_file():
        try:
            refs_payload = json.loads(refs_path.read_text(encoding="utf-8"))
            beat_ids_by_shot: Dict[int, List[str]] = {}
            for shot_key, entries in refs_payload.items():
                try:
                    shot_idx = int(shot_key)
                except (TypeError, ValueError):
                    continue
                ids = [str(e.get("beat_id", "")).strip() for e in entries if isinstance(e, dict)]
                beat_ids_by_shot[shot_idx] = [i for i in ids if i]
            for prov in provenance:
                if not prov.source_beat_ids and beat_ids_by_shot.get(prov.shot_idx):
                    prov.source_beat_ids = beat_ids_by_shot[prov.shot_idx]
                    prov.category = "canon"
                    prov.confidence = max(prov.confidence, 0.8)
        except (OSError, ValueError):
            pass

    (scene_dir / "shot_provenance.json").write_text(
        json.dumps([p.model_dump() for p in provenance], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def _run_scene_visual_qa(
    *,
    pipeline: "Manga2VideoPipeline",
    scene_idx: int,
    scene_dir: Path,
    progress: Callable[[str, str, Dict[str, Any] | None], None] | None,
) -> Optional[VisualQAReport]:
    """Report-only visual QA for one rendered scene (Manga Vision Layer).

    Never raises: QA failures are logged/emitted and rendering continues.
    Returns None when the feature flag is off or required artifacts are missing.
    """
    from agent_runtime.config import manga_vision_enabled, manga_vision_qa_max_frame_pairs

    workspace_root = _resolve_workspace_root(str(scene_dir))
    if not manga_vision_enabled(workspace_root):
        return None

    storyboard_path = scene_dir / "storyboard.json"
    beat_map_path = scene_dir / "beat_map.json"
    if not storyboard_path.is_file() or not beat_map_path.is_file():
        return None

    try:
        storyboard = [
            ShotBriefDescription.model_validate(s)
            for s in json.loads(storyboard_path.read_text(encoding="utf-8"))
        ]
        beats = [
            MangaBeat.model_validate(b)
            for b in json.loads(beat_map_path.read_text(encoding="utf-8"))
        ]
        world_model = load_world_model(scene_dir.parent)
    except (OSError, ValueError):
        return None

    _emit_progress(progress, "visual_qa_start", f"Visual QA review for scene {scene_idx} (report-only)")
    try:
        report = await VisualQAReviewer(chat_model=pipeline.chat_model).review_scene(
            scene_idx=scene_idx,
            scene_dir=scene_dir,
            shot_descriptions=storyboard,
            beats=beats,
            world_model=world_model,
            max_frame_pairs=manga_vision_qa_max_frame_pairs(workspace_root),
        )
    except Exception as exc:  # noqa: BLE001 - QA must never break rendering
        _emit_progress(progress, "visual_qa_failed", f"Visual QA failed for scene {scene_idx}: {exc}")
        return None

    _emit_progress(
        progress,
        "visual_qa_done",
        f"Visual QA scene {scene_idx}: {report.summary}",
        {"failed_shots": report.failed_shots, "passed_shots": report.passed_shots},
    )
    return report


async def _apply_canon_gate(
    *,
    storyboard,
    scene_beats,
    characters,
    scene_idx: int,
    scene_dir: Path,
    script_pipe,
    scene_script: str,
    extra_req: str,
    quiet: bool,
    progress=None,
):
    """Story-fidelity gate: re-ask + deterministic repair until grounded.

    Writes ``<scene>/canon_report.json`` and returns the (possibly repaired)
    storyboard. Never raises: on any error it keeps the input storyboard so the
    run continues.
    """
    try:
        from agents.canon_gate import check_storyboard, repair_storyboard
    except Exception as exc:  # pragma: no cover - defensive import guard
        _emit_progress(progress, "canon_gate_skip", f"Canon gate unavailable: {exc}")
        return storyboard

    report = check_storyboard(storyboard, scene_beats, characters, scene_idx=scene_idx)
    for _attempt in range(2):
        if report.passed:
            break
        failures = "\n".join(f"- shot {i.shot_idx} [{i.kind}]: {i.detail}" for i in report.issues[:20])
        _emit_progress(
            progress,
            "canon_gate_retry",
            f"Canon gate scene {scene_idx}: {len(report.issues)} issue(s), re-designing",
            {"issues": len(report.issues)},
        )
        retry_req = (
            f"{extra_req}\n\nThe previous storyboard failed story-fidelity checks. "
            "Fix ALL of these without inventing new content:\n" + failures
        )
        try:
            storyboard = await script_pipe.design_storyboard(
                script=scene_script,
                characters=characters,
                user_requirement=retry_req,
                quiet=quiet,
                grounded=True,
                force=True,
            )
        except Exception as exc:
            _emit_progress(progress, "canon_gate_retry_failed", f"Re-design failed: {exc}")
            break
        report = check_storyboard(storyboard, scene_beats, characters, scene_idx=scene_idx)

    if not report.passed:
        storyboard = repair_storyboard(storyboard, scene_beats, report, characters)
        report = check_storyboard(storyboard, scene_beats, characters, scene_idx=scene_idx)
        _emit_progress(
            progress,
            "canon_gate_repaired",
            f"Canon gate scene {scene_idx}: deterministic repair -> passed={report.passed}",
            {"passed": report.passed, "issues": len(report.issues)},
        )

    try:
        (scene_dir / "canon_report.json").write_text(
            report.model_dump_json(indent=2), encoding="utf-8"
        )
    except OSError:
        pass
    return storyboard


class Manga2VideoPipeline:
    """Canon-faithful Manga / Manhwa / Webtoon to Hybrid Anime Video Pipeline."""

    def __init__(
        self,
        chat_model: Any,
        image_generator: Any,
        video_generator: Any,
        working_dir: str | Path,
        num_image_candidates: int = DEFAULT_IMAGE_CANDIDATES,
    ) -> None:
        self.chat_model = chat_model
        self.image_generator = image_generator
        self.video_generator = video_generator
        self.working_dir = str(Path(working_dir).resolve())
        os.makedirs(self.working_dir, exist_ok=True)
        self.num_image_candidates = validate_image_candidate_count(num_image_candidates)
        self.manga_reader = MangaReader(chat_model=self.chat_model)

    async def plan_episode(
        self,
        *,
        sources: str | Path | Sequence[str | Path] | None = None,
        chapter_range: str | int | Sequence[int] | None = None,
        character_refs: Optional[Dict[str, Any]] = None,
        upload_image_paths: Optional[Sequence[str | Path]] = None,
        style: str = "2D modern Japanese anime film style, crisp cel shading, expressive eyes, atmospheric lighting",
        user_requirement: str = "",
        episode_title: str = "",
        batch_size: int = 3,
        force_replan: bool = False,
        scene_range: str | int | Sequence[int] | None = None,
        webtoon_fetcher=None,
        allow_private_urls: bool = False,
        manifest: Optional[ComicManifest] = None,
        series_characters: Optional[Sequence[CharacterInScene]] = None,
        series_seed_refs: Optional[Dict[str, List[str]]] = None,
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
        quiet: bool = False,
    ) -> MangaEpisodePlan:
        work_root = Path(self.working_dir)
        work_root.mkdir(parents=True, exist_ok=True)
        manifest_path = work_root / "comic_manifest.json"
        chunks_dir = work_root / "chunks"

        if force_replan:
            for dname in ("chunks", "panel_anchors", "character_seeds"):
                target_d = work_root / dname
                if target_d.exists():
                    shutil.rmtree(target_d, ignore_errors=True)
            for stale_scene in work_root.glob("scene_*"):
                if stale_scene.is_dir():
                    shutil.rmtree(stale_scene, ignore_errors=True)
            for fname in ("characters.json", "character_seed_refs.json", "episode_plan.json", "script.json"):
                (work_root / fname).unlink(missing_ok=True)

        # 1. Ingest manga sources or load valid cached ComicManifest
        if manifest is not None:
            manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        else:
            if not force_replan and manifest_path.is_file():
                try:
                    cached_manifest = ComicManifest.model_validate(
                        json.loads(manifest_path.read_text(encoding="utf-8"))
                    )
                    pages_ok = (
                        cached_manifest.chapters
                        and all(Path(p.path).is_file() for ch in cached_manifest.chapters for p in ch.pages)
                    )
                    if pages_ok:
                        if sources is None:
                            manifest = cached_manifest
                        else:
                            meta_path = work_root / "source_meta.json"
                            if meta_path.is_file():
                                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                                sig = json.dumps({"sources": sources, "chapter_range": chapter_range}, default=str, sort_keys=True)
                                if meta.get("signature") == sig:
                                    manifest = cached_manifest
                except (OSError, ValueError):
                    manifest = None

            if manifest is None:
                if sources is None:
                    raise ValueError("No comic sources provided and no valid comic_manifest.json cached.")
                _emit_progress(progress, "manga_ingest_start", "Ingesting and normalizing manga/webtoon pages")
                manifest = await ingest_manga_sources(
                    sources=sources,
                    working_dir=work_root,
                    chapter_range=chapter_range,
                    webtoon_fetcher=webtoon_fetcher,
                    allow_private_urls=allow_private_urls,
                )
                sig = json.dumps({"sources": sources, "chapter_range": chapter_range}, default=str, sort_keys=True)
                (work_root / "source_meta.json").write_text(
                    json.dumps({"signature": sig}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                _emit_progress(
                    progress,
                    "manga_ingest_done",
                    f"Ingested {len(manifest.chapters)} chapter(s), {manifest.total_pages} pages",
                    {"chapters": [c.chapter for c in manifest.chapters], "total_pages": manifest.total_pages},
                )

        # 2. Vision reading per chapter (batch size ~3 pages with tail continuity & per-chunk cache)
        all_chunks: List[MangaChunkExtraction] = []
        for ch_pages in manifest.chapters:
            ch_chunks = await self.manga_reader.read_chapter_pages(
                ch_pages,
                chunks_dir,
                batch_size=batch_size,
                style_hint=style,
                progress=progress,
            )
            all_chunks.extend(ch_chunks)

        # 3. Consolidate character roster & scene structure (or load cached plan if present)
        plan_path = work_root / "episode_plan.json"
        chars_path = work_root / "characters.json"
        seeds_path = work_root / "character_seed_refs.json"

        episode_plan: Optional[MangaEpisodePlan] = None
        characters: List[CharacterInScene] = []
        character_seed_refs: Dict[str, List[str]] = {}
        beats_by_scene: Dict[int, List[MangaBeat]] = {}

        if not force_replan and plan_path.is_file() and chars_path.is_file():
            try:
                cached_plan = MangaEpisodePlan.model_validate(json.loads(plan_path.read_text(encoding="utf-8")))
                cached_chars = [
                    CharacterInScene.model_validate(c)
                    for c in json.loads(chars_path.read_text(encoding="utf-8"))
                ]
                cached_seeds = (
                    json.loads(seeds_path.read_text(encoding="utf-8"))
                    if seeds_path.is_file()
                    else cached_plan.character_seed_refs
                )
                scenes_ready = bool(cached_plan.scenes) and all(
                    (work_root / f"scene_{sc.scene_idx}" / "script.txt").is_file()
                    and (work_root / f"scene_{sc.scene_idx}" / "beat_map.json").is_file()
                    for sc in cached_plan.scenes
                )
                if scenes_ready and cached_chars:
                    episode_plan = cached_plan
                    characters = cached_chars
                    character_seed_refs = dict(cached_seeds or {})
                    for sc in cached_plan.scenes:
                        raw_beats = json.loads(
                            (work_root / f"scene_{sc.scene_idx}" / "beat_map.json").read_text(encoding="utf-8")
                        )
                        beats_by_scene[sc.scene_idx] = [MangaBeat.model_validate(b) for b in raw_beats]
            except (OSError, ValueError):
                episode_plan = None

        if episode_plan is None:
            _emit_progress(progress, "manga_roster_start", "Consolidating canon character forms & reference seeds")
            characters, character_seed_refs, alias_map = self.manga_reader.build_character_roster_and_seeds(
                manifest=manifest,
                chunks=all_chunks,
                working_dir=work_root,
                character_refs=character_refs,
                upload_image_paths=upload_image_paths,
                series_characters=series_characters,
                series_seed_refs=series_seed_refs,
            )
            _emit_progress(
                progress,
                "manga_episode_grouping",
                "Grouping canon beats into scenes & Vietsub scripts",
                {"character_count": len(characters)},
            )
            episode_plan, beats_by_scene = self.manga_reader.build_episode_scenes(
                manifest=manifest,
                chunks=all_chunks,
                characters=characters,
                character_seed_refs=character_seed_refs,
                alias_map=alias_map,
                working_dir=work_root,
                style=style,
                episode_title=episode_title,
            )

        # 3.45 Character profiles (feature-flagged): populate draft profiles from the
        # roster so later chapters/outfits/states are addressable. Fail-soft.
        try:
            from agent_runtime.config import quality_profiles as _quality_profiles
            if _quality_profiles(_resolve_workspace_root(str(work_root))):
                from utils.profile_store import ProfileStore
                from utils.profile_builder import build_draft_profiles
                from utils.series_bible import load_configured_series
                _store = ProfileStore(str(work_root))
                _bible = load_configured_series(_resolve_workspace_root(str(work_root)))
                build_draft_profiles(characters, character_seed_refs, beats_by_scene, _store, _bible)
                _emit_progress(progress, "profiles_built", f"Built draft character profiles ({len(_store.all())})")
        except Exception as _exc:  # never block a render
            _emit_progress(progress, "profiles_skip", f"Profile build skipped: {_exc}")

        # 3.5 Manga Vision Layer (feature-flagged): chapter-level world model
        from agent_runtime.config import manga_vision_enabled as _manga_vision_enabled
        from agent_runtime.config import quality_canon_gate as _quality_canon_gate
        from agent_runtime.config import quality_generic_linter as _quality_generic_linter

        _ws_root = _resolve_workspace_root(str(work_root))
        vision_enabled = _manga_vision_enabled(_ws_root)
        canon_gate_enabled = _quality_canon_gate(_ws_root)
        generic_linter_enabled = _quality_generic_linter(_ws_root)
        world_model = await _build_or_load_world_model(
            pipeline=self,
            manifest=manifest,
            all_chunks=_align_chunks_to_episode_scenes(all_chunks, beats_by_scene),
            characters=characters,
            character_seed_refs=character_seed_refs,
            work_root=work_root,
            progress=progress,
        )

        # 4. Design scene storyboards, bind & strip [RefBeat], decompose shots, construct camera tree
        extra_req = (
            f"{user_requirement.strip()}\n"
            "STRICT MANGA-TO-ANIME FIDELITY: Keep exact canon dialogue, preserve every Shot's "
            "[Thought]/[Inner Monologue]/[Speaker] tags and (Vietsub: ...) lines verbatim, stay <= 14 spoken words per shot, "
            "and maintain visual character & panel fidelity."
        ).strip()

        selected_scenes = parse_chapter_range(scene_range) if scene_range is not None else None

        for scene_meta in episode_plan.scenes:
            sc_idx = scene_meta.scene_idx
            if selected_scenes is not None and sc_idx not in selected_scenes:
                continue
            scene_dir = work_root / f"scene_{sc_idx}"
            if (scene_dir / "camera_tree.json").is_file() and (scene_dir / "storyboard.json").is_file() and not force_replan:
                continue
            scene_dir.mkdir(parents=True, exist_ok=True)
            scene_script = (scene_dir / "script.txt").read_text(encoding="utf-8")
            scene_beats = beats_by_scene.get(sc_idx, [])
            sc_prog = _scene_progress_scope(progress, sc_idx)

            script_pipe = Script2VideoPipeline(
                chat_model=self.chat_model,
                image_generator=self.image_generator,
                video_generator=self.video_generator,
                working_dir=str(scene_dir),
                num_image_candidates=self.num_image_candidates,
            )
            _write_chars_if_missing(scene_dir / "characters.json", characters)
            script_pipe.character_portrait_events = {c.idx: asyncio.Event() for c in characters}

            # 3.6 Manga Vision Layer (feature-flagged): cinematic expansion of the
            # in-memory scene script only — canon script.txt on disk is untouched.
            scene_script = await _augment_scene_script_with_vision_async(
                scene_idx=sc_idx,
                scene_dir=scene_dir,
                scene_beats=scene_beats,
                world_model=world_model,
                chat_model=self.chat_model,
                progress=sc_prog,
            )

            _emit_progress(sc_prog, "design_storyboard", f"Designing storyboard for scene {sc_idx}")
            sb_path = scene_dir / "storyboard.json"
            sb_existed_before = sb_path.is_file()
            storyboard = await script_pipe.design_storyboard(
                script=scene_script,
                characters=characters,
                user_requirement=extra_req,
                quiet=quiet,
                grounded=canon_gate_enabled,
            )
            if not sb_existed_before:
                if generic_linter_enabled:
                    storyboard = shot_linter.lint_and_auto_correct_storyboard(
                        storyboard, characters=characters
                    )
                if callable(getattr(self.chat_model, "ainvoke", None)):
                    pre_hints = shot_linter.audit_storyboard(storyboard, characters=characters) if generic_linter_enabled else []
                    _emit_progress(
                        sc_prog,
                        "storyboard_llm_review",
                        f"LLM director reviewing scene {sc_idx} storyboard ({len(pre_hints)} structural hint(s))",
                        {"hints": pre_hints},
                    )
                    storyboard = await StoryboardArtist(chat_model=self.chat_model).review_and_repair_storyboard(
                        storyboard=storyboard,
                        characters=characters,
                        script=scene_script,
                        grounded=canon_gate_enabled,
                    )

            # 3.65 Story-fidelity canon gate (feature-flagged): keep the storyboard
            # grounded in this scene's beats before [RefBeat] tags are stripped.
            if canon_gate_enabled and scene_beats and not sb_existed_before:
                storyboard = await _apply_canon_gate(
                    storyboard=storyboard,
                    scene_beats=scene_beats,
                    characters=characters,
                    scene_idx=sc_idx,
                    scene_dir=scene_dir,
                    script_pipe=script_pipe,
                    scene_script=scene_script,
                    extra_req=extra_req,
                    quiet=quiet,
                    progress=sc_prog,
                )

            panel_refs_path = scene_dir / "shot_panel_refs.json"
            has_refbeat_tags = any(
                bool(_clean_refbeat_tags(s.visual_desc)[1] or _clean_refbeat_tags(s.audio_desc)[1])
                for s in storyboard
            )
            if panel_refs_path.is_file() and not has_refbeat_tags:
                clean_storyboard = storyboard
            else:
                clean_storyboard, _ = bind_storyboard_to_panel_anchors(storyboard, scene_beats, scene_dir)
            if generic_linter_enabled:
                clean_storyboard = [
                    sb.model_copy(
                        update={"visual_desc": shot_linter.strip_comic_artifacts(sb.visual_desc or "")}
                    )
                    for sb in clean_storyboard
                ]
            sb_path.write_text(
                json.dumps([s.model_dump() for s in clean_storyboard], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            # 3.7 Manga Vision Layer: per-shot provenance (CANON/INFERRED/CREATIVE
            # + source beat ids) for traceability and later visual QA.
            if vision_enabled:
                _write_scene_shot_provenance(
                    scene_idx=sc_idx,
                    scene_dir=scene_dir,
                    clean_storyboard=clean_storyboard,
                    scene_beats=scene_beats,
                    expansion_plan_json_path=scene_dir / "expansion_plan.json",
                )
            script_pipe.shot_desc_events = {sb.idx: asyncio.Event() for sb in clean_storyboard}

            shots_existed_before = bool(clean_storyboard) and all(
                (scene_dir / "shots" / f"{sb.idx}" / "shot_description.json").is_file()
                for sb in clean_storyboard
            )
            _emit_progress(sc_prog, "decompose_shots", f"Decomposing shot visuals for scene {sc_idx}", {"shot_count": len(clean_storyboard)})
            shot_descriptions = await script_pipe.decompose_visual_descriptions(
                shot_brief_descriptions=clean_storyboard,
                characters=characters,
                quiet=quiet,
            )
            # Guard first, then let LLM director review & finalize before writing to disk
            pre_cleaned_descs: List[ShotDescription] = []
            for sd in shot_descriptions:
                new_vis, t1 = _clean_refbeat_tags(sd.visual_desc)
                new_aud, t2 = _clean_refbeat_tags(sd.audio_desc)
                if t1 or t2:
                    sd = sd.model_copy(update={"visual_desc": new_vis, "audio_desc": new_aud})
                if generic_linter_enabled:
                    sd = shot_linter.lint_and_auto_correct_shot_description(sd, characters=characters)
                pre_cleaned_descs.append(sd)

            if not shots_existed_before and callable(getattr(self.chat_model, "ainvoke", None)):
                _emit_progress(
                    sc_prog,
                    "shot_description_llm_review",
                    f"LLM director reviewing decomposed keyframes for scene {sc_idx}",
                    {"shot_count": len(pre_cleaned_descs)},
                )
                pre_cleaned_descs = await StoryboardArtist(chat_model=self.chat_model).review_and_repair_shot_descriptions(
                    shot_descriptions=pre_cleaned_descs,
                    characters=characters,
                )

            cleaned_descs: List[ShotDescription] = []
            for sd in pre_cleaned_descs:
                if generic_linter_enabled:
                    sd = sd.model_copy(
                        update={
                            "visual_desc": shot_linter.strip_comic_artifacts(sd.visual_desc or ""),
                            "variation_reason": shot_linter.strip_comic_artifacts(getattr(sd, "variation_reason", "") or ""),
                            "ff_desc": shot_linter.strip_comic_artifacts(sd.ff_desc or ""),
                            "lf_desc": shot_linter.strip_comic_artifacts(sd.lf_desc or ""),
                            "motion_desc": shot_linter.sanitize_motion_prompt(
                                shot_linter.strip_comic_artifacts(sd.motion_desc or "")
                            ),
                        }
                    )
                shot_file = scene_dir / "shots" / f"{sd.idx}" / "shot_description.json"
                if shot_file.is_file():
                    shot_file.write_text(json.dumps(sd.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
                cleaned_descs.append(sd)

            camera_tree: Optional[List[Camera]] = None
            for attempt in range(2):
                try:
                    _emit_progress(
                        sc_prog,
                        "construct_camera_tree" if attempt == 0 else "construct_camera_tree_retry",
                        f"Constructing camera tree for scene {sc_idx}",
                        {"shot_count": len(cleaned_descs), "attempt": attempt + 1},
                    )
                    camera_tree = await script_pipe.construct_camera_tree(
                        shot_descriptions=cleaned_descs,
                        quiet=quiet,
                    )
                    break
                except Exception:
                    cam_path = scene_dir / "camera_tree.json"
                    if cam_path.exists():
                        cam_path.unlink(missing_ok=True)
                    if attempt == 1:
                        raise
            assert camera_tree is not None

        _emit_progress(
            progress,
            "manga_plan_done",
            f"Manga episode planned: {len(episode_plan.scenes)} scene(s), {episode_plan.total_beats} beats",
            {"scene_count": len(episode_plan.scenes), "total_beats": episode_plan.total_beats},
        )
        return episode_plan

    async def render_episode(
        self,
        *,
        style: str = "",
        user_requirement: str = "",
        stop_after: Literal["portraits", "frames", "full"] = "full",
        character_portraits_registry: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None,
        scene_range: str | int | Sequence[int] | None = None,
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
        quiet: bool = False,
    ) -> Any:
        if stop_after not in {"portraits", "frames", "full"}:
            raise ValueError(f"Unsupported stop_after value: {stop_after!r}")

        work_root = Path(self.working_dir)
        plan = MangaEpisodePlan.model_validate(json.loads((work_root / "episode_plan.json").read_text(encoding="utf-8")))
        characters = [
            CharacterInScene.model_validate(c)
            for c in json.loads((work_root / "characters.json").read_text(encoding="utf-8"))
        ]
        seeds_path = work_root / "character_seed_refs.json"
        character_seed_refs = (
            json.loads(seeds_path.read_text(encoding="utf-8"))
            if seeds_path.is_file()
            else plan.character_seed_refs
        )

        effective_style = style.strip() or plan.style or "2D modern Japanese anime film style"

        selected_scenes = parse_chapter_range(scene_range) if scene_range is not None else None

        # 1. Generate or reuse canonical anime character turnarounds.
        # With a scene subset, only generate turnarounds for characters that
        # actually appear in the selected scenes (lazy per-scene portraits;
        # full renders keep generating the whole roster).
        portrait_characters = characters
        if selected_scenes is not None:
            needed: set[str] = set()
            for scene_meta in plan.scenes:
                if scene_meta.scene_idx not in selected_scenes:
                    continue
                bmap = work_root / f"scene_{scene_meta.scene_idx}" / "beat_map.json"
                if not bmap.is_file():
                    continue
                try:
                    for b in json.loads(bmap.read_text(encoding="utf-8")):
                        needed.update(str(n) for n in b.get("characters_present", []) if n)
                except (OSError, ValueError):
                    continue
            if needed:
                portrait_characters = [c for c in characters if c.identifier_in_scene in needed]

        if character_portraits_registry is None:
            portrait_pipe = Script2VideoPipeline(
                chat_model=self.chat_model,
                image_generator=self.image_generator,
                video_generator=self.video_generator,
                working_dir=str(work_root),
                num_image_candidates=self.num_image_candidates,
            )
            character_portraits_registry = await portrait_pipe(
                script="",
                user_requirement=user_requirement,
                style=effective_style,
                characters=portrait_characters,
                quiet=quiet,
                progress=progress,
                character_ref_paths=character_seed_refs,
                stop_after="portraits",
            )
        else:
            missing_visible = [
                c
                for c in portrait_characters
                if getattr(c, "is_visible", True)
                and c.identifier_in_scene not in character_portraits_registry
            ]
            if missing_visible:
                portrait_pipe = Script2VideoPipeline(
                    chat_model=self.chat_model,
                    image_generator=self.image_generator,
                    video_generator=self.video_generator,
                    working_dir=str(work_root),
                    num_image_candidates=self.num_image_candidates,
                )
                character_portraits_registry = await portrait_pipe.generate_character_portraits(
                    characters=missing_visible,
                    character_portraits_registry=dict(character_portraits_registry),
                    style=effective_style,
                    progress=progress,
                    character_ref_paths=character_seed_refs,
                )
        if stop_after == "portraits":
            return character_portraits_registry

        # 2. Render each scene sharing canonical character portraits and scene-local shot_panel_refs.json
        all_frames: List[str] = []
        scene_videos: List[str] = []
        scene_dirs: List[Path] = []

        for scene_meta in plan.scenes:
            sc_idx = scene_meta.scene_idx
            if selected_scenes is not None and sc_idx not in selected_scenes:
                continue
            scene_dir = work_root / f"scene_{sc_idx}"
            scene_dirs.append(scene_dir)
            scene_script = (scene_dir / "script.txt").read_text(encoding="utf-8")

            from agent_runtime.config import (
                manga_vision_auto_rerender_critical,
                manga_vision_enabled,
                manga_vision_max_rerender_cycles,
            )

            workspace_root = _resolve_workspace_root(str(work_root))
            auto_rerender = (
                manga_vision_enabled(workspace_root)
                and manga_vision_auto_rerender_critical(workspace_root)
                and stop_after in {"frames", "full"}
            )
            max_cycles = manga_vision_max_rerender_cycles(workspace_root) if auto_rerender else 1

            result = None
            report = None
            for cycle in range(max_cycles):
                sc_prog = _scene_progress_scope(progress, sc_idx)
                if cycle > 0:
                    _emit_progress(
                        progress,
                        "visual_qa_rerender",
                        f"Re-rendering failed shots for scene {sc_idx} (cycle {cycle + 1}/{max_cycles})",
                        {"scene_idx": sc_idx, "cycle": cycle + 1},
                    )
                scene_pipe = Script2VideoPipeline(
                    chat_model=self.chat_model,
                    image_generator=self.image_generator,
                    video_generator=self.video_generator,
                    working_dir=str(scene_dir),
                    num_image_candidates=self.num_image_candidates,
                )
                result = await scene_pipe(
                    script=scene_script,
                    user_requirement=user_requirement,
                    style=effective_style,
                    characters=characters,
                    character_portraits_registry=character_portraits_registry,
                    quiet=quiet,
                    progress=sc_prog,
                    stop_after=stop_after,
                )
                if stop_after in {"frames", "full"}:
                    report = await _run_scene_visual_qa(
                        pipeline=self,
                        scene_idx=sc_idx,
                        scene_dir=scene_dir,
                        progress=sc_prog,
                    )
                failed = list(report.shots_with_critical_issues()) if report is not None else []
                if not (auto_rerender and cycle + 1 < max_cycles and failed):
                    break
                # Delete the failed shots' frames so the next cycle regenerates them.
                for fidx in failed:
                    for name in ("first_frame.png", "last_frame.png"):
                        fp = scene_dir / "shots" / str(fidx) / name
                        try:
                            fp.unlink(missing_ok=True)
                        except OSError:
                            pass
                _emit_progress(
                    progress,
                    "visual_qa_rerender_shots",
                    f"Re-rendering shots {failed} for scene {sc_idx}",
                    {"scene_idx": sc_idx, "shots": failed},
                )

            if stop_after == "frames":
                all_frames.extend(result or [])
            else:
                scene_videos.append(str(result))

        if stop_after == "frames":
            return all_frames

        if selected_scenes is not None and len(selected_scenes) < len(plan.scenes):
            return scene_videos[0] if len(scene_videos) == 1 else scene_videos

        # 3. Concatenate all scene videos cleanly + write combined episode .srt/.ass sidecars
        final_video_path = work_root / "final_video.mp4"
        if not final_video_path.is_file():
            if not scene_videos:
                raise RuntimeError("Manga2VideoPipeline rendered 0 scene videos.")
            _emit_progress(progress, "episode_concat_start", f"Concatenating {len(scene_videos)} scene video(s)")
            if len(scene_videos) == 1:
                tmp_copy = work_root / f".final_episode.{os.getpid()}.tmp.mp4"
                shutil.copyfile(scene_videos[0], tmp_copy)
                os.replace(tmp_copy, final_video_path)
            else:
                concatenate_video_files(
                    scene_videos,
                    str(final_video_path),
                    scene_fade_sec=0.35,
                )
            _emit_progress(progress, "episode_concat_done", f"Episode final video saved to {final_video_path}", {"final_video_path": str(final_video_path)})

        # Synchronize combined episode subtitle sidecar files without double-burning scenes
        from agent_runtime.config import subtitles_enabled, subtitles_font, subtitles_font_size, subtitles_language

        ws_root = _resolve_workspace_root(work_root)
        if subtitles_enabled(ws_root):
            sub_lang = subtitles_language(ws_root).strip().lower()
            suffix = "vietsub" if sub_lang in {"vi", "vn", "vietnamese"} else sub_lang
            durations: List[float] = []
            for sv in scene_videos:
                clip = None
                try:
                    clip = VideoFileClip(sv)
                    durations.append(float(clip.duration or 0.0))
                except Exception:
                    durations.append(0.0)
                finally:
                    if clip is not None:
                        clip.close()
            _merge_scene_subtitles(
                scene_dirs,
                durations,
                work_root,
                suffix=suffix,
                font=subtitles_font(ws_root),
                font_size=subtitles_font_size(ws_root),
            )

        return str(final_video_path)

    async def plan_series(
        self,
        *,
        sources: str | Path | Sequence[str | Path] | None = None,
        chapter_range: str | int | Sequence[int] | None = None,
        chapters_per_episode: int = 1,
        episode_range: str | int | Sequence[int] | None = None,
        character_refs: Optional[Dict[str, Any]] = None,
        upload_image_paths: Optional[Sequence[str | Path]] = None,
        style: str = "2D modern Japanese anime film style, crisp cel shading, expressive eyes, atmospheric lighting",
        user_requirement: str = "",
        series_title: str = "",
        batch_size: int = 3,
        force_replan: bool = False,
        webtoon_fetcher=None,
        allow_private_urls: bool = False,
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
        quiet: bool = False,
    ) -> MangaSeriesPlan:
        per_ep = int(chapters_per_episode)
        if per_ep < 1:
            raise ValueError(f"chapters_per_episode must be >= 1 for series planning, got {chapters_per_episode!r}")

        work_root = Path(self.working_dir)
        work_root.mkdir(parents=True, exist_ok=True)
        manifest_path = work_root / "comic_manifest.json"
        series_manifest_path = work_root / "series_manifest.json"
        series_chars_path = work_root / "series_characters.json"
        series_seeds_path = work_root / "series_character_seed_refs.json"
        root_chars_path = work_root / "characters.json"
        portraits_registry_path = work_root / "character_portraits_registry.json"

        prior_entries_by_idx: Dict[int, MangaSeriesEpisodeEntry] = {}
        if force_replan:
            episodes_root = work_root / "episodes"
            if episodes_root.exists():
                shutil.rmtree(episodes_root, ignore_errors=True)
            for fname in (
                "series_manifest.json",
                "series_state.json",
                "series_characters.json",
                "series_character_seed_refs.json",
                "characters.json",
            ):
                (work_root / fname).unlink(missing_ok=True)
        elif series_manifest_path.is_file():
            try:
                prior_series = MangaSeriesPlan.model_validate(json.loads(series_manifest_path.read_text(encoding="utf-8")))
                prior_entries_by_idx = {e.episode_idx: e for e in prior_series.episodes}
            except (OSError, ValueError):
                prior_entries_by_idx = {}

        # 1. Ingest full comic series once or reuse valid cached ComicManifest
        manifest: Optional[ComicManifest] = None
        if not force_replan and manifest_path.is_file():
            try:
                cached_manifest = ComicManifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
                pages_ok = (
                    cached_manifest.chapters
                    and all(Path(p.path).is_file() for ch in cached_manifest.chapters for p in ch.pages)
                )
                if pages_ok:
                    if sources is None:
                        manifest = cached_manifest
                    else:
                        meta_path = work_root / "source_meta.json"
                        if meta_path.is_file():
                            meta = json.loads(meta_path.read_text(encoding="utf-8"))
                            sig = json.dumps({"sources": sources, "chapter_range": chapter_range}, default=str, sort_keys=True)
                            if meta.get("signature") == sig:
                                manifest = cached_manifest
            except (OSError, ValueError):
                manifest = None

        if manifest is None:
            if sources is None:
                raise ValueError("No comic sources provided and no valid comic_manifest.json cached.")
            _emit_progress(progress, "manga_series_ingest_start", "Ingesting full comic series")
            manifest = await ingest_manga_sources(
                sources=sources,
                working_dir=work_root,
                chapter_range=chapter_range,
                webtoon_fetcher=webtoon_fetcher,
                allow_private_urls=allow_private_urls,
            )
            sig = json.dumps({"sources": sources, "chapter_range": chapter_range}, default=str, sort_keys=True)
            (work_root / "source_meta.json").write_text(
                json.dumps({"signature": sig}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            _emit_progress(
                progress,
                "manga_series_ingest_done",
                f"Ingested series: {len(manifest.chapters)} chapter(s), {manifest.total_pages} pages",
                {"chapters": [c.chapter for c in manifest.chapters], "total_pages": manifest.total_pages},
            )

        episode_manifests = group_chapters_for_episodes(manifest, chapters_per_episode=per_ep)
        if not episode_manifests:
            raise ValueError("No chapters available to partition into series episodes.")

        selected_episodes = parse_chapter_range(episode_range)
        series_characters: List[CharacterInScene] = []
        if not force_replan and series_chars_path.is_file():
            try:
                series_characters = [
                    CharacterInScene.model_validate(item)
                    for item in json.loads(series_chars_path.read_text(encoding="utf-8"))
                ]
            except (OSError, ValueError):
                series_characters = []
        series_seed_refs: Dict[str, List[str]] = {}
        if not force_replan and series_seeds_path.is_file():
            try:
                raw_seeds = json.loads(series_seeds_path.read_text(encoding="utf-8"))
                if isinstance(raw_seeds, dict):
                    series_seed_refs = {str(k): [str(p) for p in (v or [])] for k, v in raw_seeds.items()}
            except (OSError, ValueError):
                series_seed_refs = {}

        def _merge_series_bank(ep_chars: Sequence[CharacterInScene], ep_seeds: Dict[str, List[str]]) -> None:
            known_names = {c.identifier_in_scene: idx for idx, c in enumerate(series_characters)}
            next_idx = max((c.idx for c in series_characters), default=-1) + 1
            for ch in ep_chars:
                name = ch.identifier_in_scene
                if name not in known_names:
                    series_characters.append(
                        ch.model_copy(update={"idx": next_idx})
                    )
                    known_names[name] = next_idx
                    next_idx += 1
                seeds = [str(p) for p in (ep_seeds.get(name) or []) if str(p).strip() and Path(str(p)).is_file()]
                if seeds and name not in series_seed_refs:
                    series_seed_refs[name] = seeds[:4]
            chars_json = json.dumps([c.model_dump() for c in series_characters], ensure_ascii=False, indent=2)
            series_chars_path.write_text(chars_json, encoding="utf-8")
            root_chars_path.write_text(chars_json, encoding="utf-8")
            series_seeds_path.write_text(
                json.dumps(series_seed_refs, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        base_title = series_title.strip() or (manifest.chapters[0].title if manifest.chapters else "Manga Series")
        episode_entries: List[MangaSeriesEpisodeEntry] = []

        def _persist_series_plan() -> MangaSeriesPlan:
            completed = sum(1 for e in episode_entries if e.status == "rendered" and e.final_video_path and Path(e.final_video_path).is_file())
            plan_obj = MangaSeriesPlan(
                series_title=base_title,
                style=style,
                total_chapters=len(manifest.chapters),
                chapters_per_episode=per_ep,
                episodes=episode_entries,
                series_characters_path=str(series_chars_path),
                series_portraits_registry_path=str(portraits_registry_path),
                completed_episodes=completed,
            )
            series_manifest_path.write_text(plan_obj.model_dump_json(indent=2), encoding="utf-8")
            return plan_obj

        for ep_idx, ep_manifest in enumerate(episode_manifests, start=1):
            if selected_episodes is not None and ep_idx not in selected_episodes:
                if ep_idx in prior_entries_by_idx:
                    episode_entries.append(prior_entries_by_idx[ep_idx])
                continue
            ep_dir = work_root / "episodes" / f"ep_{ep_idx:03d}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            ep_chapters = [ch.chapter for ch in ep_manifest.chapters]
            ch_label = (
                f"Ch.{ep_chapters[0]}"
                if len(ep_chapters) == 1
                else f"Ch.{ep_chapters[0]}-{ep_chapters[-1]}"
            )
            ep_title = f"{base_title} - Ep.{ep_idx} ({ch_label})"
            _emit_progress(
                progress,
                "manga_series_plan_episode",
                f"Planning episode {ep_idx}/{len(episode_manifests)} ({ch_label})",
                {"episode_idx": ep_idx, "chapters": ep_chapters, "total_episodes": len(episode_manifests)},
            )
            ep_pipe = Manga2VideoPipeline(
                chat_model=self.chat_model,
                image_generator=self.image_generator,
                video_generator=self.video_generator,
                working_dir=ep_dir,
                num_image_candidates=self.num_image_candidates,
            )
            ep_plan = await ep_pipe.plan_episode(
                manifest=ep_manifest,
                character_refs=character_refs,
                upload_image_paths=upload_image_paths,
                style=style,
                user_requirement=user_requirement,
                episode_title=ep_title,
                batch_size=batch_size,
                force_replan=force_replan,
                series_characters=series_characters or None,
                series_seed_refs=series_seed_refs or None,
                progress=progress,
                quiet=quiet,
            )
            _merge_series_bank(ep_plan.characters, ep_plan.character_seed_refs)

            prev_entry = prior_entries_by_idx.get(ep_idx)
            final_vid = str(ep_dir / "final_video.mp4") if (ep_dir / "final_video.mp4").is_file() else None
            sub_paths = sorted(
                str(p)
                for pat in ("final_video_*.srt", "final_video_*.ass")
                for p in ep_dir.glob(pat)
                if p.is_file()
            )
            status: Literal["planned", "portraits", "frames", "rendered", "error"] = "planned"
            if final_vid and not force_replan:
                status = "rendered"
            elif prev_entry is not None and prev_entry.status in {"portraits", "frames"} and not force_replan:
                status = prev_entry.status

            episode_entries.append(
                MangaSeriesEpisodeEntry(
                    episode_idx=ep_idx,
                    episode_title=ep_plan.episode_title or ep_title,
                    chapters=ep_chapters,
                    working_dir=str(ep_dir),
                    scene_count=len(ep_plan.scenes),
                    total_beats=ep_plan.total_beats,
                    character_names=[c.identifier_in_scene for c in ep_plan.characters if getattr(c, "is_visible", True)],
                    status=status,
                    final_video_path=final_vid,
                    subtitle_paths=sub_paths,
                    error=None,
                )
            )
            _persist_series_plan()

        series_plan = _persist_series_plan()
        _emit_progress(
            progress,
            "manga_series_plan_done",
            f"Series planned: {len(series_plan.episodes)} episode(s), {series_plan.total_chapters} chapter(s)",
            {"episode_count": len(series_plan.episodes), "total_chapters": series_plan.total_chapters},
        )
        return series_plan

    async def render_series(
        self,
        *,
        style: str = "",
        user_requirement: str = "",
        stop_after: Literal["portraits", "frames", "full"] = "full",
        episode_range: str | int | Sequence[int] | None = None,
        scene_range: str | int | Sequence[int] | None = None,
        resume: bool = True,
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
        quiet: bool = False,
    ) -> Any:
        if stop_after not in {"portraits", "frames", "full"}:
            raise ValueError(f"Unsupported stop_after value: {stop_after!r}")

        work_root = Path(self.working_dir)
        series_manifest_path = work_root / "series_manifest.json"
        series_state_path = work_root / "series_state.json"
        series_plan = MangaSeriesPlan.model_validate(json.loads(series_manifest_path.read_text(encoding="utf-8")))

        series_chars_path = Path(series_plan.series_characters_path or (work_root / "series_characters.json"))
        series_characters: List[CharacterInScene] = []
        if series_chars_path.is_file():
            series_characters = [
                CharacterInScene.model_validate(c)
                for c in json.loads(series_chars_path.read_text(encoding="utf-8"))
            ]
        series_char_by_name = {c.identifier_in_scene: c for c in series_characters}

        series_seeds_path = work_root / "series_character_seed_refs.json"
        series_seed_refs: Dict[str, List[str]] = (
            json.loads(series_seeds_path.read_text(encoding="utf-8"))
            if series_seeds_path.is_file()
            else {}
        )

        portraits_registry_path = Path(
            series_plan.series_portraits_registry_path or (work_root / "character_portraits_registry.json")
        )
        shared_registry: Dict[str, Dict[str, Dict[str, str]]] = {}
        if portraits_registry_path.is_file():
            try:
                raw_reg = json.loads(portraits_registry_path.read_text(encoding="utf-8"))
                shared_registry = _sanitize_portraits_registry(raw_reg)
            except (OSError, ValueError):
                shared_registry = {}

        effective_style = style.strip() or series_plan.style or "2D modern Japanese anime film style"
        selected_episodes = parse_chapter_range(episode_range)

        portrait_pipe = Script2VideoPipeline(
            chat_model=self.chat_model,
            image_generator=self.image_generator,
            video_generator=self.video_generator,
            working_dir=str(work_root),
            num_image_candidates=self.num_image_candidates,
        )

        def _save_series_state() -> None:
            series_plan.completed_episodes = sum(
                1
                for ep in series_plan.episodes
                if ep.status == "rendered" and ep.final_video_path and Path(ep.final_video_path).is_file()
            )
            series_manifest_path.write_text(series_plan.model_dump_json(indent=2), encoding="utf-8")
            series_state_path.write_text(
                json.dumps(
                    {
                        "series_title": series_plan.series_title,
                        "total_episodes": len(series_plan.episodes),
                        "completed_episodes": series_plan.completed_episodes,
                        "episodes": [
                            {
                                "episode_idx": e.episode_idx,
                                "chapters": e.chapters,
                                "status": e.status,
                                "final_video_path": e.final_video_path,
                                "error": e.error,
                            }
                            for e in series_plan.episodes
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        all_frames: List[str] = []
        for entry in series_plan.episodes:
            if selected_episodes is not None and entry.episode_idx not in selected_episodes:
                continue
            ep_dir = Path(entry.working_dir)
            if (
                stop_after == "full"
                and resume
                and entry.status == "rendered"
                and entry.final_video_path
                and Path(entry.final_video_path).is_file()
            ):
                _emit_progress(
                    progress,
                    "manga_series_episode_skipped",
                    f"Skipping already rendered episode {entry.episode_idx}",
                    {"episode_idx": entry.episode_idx, "final_video_path": entry.final_video_path},
                )
                continue

            ep_chars_file = ep_dir / "characters.json"
            ep_chars = [
                CharacterInScene.model_validate(c)
                for c in json.loads(ep_chars_file.read_text(encoding="utf-8"))
            ]
            ep_seeds_file = ep_dir / "character_seed_refs.json"
            ep_seeds = (
                json.loads(ep_seeds_file.read_text(encoding="utf-8"))
                if ep_seeds_file.is_file()
                else {}
            )
            merged_seeds = dict(series_seed_refs)
            for k, v in (ep_seeds or {}).items():
                if v:
                    merged_seeds[str(k)] = list(v)

            # Generate only missing turnaround portraits into the shared series portrait bank
            ep_series_chars = [
                series_char_by_name.get(c.identifier_in_scene, c)
                for c in ep_chars
                if getattr(c, "is_visible", True)
            ]
            missing_chars = [
                c for c in ep_series_chars if c.identifier_in_scene not in shared_registry
            ]
            if missing_chars:
                _emit_progress(
                    progress,
                    "manga_series_portraits_update",
                    f"Generating {len(missing_chars)} new series character turnaround(s) for Ep.{entry.episode_idx}",
                    {"episode_idx": entry.episode_idx, "new_characters": [c.identifier_in_scene for c in missing_chars]},
                )
                shared_registry = await portrait_pipe.generate_character_portraits(
                    characters=missing_chars,
                    character_portraits_registry=shared_registry,
                    style=effective_style,
                    progress=progress,
                    character_ref_paths=merged_seeds,
                )

            if stop_after == "portraits":
                entry.status = "portraits"
                entry.error = None
                _save_series_state()
                continue

            try:
                _emit_progress(
                    progress,
                    "manga_series_render_episode_start",
                    f"Rendering episode {entry.episode_idx}/{len(series_plan.episodes)} ({entry.episode_title})",
                    {"episode_idx": entry.episode_idx, "chapters": entry.chapters, "stop_after": stop_after},
                )
                ep_pipe = Manga2VideoPipeline(
                    chat_model=self.chat_model,
                    image_generator=self.image_generator,
                    video_generator=self.video_generator,
                    working_dir=ep_dir,
                    num_image_candidates=self.num_image_candidates,
                )
                ep_out = await ep_pipe.render_episode(
                    style=effective_style,
                    user_requirement=user_requirement,
                    stop_after=stop_after,
                    character_portraits_registry=shared_registry,
                    scene_range=scene_range,
                    progress=progress,
                    quiet=quiet,
                )
                if stop_after == "frames":
                    all_frames.extend(ep_out or [])
                    entry.status = "frames"
                else:
                    entry.status = "rendered"
                    entry.final_video_path = str(ep_out)
                    entry.subtitle_paths = sorted(
                        str(p)
                        for pat in ("final_video_*.srt", "final_video_*.ass")
                        for p in ep_dir.glob(pat)
                        if p.is_file()
                    )
                entry.error = None
                _save_series_state()
            except Exception as exc:
                entry.status = "error"
                entry.error = str(exc)
                _save_series_state()
                raise

        _save_series_state()
        if stop_after == "portraits":
            return shared_registry
        if stop_after == "frames":
            return all_frames
        return series_plan

    async def __call__(
        self,
        *,
        sources: str | Path | Sequence[str | Path] | None = None,
        chapter_range: str | int | Sequence[int] | None = None,
        chapters_per_episode: Optional[int] = None,
        episode_range: str | int | Sequence[int] | None = None,
        character_refs: Optional[Dict[str, Any]] = None,
        upload_image_paths: Optional[Sequence[str | Path]] = None,
        style: str = "2D modern Japanese anime film style, crisp cel shading, expressive eyes, atmospheric lighting",
        user_requirement: str = "",
        episode_title: str = "",
        stop_after: Literal["portraits", "frames", "full"] = "full",
        batch_size: int = 3,
        force_replan: bool = False,
        webtoon_fetcher=None,
        allow_private_urls: bool = False,
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
        quiet: bool = False,
    ) -> Any:
        if chapters_per_episode is not None and int(chapters_per_episode) >= 1:
            series_plan = await self.plan_series(
                sources=sources,
                chapter_range=chapter_range,
                chapters_per_episode=int(chapters_per_episode),
                episode_range=episode_range,
                character_refs=character_refs,
                upload_image_paths=upload_image_paths,
                style=style,
                user_requirement=user_requirement,
                series_title=episode_title,
                batch_size=batch_size,
                force_replan=force_replan,
                webtoon_fetcher=webtoon_fetcher,
                allow_private_urls=allow_private_urls,
                progress=progress,
                quiet=quiet,
            )
            return await self.render_series(
                style=series_plan.style or style,
                user_requirement=user_requirement,
                stop_after=stop_after,
                episode_range=episode_range,
                progress=progress,
                quiet=quiet,
            )

        plan = await self.plan_episode(
            sources=sources,
            chapter_range=chapter_range,
            character_refs=character_refs,
            upload_image_paths=upload_image_paths,
            style=style,
            user_requirement=user_requirement,
            episode_title=episode_title,
            batch_size=batch_size,
            force_replan=force_replan,
            webtoon_fetcher=webtoon_fetcher,
            allow_private_urls=allow_private_urls,
            progress=progress,
            quiet=quiet,
        )
        return await self.render_episode(
            style=plan.style or style,
            user_requirement=user_requirement,
            stop_after=stop_after,
            progress=progress,
            quiet=quiet,
        )


def _sanitize_portraits_registry(raw: Any) -> Dict[str, Dict[str, Dict[str, str]]]:
    if not isinstance(raw, dict):
        return {}
    clean: Dict[str, Dict[str, Dict[str, str]]] = {}
    for char_name, views in raw.items():
        if not isinstance(views, dict):
            continue
        front = views.get("front")
        if not isinstance(front, dict):
            continue
        front_path = str(front.get("path", "") or "").strip()
        if front_path and os.path.exists(front_path):
            clean[str(char_name)] = views
    return clean


def _write_chars_if_missing(path: Path, characters: List[CharacterInScene]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([c.model_dump() for c in characters], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
