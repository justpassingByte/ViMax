"""Visual QA reviewer for the Manga Vision Layer.

Compares generated keyframes against the original manga panel anchors and the
``VisionWorldModel`` before final rendering. MVP behavior is REPORT-ONLY:

- deterministic pre-checks (missing frames, missing anchors, character-count
  mismatches between shot description and world model)
- one batched VLM review per scene attaching the scene's keyframes plus their
  panel anchor references, scoring identity / outfit / environment /
  spatial / narrative consistency

Failed shots are recorded in ``scene_N/visual_qa_report.json`` so the render
loop (or a human) can selectively re-render single shots instead of regenerating
the whole chapter.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from tenacity import retry, stop_after_attempt, wait_exponential

from interfaces import (
    MangaBeat,
    ShotDescription,
    VisualQAIssue,
    VisualQAReport,
    VisionWorldModel,
)
from utils.image import image_path_to_b64
from utils.retry import after_func
from utils.robust_json_parser import strip_trailing_commas

logger = logging.getLogger(__name__)

_system_prompt_template_qa = """
[Role]
You are a strict visual continuity reviewer for an anime adaptation pipeline.

[Task]
You receive, for ONE scene: the storyboard shot descriptions, character bible
entries, environment bible entries, and pairs of (original manga panel crop,
generated anime keyframe). Review each generated keyframe against its panel
reference and the world model.

Check per keyframe:
- character_identity: does the depicted character match the bible's permanent
  features (hair/fur, eyes, body)? Flag identity drift.
- outfit_consistency: does clothing/state match the bible's temporary features
  and the panel?
- character_count: wrong number of characters, duplicated clones, invented
  bystanders.
- environment_consistency: does the background match the environment bible
  (architecture, props, lighting, time of day, weather)?
- spatial_relation: does the layout contradict the recorded spatial relations
  (left/right/behind/holding)?
- narrative_alignment: does the keyframe contradict what happens in the panel?
- artifact: speech bubbles, panel gutters, SFX text, borders, watermarks.

[Severity Rules]
- "critical": wrong character identity, invented character, contradicting plot,
  speech bubbles / comic borders left in frame.
- "warning": outfit drift, lighting mismatch, minor spatial issues.
- "info": polish suggestions.

[Output]
Return ONLY a JSON object:
{"issues": [
    {"shot_idx": <int>, "aspect": "<aspect>", "severity": "<severity>",
     "description": "<short>", "suggestion": "<short fix>"}
]}
Do not return any shots that are clean.
"""

_human_prompt_template_qa = """
CHARACTER BIBLE:
{characters_str}

ENVIRONMENT BIBLE:
{environments_str}

SHOT DESCRIPTIONS:
{shots_str}

KEYFRAMES WITH PANEL REFERENCES (for each pair: "Shot N panel ref" then
"Shot N generated keyframe"):
"""

_ALLOWED_ASPECTS = {
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
}
_ALLOWED_SEVERITIES = {"info", "warning", "critical"}


class VisualQAReviewer:
    def __init__(self, chat_model: Runnable) -> None:
        self.chat_model = chat_model

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=12),
        after=after_func,
        reraise=True,
    )
    async def _invoke_review(self, *, content: List[Dict[str, Any]]) -> Dict[str, Any]:
        messages = [SystemMessage(content=_system_prompt_template_qa), HumanMessage(content=content)]
        response = await self.chat_model.ainvoke(messages)
        text = response.content if hasattr(response, "content") else str(response)
        if isinstance(text, list):
            text = " ".join(str(part) for part in text)
        match = re.search(r"\{.*\}", str(text), re.DOTALL)
        if not match:
            raise ValueError("Visual QA reviewer returned no JSON object")
        return json.loads(strip_trailing_commas(match.group(0)))

    async def review_scene(
        self,
        *,
        scene_idx: int,
        scene_dir: str | Path,
        shot_descriptions: Sequence[ShotDescription],
        beats: Sequence[MangaBeat],
        world_model: Optional[VisionWorldModel],
        max_frame_pairs: int = 4,
        save: bool = True,
    ) -> VisualQAReport:
        scene_root = Path(scene_dir)
        report = VisualQAReport(scene_idx=scene_idx)

        # Chapter-level duplicate-identity suspicions (report-only): record as
        # info issues and hand them to the VLM so it watches for clone drift.
        suspected_dupes: List[Tuple[str, str, float]] = []
        if world_model is not None and world_model.identity_dedup is not None:
            dedup = world_model.identity_dedup
            for pair in dedup.suspected_pairs():
                suspected_dupes.append((pair.a, pair.b, pair.similarity))
            if dedup.status == "skipped" and dedup.reason:
                report.issues.append(
                    VisualQAIssue(
                        shot_idx=-1,
                        aspect="identity_dedup",
                        severity="info",
                        description=f"Duplicate-identity analysis skipped: {dedup.reason}",
                        suggestion="Configure a working embedding provider to enable it.",
                    )
                )
        for a, b, sim in suspected_dupes:
            report.issues.append(
                VisualQAIssue(
                    shot_idx=-1,
                    aspect="identity_dedup",
                    severity="info",
                    description=(
                        f"Suspected duplicate identity: '{a}' and '{b}' "
                        f"(similarity {sim:.2f} >= threshold {dedup.threshold:.2f})."
                    ),
                    suggestion=(
                        f"Confirm whether these are the same character; if so, merge the bible "
                        f"entries and reuse a single portrait for both names."
                    ),
                )
            )

        # Supplementary deterministic evidence: same-looking characters that
        # never appear together in any panel are ALSO flagged info-only, so a
        # human sees the full suspect list even when embeddings are unavailable.
        if world_model is not None and world_model.identity_dedup is not None:
            already = {(p.a, p.b) for p in world_model.identity_dedup.suspected_pairs()}
            for pair in world_model.identity_dedup.pairs:
                if pair.suspected or (pair.a, pair.b) in already or not pair.never_cooccur:
                    continue
                if pair.similarity < 0.3:
                    continue
                report.issues.append(
                    VisualQAIssue(
                        shot_idx=-1,
                        aspect="identity_dedup",
                        severity="info",
                        description=(
                            f"'{pair.a}' and '{pair.b}' look similar ({pair.similarity:.2f}) and NEVER "
                            f"appear together in any panel — possible same character, two names."
                        ),
                        suggestion="Review panel evidence; merge the bible entries if confirmed.",
                    )
                )

        anchor_by_id: Dict[str, str] = {b.beat_id: b.anchor_image_path for b in beats if b.anchor_image_path}
        all_beat_ids = {b.beat_id for b in beats}

        # Deterministic pre-checks + collect reviewable frame/panel pairs.
        pairs: List[Tuple[int, str, str]] = []  # (shot_idx, keyframe_path, panel_path)
        for sd in shot_descriptions:
            ff_path = scene_root / "shots" / str(sd.idx) / "first_frame.png"
            if not ff_path.is_file():
                report.issues.append(
                    VisualQAIssue(
                        shot_idx=sd.idx,
                        aspect="missing_frame",
                        severity="critical",
                        description="first_frame.png missing",
                        suggestion="Re-generate the keyframe for this shot.",
                    )
                )
                continue
            beat_ids = [
                bid
                for bid in _extract_beat_refs(f"{sd.visual_desc} {sd.audio_desc or ''}")
                if bid in all_beat_ids
            ]
            panel_path = ""
            for bid in beat_ids:
                anchor = anchor_by_id.get(bid)
                if anchor and Path(anchor).is_file():
                    panel_path = anchor
                    break
            if not panel_path:
                report.issues.append(
                    VisualQAIssue(
                        shot_idx=sd.idx,
                        aspect="missing_frame",
                        severity="info",
                        description="No panel anchor bound to this shot; QA compared against world model only.",
                        suggestion="Bind a source panel anchor for stronger QA.",
                    )
                )
            pairs.append((sd.idx, str(ff_path), panel_path))

        report.reviewed_frames = len(pairs)

        # VLM review (bounded: at most max_frame_pairs pairs per scene).
        if pairs:
            characters_str = "\n".join(
                f"- {e.character_key}: permanent={e.permanent_features or 'n/a'}; outfit/state={e.temporary_features or 'n/a'}"
                for e in (world_model.characters if world_model else [])
            ) or "(none)"
            if suspected_dupes:
                dupes_str = "; ".join(f"'{a}' vs '{b}' ({sim:.2f})" for a, b, sim in suspected_dupes)
                characters_str += (
                    "\nSUSPECTED DUPLICATE IDENTITIES (possibly the same person under two names; "
                    f"flag any keyframe where they appear as different people): {dupes_str}"
                )
            environments_str = "\n".join(
                f"- {e.environment_id} '{e.name}': {e.env_type or 'unknown'}; lighting={e.lighting or 'n/a'}; "
                f"time={e.time_of_day}; weather={e.weather}; props={', '.join(e.key_props[:5])}"
                for e in (world_model.environments if world_model else [])
            ) or "(none)"
            shots_str = "\n".join(
                f"Shot {sd.idx}: visual='{sd.visual_desc}' audio='{sd.audio_desc or ''}'"
                for sd in shot_descriptions
            )
            content: List[Dict[str, Any]] = [
                {
                    "type": "text",
                    "text": _human_prompt_template_qa.format(
                        characters_str=characters_str,
                        environments_str=environments_str,
                        shots_str=shots_str,
                    ),
                }
            ]
            for shot_idx, keyframe, panel in pairs[:max_frame_pairs]:
                if panel:
                    content.append({"type": "text", "text": f"Shot {shot_idx} panel ref:"})
                    content.append({"type": "image_url", "image_url": {"url": image_path_to_b64(panel, mime=True)}})
                content.append({"type": "text", "text": f"Shot {shot_idx} generated keyframe:"})
                content.append({"type": "image_url", "image_url": {"url": image_path_to_b64(keyframe, mime=True)}})

            raw: Dict[str, Any] = {}
            vlm_failed = False
            try:
                raw = await self._invoke_review(content=content)
            except Exception as exc:  # noqa: BLE001 - QA must never break rendering
                logger.warning("Visual QA batched VLM review failed for scene %s: %s", scene_idx, exc)
                vlm_failed = True

            # Per-shot fallback: if the batched review failed, try each shot
            # individually so a single bad upload does not discard all findings.
            if vlm_failed and pairs:
                any_ok = False
                for shot_idx, keyframe, panel in pairs[:max_frame_pairs]:
                    single_content: List[Dict[str, Any]] = [
                        {
                            "type": "text",
                            "text": _human_prompt_template_qa.format(
                                characters_str=characters_str,
                                environments_str=environments_str,
                                shots_str=f"Shot {shot_idx}: (single-shot review)",
                            ),
                        }
                    ]
                    if panel:
                        single_content.append({"type": "text", "text": f"Shot {shot_idx} panel ref:"})
                        single_content.append({"type": "image_url", "image_url": {"url": image_path_to_b64(panel, mime=True)}})
                    single_content.append({"type": "text", "text": f"Shot {shot_idx} generated keyframe:"})
                    single_content.append({"type": "image_url", "image_url": {"url": image_path_to_b64(keyframe, mime=True)}})
                    try:
                        single_raw = await self._invoke_review(content=single_content)
                        any_ok = True
                        for item in single_raw.get("issues", []) or []:
                            try:
                                si = int(item.get("shot_idx", shot_idx))
                            except (TypeError, ValueError):
                                si = shot_idx
                            aspect = str(item.get("aspect", "artifact"))
                            severity = str(item.get("severity", "warning"))
                            report.issues.append(
                                VisualQAIssue(
                                    shot_idx=si,
                                    aspect=aspect if aspect in _ALLOWED_ASPECTS else "artifact",
                                    severity=severity if severity in _ALLOWED_SEVERITIES else "warning",
                                    description=str(item.get("description", "")),
                                    suggestion=str(item.get("suggestion", "")),
                                )
                            )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Visual QA per-shot review failed for scene %s shot %s: %s", scene_idx, shot_idx, exc)
                        report.issues.append(
                            VisualQAIssue(
                                shot_idx=shot_idx,
                                aspect="vlm_review",
                                severity="info",
                                description=f"VLM review failed for shot {shot_idx}: {exc}",
                                suggestion="Retry when the VLM endpoint is stable.",
                            )
                        )
                if not any_ok:
                    report.issues.append(
                        VisualQAIssue(
                            shot_idx=-1,
                            aspect="vlm_review",
                            severity="info",
                            description="All VLM review attempts failed for this scene (degraded mode).",
                            suggestion="Check muse2api/OpenRouter upload stability and retry QA.",
                        )
                    )
            else:
                for item in raw.get("issues", []) or []:
                    try:
                        shot_idx = int(item.get("shot_idx", -1))
                    except (TypeError, ValueError):
                        continue
                    aspect = str(item.get("aspect", "artifact"))
                    severity = str(item.get("severity", "warning"))
                    report.issues.append(
                        VisualQAIssue(
                            shot_idx=shot_idx,
                            aspect=aspect if aspect in _ALLOWED_ASPECTS else "artifact",
                            severity=severity if severity in _ALLOWED_SEVERITIES else "warning",
                            description=str(item.get("description", "")),
                            suggestion=str(item.get("suggestion", "")),
                        )
                    )

        failed = {issue.shot_idx for issue in report.issues if issue.severity in {"critical", "warning"}}
        all_shot_idxs = {sd.idx for sd in shot_descriptions}
        report.failed_shots = sorted(failed)
        report.passed_shots = sorted(all_shot_idxs - failed)
        critical_count = sum(1 for i in report.issues if i.severity == "critical")
        report.summary = (
            f"{report.reviewed_frames} keyframe(s) reviewed; "
            f"{len(report.passed_shots)} passed, {len(report.failed_shots)} flagged "
            f"({critical_count} critical)."
        )

        if save:
            try:
                (scene_root / "visual_qa_report.json").write_text(
                    json.dumps(report.model_dump(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except OSError:
                pass
        return report


def _extract_beat_refs(text: str) -> List[str]:
    beat_ids: List[str] = []
    for match in re.finditer(r"\s*\[(?:RefBeat|BEAT)\s*:\s*([^\]]+)\]\s*", str(text or ""), re.IGNORECASE):
        for token in re.split(r"[,;\s]+", match.group(1).strip()):
            token = token.strip()
            if token and token not in beat_ids:
                beat_ids.append(token)
    return beat_ids
