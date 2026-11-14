import os
import re
import inspect
import shutil
import json
import logging
import asyncio
import time
import hashlib
from pathlib import Path
from uuid import uuid4
from typing import Any, Callable, Optional, Dict, List, Tuple, Literal, Type, TypeVar
from moviepy import VideoFileClip, concatenate_videoclips
from PIL import Image
from agents import *
from agents.best_image_selector import BestImageSelector
import yaml
from interfaces import *
from langchain.chat_models import init_chat_model
from tools.render_backend import RenderBackend
from agent_runtime.config import (
    audio_api_key,
    audio_base_url,
    audio_bgm_file,
    audio_bgm_volume,
    audio_character_voices,
    audio_model,
    audio_model_provider,
    audio_voice,
    quality_generic_linter,
    quality_max_reference_images,
    quality_ref_policy,
    subtitles_burn_in,
    subtitles_enabled,
    subtitles_font,
    subtitles_font_size,
    subtitles_language,
)
from utils.series_bible import load_configured_series
from utils.audio import (
    build_subtitle_cues,
    burn_subtitles_into_video,
    extract_all_dialogues_from_shot,
    extract_spoken_dialogues_for_subtitles,
    extract_vietsubs_from_script,
    extract_vietsubs_from_shots,
    mix_multitrack_video_and_audio,
    resolve_voice_and_prosody,
    synthesize_speech,
    translate_subtitles_with_chat_model,
    write_ass_subtitles,
    write_srt_subtitles,
)
from utils.provider_presets import resolve_chat_model_config
from utils.image_selection import DEFAULT_IMAGE_CANDIDATES, image_candidate_count_from_config, validate_image_candidate_count
from utils.video import compose_timeline_clips, normalize_clips_for_concat
from agents.camera_image_generator import _validate_camera_tree




TModel = TypeVar("TModel")


def _normalize_model_list(items: Any, model_cls: Type[TModel], field_name: str) -> List[TModel]:
    if items is None:
        return []
    if not isinstance(items, list):
        raise TypeError(f"{field_name} must be a list, got {type(items).__name__}")
    normalized: List[TModel] = []
    for idx, item in enumerate(items):
        if isinstance(item, model_cls):
            normalized.append(item)
        elif isinstance(item, dict):
            normalized.append(model_cls.model_validate(item))
        else:
            raise TypeError(f"{field_name}[{idx}] must be {model_cls.__name__} or dict, got {type(item).__name__}")
    return normalized


def _resolve_script_vietsubs(
    shot_descriptions: List[ShotDescription],
    effective_script: str,
) -> Dict[int, str]:
    shot_vietsubs = extract_vietsubs_from_shots(shot_descriptions)
    script_subs = extract_vietsubs_from_script(effective_script)
    if not shot_vietsubs:
        return script_subs
    resolved: Dict[int, str] = {}
    if len(script_subs) == len(shot_descriptions):
        resolved.update(script_subs)
    resolved.update(shot_vietsubs)
    return resolved


def _group_shots_into_cameras(shot_descriptions: List[ShotDescription]) -> List[Camera]:
    cameras_by_idx: Dict[int, Camera] = {}
    for shot_description in shot_descriptions:
        camera = cameras_by_idx.get(shot_description.cam_idx)
        if camera is None:
            camera = Camera(idx=shot_description.cam_idx, active_shot_idxs=[])
            cameras_by_idx[shot_description.cam_idx] = camera
        camera.active_shot_idxs.append(shot_description.idx)
    return list(cameras_by_idx.values())

def _collect_priority_shot_idxs(camera_tree: List[Camera]) -> List[int]:
    """Shot indices that other cameras depend on."""
    return [camera.parent_shot_idx for camera in camera_tree if camera.parent_shot_idx is not None]


def shot_requires_last_frame(shot_descriptions: List[ShotDescription], shot: ShotDescription) -> bool:
    """Return True when a shot should render and lock an explicit last_frame keyframe."""
    vis = (getattr(shot, "visual_desc", "") or "") + " " + (getattr(shot, "motion_desc", "") or "")
    
    # 1. Flexible Sakuga Keyframing: Action bursts, dashes, vanishes, and pure speed blitz
    # NEVER generate a separate last frame (Image-to-Video single frame conditioning).
    # This prevents 24fps diffusion models from hallucinating 192 frames of morphing or awkward filler running!
    if any(tag in vis for tag in ["[AnimeFX: FrameSkip]", "[AnimeFX: ActionBurst]", "[AnimeFX: SpeedStreak]", "[AnimeFX: Vanish]"]) or \
       any(kw in vis.lower() for kw in ["vanish", "biến mất", "supersonic", "blitz", "shunpo", "flash step"]):
        return False

    # 2. State transformations (e.g. human -> cat transformation) MUST use both frames
    is_transformation = any(kw in vis.lower() for kw in ["transform", "morph", "biến hình", "biến thân", "mèo béo"])
    if is_transformation:
        return True

    # 3. Small variation shots (subtle dialogue, breathing, gentle holds) strictly use single-frame conditioning
    if getattr(shot, "variation_type", "small") == "small":
        return False

    if getattr(shot, "variation_type", "small") in {"medium", "large"}:
        return True
    return False


def _previous_shot_description(shot_descriptions: List[ShotDescription], shot_idx: int) -> Optional[ShotDescription]:
    for i, candidate in enumerate(shot_descriptions):
        if candidate.idx == shot_idx:
            return shot_descriptions[i - 1] if i > 0 else None
    return None


def _pipeline_print(quiet: bool, message: str) -> None:
    if not quiet:
        print(message)


def _emit_text_plan_progress(progress, stage: str, message: str, metadata: Dict[str, Any] | None = None) -> None:
    if progress is not None:
        progress(stage, message, metadata or {})


def _emit_render_progress(progress, stage: str, message: str, metadata: Dict[str, Any] | None = None) -> None:
    if progress is not None:
        progress(stage, message, metadata or {})


def _scoped_progress(progress, **scope):
    if progress is None:
        return None

    def emit(stage: str, message: str, metadata: Dict[str, Any] | None = None) -> None:
        payload = dict(scope)
        payload.update(metadata or {})
        _emit_render_progress(progress, stage, message, payload)

    return emit


def _resolve_workspace_root(working_dir: str | Path) -> Path:
    base = Path(working_dir).resolve()
    for candidate in (base, *base.parents):
        if (candidate / "configs" / "agent.local.yaml").is_file():
            return candidate
    return base


def _scene_idx_from_working_dir(working_dir: str | Path) -> Optional[int]:
    m = re.search(r"scene_(\d+)", str(working_dir))
    return int(m.group(1)) if m else None


def _character_names_in_prompt(prompt: str, vis_chars: Optional[List[Any]] = None) -> List[str]:
    """Distinct character identifiers referenced in a prompt (<Tag> + roster idents)."""
    names: List[str] = []
    for raw in re.findall(r"<([^>]+)>", prompt or ""):
        if raw.strip() and raw.strip() not in names:
            names.append(raw.strip())
    for c in vis_chars or []:
        ident = c.get("identifier_in_scene", "") if isinstance(c, dict) else getattr(c, "identifier_in_scene", "")
        if ident and ident not in names:
            names.append(ident)
    return names


def _apply_series_enrichment(prompt: str, working_dir: str | Path, vis_chars: Optional[List[Any]] = None) -> str:
    """Append series style palette + per-present-character canon notes to a keyframe prompt.

    Generic replacement for the old hardcoded ``enforce_*`` chain. No-op when the
    generic linter is off or no series bible is configured.
    """
    workspace_root = _resolve_workspace_root(working_dir)
    if not quality_generic_linter(workspace_root):
        return prompt
    bible = load_configured_series(workspace_root)
    if not bible.available:
        return prompt
    scene_idx = _scene_idx_from_working_dir(working_dir)
    additions: List[str] = []
    seen_notes: set[str] = set()
    for name in _character_names_in_prompt(prompt, vis_chars):
        note = bible.character_notes(name, scene_idx)
        if note and note not in seen_notes:
            seen_notes.add(note)
            additions.append(f"[CHARACTER CANON — {name}]: {note}")
    style = bible.style_prompt()
    out = prompt.rstrip()
    if additions:
        out = out + "\n" + "\n".join(additions)
    if style and style not in out:
        out = out + "\n" + style
    return out


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _subtitle_burn_marker_path(working_dir: str | Path, suffix: str) -> Path:
    return Path(working_dir) / f".subtitles_burned_{suffix}.json"


def _normalize_shot_panel_refs(raw: Any, *, require_exists: bool = False) -> Dict[int, List[Tuple[str, str]]]:
    if not isinstance(raw, dict):
        return {}
    normalized: Dict[int, List[Tuple[str, str]]] = {}
    for k, entries in raw.items():
        try:
            shot_idx = int(k)
        except (TypeError, ValueError):
            continue
        pairs: List[Tuple[str, str]] = []
        for entry in entries or []:
            path_val = ""
            desc_val = ""
            if isinstance(entry, dict):
                path_val = str(entry.get("path", "") or "").strip()
                desc_val = str(entry.get("description", "") or "").strip()
            elif isinstance(entry, (list, tuple)) and len(entry) >= 1:
                path_val = str(entry[0] or "").strip()
                desc_val = str(entry[1] or "").strip() if len(entry) > 1 else ""
            if not path_val:
                continue
            if require_exists and not os.path.exists(path_val):
                continue
            if not desc_val:
                desc_val = (
                    "ORIGINAL COMIC PANEL COMPOSITION & POSE REFERENCE "
                    "(redraw cleanly in borderless textless 2D anime style matching the character portraits; "
                    "remove speech bubbles, panel frames, and SFX text)."
                )
            pairs.append((path_val, desc_val))
            if len(pairs) >= 2:
                break
        if pairs:
            normalized[shot_idx] = pairs
    return normalized


def _load_shot_panel_refs(working_dir: str | Path) -> Dict[int, List[Tuple[str, str]]]:
    refs_path = Path(working_dir) / "shot_panel_refs.json"
    if not refs_path.is_file():
        return {}
    try:
        raw = json.loads(refs_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return _normalize_shot_panel_refs(raw, require_exists=True)


def _dedupe_image_text_pairs(pairs: List[Tuple[str, str]], limit: Optional[int] = None) -> List[Tuple[str, str]]:
    deduped: List[Tuple[str, str]] = []
    seen_paths: set[str] = set()
    for item in pairs:
        if not item or len(item) < 2:
            continue
        p_raw, desc = str(item[0] or "").strip(), str(item[1] or "")
        if not p_raw:
            continue
        norm_key = os.path.normcase(os.path.normpath(p_raw))
        if norm_key in seen_paths:
            continue
        seen_paths.add(norm_key)
        deduped.append((p_raw, desc))
        if limit is not None and len(deduped) >= limit:
            break
    return deduped


def _load_character_seed_refs(working_dir: str | Path) -> Dict[str, List[str]]:
    refs_path = Path(working_dir) / "character_seed_refs.json"
    if not refs_path.is_file():
        return {}
    try:
        raw = json.loads(refs_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    result: Dict[str, List[str]] = {}
    for k, paths in raw.items():
        vals = [str(p).strip() for p in (paths or []) if str(p).strip() and os.path.exists(str(p).strip())][:4]
        if vals:
            result[str(k)] = vals
    return result


def _resolve_character_acoustic_profile(speaker: str, characters_data: List[Any], context_desc: str = "") -> Dict[str, str]:
    """Generically resolve a character's acoustic profile, age group, vocal timbre, and negative constraints."""
    spk_raw = (speaker or "").strip()
    spk_clean = re.sub(r"\s*\([^)]*\)", "", spk_raw).strip()
    if not spk_clean:
        return {
            "clean_speaker": "Character",
            "profile": "Clear expressive Japanese anime voice actor matching scene context.",
            "negatives": "Strictly clean studio recording, no robotic or synthetic artifacts."
        }

    char_match = None
    for c in (characters_data or []):
        if isinstance(c, dict):
            ident = c.get("identifier_in_scene", "")
            static_feat = c.get("static_features", "")
            dyn_feat = c.get("dynamic_features", "") or ""
        else:
            ident = getattr(c, "identifier_in_scene", "")
            static_feat = getattr(c, "static_features", "")
            dyn_feat = getattr(c, "dynamic_features", "") or ""

        ident_clean = re.sub(r"\s*\([^)]*\)", "", ident).strip()
        if spk_raw.lower() == ident.lower() or spk_clean.lower() == ident_clean.lower():
            char_match = (ident, static_feat, dyn_feat)
            break
    if not char_match:
        for c in (characters_data or []):
            if isinstance(c, dict):
                ident = c.get("identifier_in_scene", "")
                static_feat = c.get("static_features", "")
                dyn_feat = c.get("dynamic_features", "") or ""
            else:
                ident = getattr(c, "identifier_in_scene", "")
                static_feat = getattr(c, "static_features", "")
                dyn_feat = getattr(c, "dynamic_features", "") or ""

            ident_clean = re.sub(r"\s*\([^)]*\)", "", ident).strip()
            if ident_clean.lower() in spk_clean.lower() or spk_clean.lower() in ident_clean.lower():
                char_match = (ident, static_feat, dyn_feat)
                break

    desc = ""
    if char_match:
        desc = f"{char_match[1]} {char_match[2]}".lower()

    # Prefer a series-bible voice description when one is configured (series-agnostic).
    series_voice = ""
    try:
        from utils.series_bible import load_configured_series as _lcs
        series_voice = _lcs(".").voice_description(spk_clean)
    except Exception:
        series_voice = ""
    if series_voice:
        return {"clean_speaker": spk_clean, "profile": series_voice, "negatives": "Keep the voice consistent with the character's established profile."}

    is_narrator = any(w in spk_clean.lower() or w in desc for w in ["narrator", "người dẫn chuyện", "narration"])
    is_woman = any(w in spk_clean.lower() or w in desc for w in ["woman", "mother", "female", "girl", "mẹ", "bà", "cô"])
    is_child = (not is_woman) and any(w in desc for w in ["elementary", "child", "small young", "6-7", "kid"])
    is_teen = (not is_woman) and (not is_child) and any(w in desc for w in ["teen", "high school", "16", "17", "young boy"])
    is_cat = any(w in spk_clean.lower() or w in desc for w in ["cat", "tabby", "mèo"])
    is_man = (not is_woman) and (not is_child) and (not is_teen) and (not is_cat) and any(w in desc for w in ["man", "male", "athletic build", "trench coat"])

    if is_narrator:
        profile = "Authoritative resonant Japanese anime narrator baritone. Calm, cinematic, neutral."
        neg = "Strictly NO casual dialogue tone, NO emotional bias."
    elif is_woman:
        profile = "Gentle adult woman (~30-35 years old), warm, natural, emotional Japanese female voice."
        neg = "Strictly NOT male voice, NOT robotic, NOT high-pitched squeak."
    elif is_child:
        profile = "Young elementary school boy (~6-7 years old), innocent, clear young child anime voice."
        neg = "Strictly NO deep adult male voice, NO teenage voice, NO mature female tone."
    elif is_teen:
        profile = "16-year-old teenage boy (high school male), youthful, clear, earnest anime boy voice. Emotional and expressive."
        neg = "Strictly NOT female voice, NOT baby/toddler voice, NOT old man voice, NOT robot."
    elif is_cat:
        profile = "A comically deep, gritty, grumpy masculine adult baritone voice speaking out from the fat chubby orange cat. Arrogant, deadpan, comically serious."
        neg = "Strictly NOT female voice, NOT cute kitten meow, NOT high-pitched baby voice, NOT cartoon squeak."
    elif is_man:
        audio_ctx = f"{context_desc} {spk_raw}".lower()
        if any(w in audio_ctx for w in ["groan", "grunt", "deep bass", "pained", "guttural"]):
            profile = "Extremely deep resonant bass adult male (gravelly low-pitched guttural groan of intense physical battle pain near 80-100Hz). Heavy masculine chest resonance."
            neg = "Strictly NOT female voice, NOT high-pitched airy breath, NOT feminine sigh or gasp, NOT child voice, NOT soft whisper."
        else:
            profile = "Powerful awakened adult male in his late 20s, deep resonant authoritative Japanese anime male baritone. Confident, charismatic, proud."
            neg = "Strictly NOT female voice, NOT high-pitched baby/child voice, NOT generic cartoon meow."
    else:
        profile = "Clear expressive Japanese anime voice actor matching scene context."
        neg = "Strictly clean studio recording, no robotic or synthetic artifacts."

    return {
        "clean_speaker": spk_clean,
        "profile": profile,
        "negatives": neg,
    }


def _clean_sfx_onomatopoeia(text: str) -> str:
    """Clean comic onomatopoeia letters and strip mouth sounds from SFX text."""
    if not text:
        return ""
    # Strip quoted comic words like 'SPPPPPT!', 'SSSSSHHHHHH', "BZZZT!"
    c = re.sub(r"['\"][A-Za-z0-9!?-]{2,}['\"]", "", text)
    # Strip uppercase sound words longer than 2 chars like SSSSSHHHHHHH...
    c = re.sub(r"\b[A-Z]{3,}[.!?]*", "", c)
    c = re.sub(r"[:\s,-]+$", "", c).strip()
    return c


def _format_video_prompt_audio(
    clean_audio_desc: str,
    characters_data: List[Any],
    timing_note: str = "",
    is_back_view: bool = False,
    is_no_face: bool = False,
) -> str:
    """Format dialogue into structured, hallucination-proof audio & vocal profile specifications."""
    if not clean_audio_desc or not clean_audio_desc.strip():
        return ""

    sfx_match = re.findall(r"\[Sound\s*Effect\]\s*([^\[\n]+)", clean_audio_desc, re.IGNORECASE)
    sfx_text = ", ".join(_clean_sfx_onomatopoeia(s.strip()) for s in sfx_match if _clean_sfx_onomatopoeia(s.strip()))

    narrator_match = re.search(
        r"\[(?:Speaker|Off-screen.*?Voiceover)\]\s*(?:Người Dẫn Chuyện|Narrator)\s*(?:\([^)]*\))?\s*:\s*(?:'|\")?([^'\"]+)(?:'|\")?",
        clean_audio_desc,
        re.IGNORECASE,
    )
    if narrator_match:
        line = narrator_match.group(1).strip()
        prof = _resolve_character_acoustic_profile("Narrator", characters_data)
        out = [
            "[Audio & Voiceover Track ONLY - Spoken Japanese Narration & SFX, No On-screen Subtitles]",
            f"- Active Speaker: Narrator",
            f"- Vocal Acoustic Profile: {prof['profile']}",
            f"- Acoustic Rules: {prof['negatives']} All visible characters remain completely silent.",
            f"- Narration Line: '{line}'",
        ]
        if sfx_text:
            out.append(f"- Sound Effects: {sfx_text}")
        if timing_note:
            out.append(timing_note.strip())
        return "\n".join(out)

    thought_match = re.search(
        r"\[(?:Thought|Inner\s*Monologue)\]\s*([^:\n]+?)\s*:\s*(?:'|\")?([^'\"]+)(?:'|\")?",
        clean_audio_desc,
        re.IGNORECASE,
    )
    if not thought_match:
        thought_match = re.search(
            r"\[Speaker\]\s*([^:\n]+?)\s*\(([^)]*inner\s*monologue[^)]*)\)\s*:\s*(?:'|\")?([^'\"]+)(?:'|\")?",
            clean_audio_desc,
            re.IGNORECASE,
        )
    if not thought_match:
        thought_match = re.search(
            r"\[Off-screen\s+Japanese\s+Inner\s+Monologue\s+Voiceover\s+of\s+([^:\]\n]+?)\]\s*:\s*(?:'|\")?([^'\"]+)(?:'|\")?",
            clean_audio_desc,
            re.IGNORECASE,
        )
    if thought_match:
        speaker_block = thought_match.group(1).strip()
        line = thought_match.group(thought_match.lastindex).strip() if thought_match.lastindex else ""
        prof = _resolve_character_acoustic_profile(speaker_block, characters_data)
        out = [
            "[Audio & Voiceover Track ONLY - Spoken Japanese Inner Monologue & SFX, No On-screen Subtitles]",
            f"- Active Speaker: {prof['clean_speaker']}",
            f"- Vocal Acoustic Profile: {prof['profile']}",
            f"- Delivery Mode: Off-screen inner monologue voiceover (character lips remain completely closed, subtle introspective resonance).",
            f"- Acoustic Rules: {prof['negatives']} Only {prof['clean_speaker']}'s internal voice is heard. Visible character lips stay closed. No other character speaks.",
            f"- Spoken Line: '{line}'",
        ]
        if sfx_text:
            out.append(f"- Sound Effects: {sfx_text}")
        if timing_note:
            out.append(timing_note.strip())
        return "\n".join(out)

    spoken_match = re.search(
        r"\[Speaker\]\s*([^:\n]+?)\s*:\s*(?:'|\")?([^'\"]+)(?:'|\")?",
        clean_audio_desc,
        re.IGNORECASE,
    )
    if spoken_match:
        speaker_block = spoken_match.group(1).strip()
        em = re.findall(r"\(([^)]+)\)", speaker_block)
        raw_em = em[-1].strip() if em else ""
        emotion = "" if re.search(r"\b(cat|human|boy|girl|man|woman)\b", raw_em.lower()) else raw_em
        line = spoken_match.group(spoken_match.lastindex).strip() if spoken_match.lastindex else ""
        prof = _resolve_character_acoustic_profile(speaker_block, characters_data, context_desc=clean_audio_desc)
        if is_no_face:
            delivery_mode = "Pure off-screen voiceover. Camera remains strictly locked on the subject/prop/environment in Image 0. Strictly NO character face in frame, NO human head, NO mouth visible."
        elif is_back_view:
            delivery_mode = "Physical guttural vocal groan from behind (rear framing; character remains completely facing away with head bowed down toward the ground; camera holds strictly static behind the character; strictly DO NOT turn the character around, DO NOT reveal face or mouth)."
        else:
            delivery_mode = "Spoken dialogue (expressive, natural dialogue pacing, lips move in sync)."
        out = [
            "[Audio & Spoken Japanese Dialogue Track ONLY - Studio Anime Voice Acting, No On-screen Subtitles]",
            f"- Active Speaker: {prof['clean_speaker']}" + (f" ({emotion})" if emotion else ""),
            f"- Vocal Acoustic Profile: {prof['profile']}",
            f"- Delivery Mode: {delivery_mode}",
            f"- Acoustic Rules: {prof['negatives']} Only {prof['clean_speaker']} speaks. All other characters in the shot remain completely silent.",
            f"- Spoken Line: '{line}'",
        ]
        if sfx_text:
            out.append(f"- Sound Effects: {sfx_text}")
        if timing_note:
            out.append(timing_note.strip())
        return "\n".join(out)

    if sfx_text or clean_audio_desc:
        char_vocal_rule = ""
        for c in (characters_data or []):
            ident = c.get("identifier_in_scene", "") if isinstance(c, dict) else getattr(c, "identifier_in_scene", "")
            ident_clean = re.sub(r"\s*\([^)]*\)", "", ident).strip()
            if ident_clean and re.search(rf"\b{re.escape(ident_clean)}\b", clean_audio_desc, re.IGNORECASE):
                prof = _resolve_character_acoustic_profile(ident_clean, characters_data, context_desc=clean_audio_desc)
                char_vocal_rule = (
                    f"- Character Vocal Identity: {prof['clean_speaker']}\n"
                    f"- Vocal Acoustic Profile: {prof['profile']}\n"
                    f"- Acoustic Rules: {prof['negatives']} Any coughs, groans, labored breathing, or vocal sounds from {prof['clean_speaker']} MUST strictly match this deep adult male seiyuu identity. Strictly NO female voice, NO high-pitched sounds."
                )
                break

        cleaned_ambient = _clean_sfx_onomatopoeia(clean_audio_desc) or "Atmospheric cinematic sound design"
        out = [
            "[Audio & Sound Effects Track ONLY - No Spoken Dialogue, No On-screen Subtitles]",
            f"- Sound Effects & Ambient: {cleaned_ambient}",
            char_vocal_rule if char_vocal_rule else "- Acoustic Rules: Pure Foley and environmental sound effects. Strictly NO human mouth noises, NO vocal hissing, NO random speech.",
        ]
        return "\n".join(out)

    return ""


def _portrait_pairs_for_characters(visible_characters, character_portraits_registry):
    """One best-angle portrait (path, ident, desc) per present character."""
    out = []
    seen = set()
    if character_portraits_registry and visible_characters:
        for char in visible_characters:
            if not getattr(char, "is_visible", True):
                continue
            ident = getattr(char, "identifier_in_scene", "")
            reg = character_portraits_registry.get(ident) or {}
            chosen = (
                reg.get("front")
                or reg.get("front_three_quarter")
                or reg.get("side")
                or next(iter(reg.values()), None)
            )
            if chosen and isinstance(chosen, dict) and os.path.exists(chosen.get("path", "")):
                p_path = chosen["path"]
                if p_path not in seen:
                    desc = (
                        f"Canonical character portrait reference for <{ident}>: maintain exact facial features, "
                        f"hairstyle, hair color, eye shape, and clothing colors directly matching this portrait. "
                        f"(NOTE: Do NOT copy the studio background of this portrait; adopt the scene lighting anchor "
                        f"for environment and color grading)."
                    )
                    out.append((p_path, ident, desc))
                    seen.add(p_path)
    return out


def _ensure_character_portraits_and_spatial_anchors(
    pairs: List[Tuple[str, str]],
    visible_characters: List[Any],
    character_portraits_registry: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None,
    scene_anchor_pair: Optional[Tuple[str, str]] = None,
    limit: int = 4,
    *,
    sheet_dir: Optional[str] = None,
    sheet_tag: str = "",
) -> List[Tuple[str, str]]:
    """Guarantees that canonical portraits for all visible characters appear FIRST (Image 0, Image 1...)
    so diffusion models lock onto authentic facial likeness and wardrobe before secondary anchors.

    When there are more present characters than ``limit - 1`` reference slots and a
    ``sheet_dir`` is given, their portraits are tiled into a single reference sheet so
    no character is dropped (phase 03, quality.ref_policy != legacy).
    """
    portraits = _portrait_pairs_for_characters(visible_characters, character_portraits_registry)
    portrait_paths = {p for p, _ident, _d in portraits}

    portrait_pairs: List[Tuple[str, str]] = []
    # If characters outnumber the slot budget (keep >=1 slot for a scene anchor), tile a sheet.
    if sheet_dir and len(portraits) > max(1, limit - 1):
        from utils import reference_sheet
        entries = [(p, ident) for p, ident, _d in portraits]
        out_path = os.path.join(sheet_dir, f"reference_sheet{('_' + sheet_tag) if sheet_tag else ''}.png")
        try:
            saved = reference_sheet.compose_to_file(entries, out_path)
        except Exception:
            saved = None
        if saved:
            portrait_pairs.append((saved, reference_sheet.order_label(entries)))
        else:
            portrait_pairs.extend((p, d) for p, _ident, d in portraits)
    else:
        portrait_pairs.extend((p, d) for p, _ident, d in portraits)

    other_pairs = [p for p in (pairs or []) if p[0] not in portrait_paths]

    if scene_anchor_pair and scene_anchor_pair[0] and os.path.exists(scene_anchor_pair[0]):
        if scene_anchor_pair[0] not in portrait_paths and scene_anchor_pair[0] not in {p[0] for p in other_pairs}:
            other_pairs.append(scene_anchor_pair)

    # Ensure manga panel anchors do not cause beard/cross-hatch or chibi hallucination
    cleaned_other = []
    for p_path, p_text in other_pairs:
        if "panel_anchors" in p_path:
            p_text = (
                f"{p_text} (Layout/composition anchor only. Do NOT copy simplified cartoon/chibi expressions, "
                f"dot eyes, screentone, or black cross-hatching. Character facial likeness and wardrobe must "
                f"strictly match the turnaround portrait Image 0)."
            )
        cleaned_other.append((p_path, p_text))

    all_pairs = portrait_pairs + cleaned_other
    return _dedupe_image_text_pairs(all_pairs, limit=limit)


def _classify_ref_role(path: str, text: str, portrait_paths: set) -> str:
    p = (path or "").lower()
    t = (text or "").lower()
    if path in portrait_paths or "reference_sheet" in p or "portrait" in t or "turnaround" in t:
        return "portrait"
    if "panel_anchors" in p or "panel" in t:
        return "panel"
    if "anchor" in t or "video_end" in p or "establishing" in t:
        return "scene_anchor"
    if "continuity" in t or "previous shot" in t:
        return "continuity"
    return "other"


def _write_refs_used(working_dir, shot_idx, frame_type, pairs, visible_characters, character_portraits_registry):
    """Record the reference images actually sent to the image model (debug aid, always on)."""
    try:
        portraits = _portrait_pairs_for_characters(visible_characters, character_portraits_registry)
        portrait_paths = {p for p, _i, _d in portraits}
        n_present = sum(1 for c in (visible_characters or []) if getattr(c, "is_visible", True))
        records = [
            {"path": p, "text": t, "role": _classify_ref_role(p, t, portrait_paths)}
            for p, t in pairs
        ]
        n_portrait = sum(1 for r in records if r["role"] == "portrait")
        out_dir = os.path.join(working_dir, "shots", str(shot_idx))
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{frame_type}_refs_used.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"n_present": n_present, "n_portrait_refs": n_portrait, "n_refs": len(records), "refs": records},
                f, ensure_ascii=False, indent=2,
            )
    except Exception:
        pass


class Script2VideoPipeline:

    def __init__(
        self,
        chat_model: str,
        image_generator,
        video_generator,
        working_dir: str,
        num_image_candidates: int = DEFAULT_IMAGE_CANDIDATES,
    ):

        self.chat_model = chat_model
        self.image_generator = image_generator
        self.video_generator = video_generator
        self.num_image_candidates = validate_image_candidate_count(num_image_candidates)
        self.best_image_selector = BestImageSelector(chat_model=self.chat_model)

        self.character_extractor = CharacterExtractor(chat_model=self.chat_model)
        self.character_portraits_generator = CharacterPortraitsGenerator(image_generator=self.image_generator)
        self.storyboard_artist = StoryboardArtist(chat_model=self.chat_model)
        self.camera_image_generator = CameraImageGenerator(chat_model=self.chat_model, image_generator=self.image_generator, video_generator=self.video_generator)
        self.reference_image_selector = ReferenceImageSelector(chat_model=self.chat_model)

        self.working_dir = working_dir
        os.makedirs(self.working_dir, exist_ok=True)
        self.character_portrait_events = {}
        self.shot_desc_events = {}
        self.frame_events = {}


    async def plan_text_artifacts(
        self,
        script: str,
        user_requirement: str,
        style: str,
        characters: List[CharacterInScene] = None,
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
        quiet: bool = False,
    ):
        """Generate only structured text artifacts required before rendering.

        This helper intentionally stops before character portraits, frame generation,
        video generation, and final concatenation so an agent loop can pause for
        user review after narrative planning.
        """
        self.character_portrait_events = {}
        self.shot_desc_events = {}
        self.frame_events = {}

        if characters is None:
            _emit_text_plan_progress(progress, "extract_characters", "Extracting characters from script")
            characters = await self.extract_characters(script=script, quiet=quiet)
        else:
            characters = _normalize_model_list(characters, CharacterInScene, "characters")
            _emit_text_plan_progress(progress, "extract_characters", "Using provided characters", {"provided": True, "count": len(characters)})
            characters_path = os.path.join(self.working_dir, "characters.json")
            if not os.path.exists(characters_path):
                with open(characters_path, "w", encoding="utf-8") as f:
                    json.dump([character.model_dump() for character in characters], f, ensure_ascii=False, indent=4)
            for character in characters:
                self.character_portrait_events[character.idx] = asyncio.Event()

        _emit_text_plan_progress(progress, "design_storyboard", "Designing storyboard")
        storyboard = await self.design_storyboard(
            script=script,
            characters=characters,
            user_requirement=user_requirement,
            quiet=quiet,
        )
        _emit_text_plan_progress(progress, "decompose_shots", "Decomposing shot visual descriptions", {"shot_count": len(storyboard)})
        shot_descriptions = await self.decompose_visual_descriptions(
            shot_brief_descriptions=storyboard,
            characters=characters,
            quiet=quiet,
        )
        camera_tree = None
        for attempt in range(2):
            try:
                stage = "construct_camera_tree" if attempt == 0 else "construct_camera_tree_retry"
                message = "Constructing camera tree" if attempt == 0 else "Retrying camera tree construction after schema/type failure"
                _emit_text_plan_progress(progress, stage, message, {"shot_count": len(shot_descriptions), "attempt": attempt + 1})
                camera_tree = await self.construct_camera_tree(
                    shot_descriptions=shot_descriptions,
                    quiet=quiet,
                )
                break
            except Exception:
                camera_tree_path = os.path.join(self.working_dir, "camera_tree.json")
                if os.path.exists(camera_tree_path):
                    os.remove(camera_tree_path)
                if attempt == 1:
                    raise
        assert camera_tree is not None
        return {
            "characters": characters,
            "storyboard": storyboard,
            "shot_descriptions": shot_descriptions,
            "camera_tree": camera_tree,
        }


    @classmethod
    def init_from_config(cls, config_path: str):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        chat_model_args = resolve_chat_model_config(config["chat_model"]["init_args"])
        chat_model = init_chat_model(**chat_model_args)
        backend = RenderBackend.from_config(config)

        return cls(
            chat_model=chat_model,
            image_generator=backend.image_generator,
            video_generator=backend.video_generator,
            working_dir=config["working_dir"],
            num_image_candidates=image_candidate_count_from_config(config),
        )

    async def __call__(
        self,
        script: str,
        user_requirement: str,
        style: str,
        characters: List[CharacterInScene] = None,
        character_portraits_registry: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None,
        quiet: bool = False,
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
        character_ref_paths: Optional[Dict[str, List[str]]] = None,
        shot_panel_refs: Optional[Dict[int, List[Tuple[str, str]]]] = None,
        stop_after: Literal["portraits", "frames", "full"] = "full",
    ):
        _emit_render_progress(progress, "render_start", "Starting script2video render")
        if stop_after not in {"portraits", "frames", "full"}:
            raise ValueError(f"Unsupported stop_after value: {stop_after!r}")

        effective_char_refs = (
            _load_character_seed_refs(self.working_dir)
            if character_ref_paths is None
            else {str(k): [str(p) for p in (v or []) if str(p).strip()][:4] for k, v in character_ref_paths.items()}
        )
        effective_panel_refs = (
            _load_shot_panel_refs(self.working_dir)
            if shot_panel_refs is None
            else _normalize_shot_panel_refs(shot_panel_refs, require_exists=False)
        )

        if characters is None:
            _emit_render_progress(progress, "extract_characters", "Extracting characters before render")
            characters = await self.extract_characters(script=script, quiet=quiet)

            # characters_path = os.path.join(self.working_dir, "characters.json")
            # if os.path.exists(characters_path):
            #     with open(characters_path, "r", encoding="utf-8") as f:
            #         characters = [CharacterInScene.model_validate(c) for c in json.load(f)]
            #     print(f"🚀 Loaded {len(characters)} characters from existing file.")
            # else:
            #     print(f"🔍 Extracting characters from script...")
            #     characters = await self.extract_characters(script=script)
            #     with open(characters_path, "w", encoding="utf-8") as f:
            #         json.dump([c.model_dump() for c in characters], f, ensure_ascii=False, indent=4)
            #     print(f"☑️ Extracted {len(characters)} characters from script and saved to {characters_path}.")
        else:
            characters = _normalize_model_list(characters, CharacterInScene, "characters")
            _emit_render_progress(progress, "extract_characters", "Using provided characters for render", {"provided": True, "count": len(characters)})
            for character in characters:
                self.character_portrait_events[character.idx] = asyncio.Event()

        if character_portraits_registry is None:
            character_portraits_registry_path = os.path.join(self.working_dir, "character_portraits_registry.json")
            if os.path.exists(character_portraits_registry_path):
                with open(character_portraits_registry_path, "r", encoding="utf-8") as f:
                    character_portraits_registry = json.load(f)
                print(f"🚀 Loaded {len(character_portraits_registry)} character portraits from existing file.")
                _emit_render_progress(progress, "character_portraits_loaded", "Loaded existing character portraits", {"count": len(character_portraits_registry)})
            else:
                print(f"🔍 Generating character portraits...")
                _emit_render_progress(progress, "character_portraits_start", "Generating character portraits", {"character_count": len(characters)})
                if effective_char_refs:
                    character_portraits_registry = await self.generate_character_portraits(
                        characters=characters,
                        character_portraits_registry=None,
                        style=style,
                        progress=progress,
                        character_ref_paths=effective_char_refs,
                    )
                else:
                    character_portraits_registry = await self.generate_character_portraits(
                        characters=characters,
                        character_portraits_registry=None,
                        style=style,
                        progress=progress,
                    )

                with open(character_portraits_registry_path, "w", encoding="utf-8") as f:
                    json.dump(character_portraits_registry, f, ensure_ascii=False, indent=4)
                print(f"☑️ Generated {len(character_portraits_registry)} character portraits and saved to {character_portraits_registry_path}.")
                _emit_render_progress(progress, "character_portraits_done", "Character portraits ready", {"count": len(character_portraits_registry)})

        if stop_after == "portraits":
            _emit_render_progress(progress, "portraits_only_done", "Character portraits complete (stop_after=portraits)", {"count": len(character_portraits_registry)})
            return character_portraits_registry



        # design shots
        _emit_render_progress(progress, "load_storyboard", "Loading or designing storyboard")
        storyboard = await self.design_storyboard(
            script=script,
            characters=characters,
            user_requirement=user_requirement,
            quiet=quiet,
        )
        if quality_generic_linter(_resolve_workspace_root(self.working_dir)):
            from agents.shot_linter import lint_and_auto_correct_storyboard
            storyboard = lint_and_auto_correct_storyboard(storyboard, characters=characters)
        try:
            with open(os.path.join(self.working_dir, "storyboard.json"), "w", encoding="utf-8") as f:
                json.dump([shot.model_dump() for shot in storyboard], f, ensure_ascii=False, indent=4)
        except OSError:
            pass
        _emit_render_progress(progress, "storyboard_ready", "Storyboard ready", {"shot_count": len(storyboard)})

        # decompose visual descriptions of shots
        _emit_render_progress(progress, "load_shot_descriptions", "Loading or decomposing shot descriptions", {"shot_count": len(storyboard)})
        shot_descriptions = await self.decompose_visual_descriptions(
            shot_brief_descriptions=storyboard,
            characters=characters,
            quiet=quiet,
        )
        if quality_generic_linter(_resolve_workspace_root(self.working_dir)):
            from agents.shot_linter import lint_and_auto_correct_shot_description
            shot_descriptions = [lint_and_auto_correct_shot_description(sd, characters=characters) for sd in shot_descriptions]
        for sd in shot_descriptions:
            try:
                sd_path = os.path.join(self.working_dir, "shots", f"{sd.idx}", "shot_description.json")
                with open(sd_path, "w", encoding="utf-8") as f:
                    json.dump(sd.model_dump(), f, ensure_ascii=False, indent=4)
            except OSError:
                pass
        _emit_render_progress(progress, "shot_descriptions_ready", "Shot descriptions ready", {"shot_count": len(shot_descriptions)})

        # construct camera tree
        _emit_render_progress(progress, "load_camera_tree", "Loading or constructing camera tree", {"shot_count": len(shot_descriptions)})
        camera_tree = await self.construct_camera_tree(
            shot_descriptions=shot_descriptions,
            quiet=quiet,
        )
        _validate_camera_tree(camera_tree)
        _emit_render_progress(progress, "camera_tree_ready", "Camera tree ready", {"camera_count": len(camera_tree)})

        self._active_shot_descriptions = shot_descriptions
        self.frame_events = {}
        for sd in shot_descriptions:
            ev_map = {"first_frame": asyncio.Event()}
            if shot_requires_last_frame(shot_descriptions, sd):
                ev_map["last_frame"] = asyncio.Event()
            self.frame_events[sd.idx] = ev_map

        priority_shot_idxs = _collect_priority_shot_idxs(camera_tree)
        _emit_render_progress(progress, "frames_start", "Generating frames for cameras", {"camera_count": len(camera_tree), "shot_count": len(shot_descriptions)})
        tasks = [
            (
                self.generate_frames_for_single_camera(
                    camera=camera,
                    shot_descriptions=shot_descriptions,
                    characters=characters,
                    character_portraits_registry=character_portraits_registry,
                    priority_shot_idxs=priority_shot_idxs,
                    progress=progress,
                    shot_panel_refs=effective_panel_refs,
                    allow_transition_video=(stop_after == "full"),
                )
                if effective_panel_refs
                else self.generate_frames_for_single_camera(
                    camera=camera,
                    shot_descriptions=shot_descriptions,
                    characters=characters,
                    character_portraits_registry=character_portraits_registry,
                    priority_shot_idxs=priority_shot_idxs,
                    progress=progress,
                    allow_transition_video=(stop_after == "full"),
                )
            )
            for camera in camera_tree
        ]

        if stop_after == "frames":
            if tasks:
                await asyncio.gather(*tasks)
            frame_paths: List[str] = []
            for sd in shot_descriptions:
                ff = os.path.join(self.working_dir, "shots", f"{sd.idx}", "first_frame.png")
                if os.path.exists(ff):
                    frame_paths.append(ff)
                lf = os.path.join(self.working_dir, "shots", f"{sd.idx}", "last_frame.png")
                if os.path.exists(lf):
                    frame_paths.append(lf)
            _emit_render_progress(progress, "frames_only_done", "Keyframes complete (stop_after=frames)", {"frame_count": len(frame_paths)})
            return frame_paths

        _emit_render_progress(progress, "video_clips_start", "Generating video clips for shots", {"shot_count": len(shot_descriptions)})
        video_tasks = [
            self.generate_video_for_single_shot(
                shot_description=shot_description,
                progress=progress,
            )
            for shot_description in shot_descriptions
        ]
        tasks.extend(video_tasks)
        if tasks:
            await asyncio.gather(*tasks)

        final_video_path = os.path.join(self.working_dir, "final_video.mp4")
        workspace_root = _resolve_workspace_root(self.working_dir)
        sub_lang = subtitles_language(workspace_root).strip().lower()
        effective_script = (
            script
            if isinstance(script, str)
            else json.dumps(script, ensure_ascii=False, indent=2)
            if script is not None
            else ""
        )
        script_txt_path = os.path.join(self.working_dir, "script.txt")
        if "vietsub:" not in effective_script.lower() and os.path.exists(script_txt_path):
            effective_script = Path(script_txt_path).read_text(encoding="utf-8")
        if sub_lang in {"vi", "vn", "vietnamese"}:
            script_vietsubs = _resolve_script_vietsubs(shot_descriptions, effective_script)
        else:
            script_vietsubs = {}
        timeline_manifest_path = os.path.join(self.working_dir, "timeline_manifest.json")
        shot_durations: Optional[List[float]] = None
        shot_start_offsets: Optional[List[float]] = None
        final_video_existed = os.path.exists(final_video_path)
        if final_video_existed:
            print(f"🚀 Skipped concatenating videos, already exists.")
            _emit_render_progress(progress, "final_video_exists", "Final video already exists", {"path": final_video_path})
            if os.path.exists(timeline_manifest_path):
                try:
                    manifest_data = json.loads(Path(timeline_manifest_path).read_text(encoding="utf-8"))
                    if isinstance(manifest_data, list) and len(manifest_data) == len(shot_descriptions):
                        shot_start_offsets = [float(item["start_sec"]) for item in manifest_data]
                        shot_durations = [float(item["duration_sec"]) for item in manifest_data]
                except (OSError, ValueError, KeyError, TypeError):
                    shot_start_offsets = None
                    shot_durations = None
        else:
            for stale_marker in Path(self.working_dir).glob(".subtitles_burned_*.json"):
                stale_marker.unlink(missing_ok=True)
            print(f"🎬 Starting concatenating videos...")
            _emit_render_progress(progress, "concat_start", "Concatenating video clips", {"shot_count": len(shot_descriptions)})
            spoken_mask = [
                bool(
                    extract_spoken_dialogues_for_subtitles(
                        getattr(sd, "audio_desc", ""),
                        getattr(sd, "visual_desc", ""),
                    )
                    or sd.idx in script_vietsubs
                    or i in script_vietsubs
                )
                for i, sd in enumerate(shot_descriptions)
            ]
            video_clips = []
            extra_clips = []
            final_video = None
            tmp_final_video_path = os.path.join(self.working_dir, f".final_video.{uuid4().hex[:8]}.tmp.mp4")
            try:
                video_clips = [
                    VideoFileClip(os.path.join(self.working_dir, "shots", f"{shot_description.idx}", "video.mp4"))
                    for shot_description in shot_descriptions
                ]
                final_video, shot_start_offsets, shot_durations, extra_clips = compose_timeline_clips(
                    video_clips,
                    shot_descriptions,
                    spoken_shot_mask=spoken_mask,
                )
                final_video.write_videofile(tmp_final_video_path, codec="libx264", audio_codec="aac", preset="medium")
            finally:
                if final_video is not None:
                    final_video.close()
                for clip in extra_clips:
                    clip.close()
                for clip in video_clips:
                    clip.close()
            if os.path.exists(tmp_final_video_path):
                os.replace(tmp_final_video_path, final_video_path)
            try:
                manifest_payload = [
                    {
                        "idx": sd.idx,
                        "start_sec": shot_start_offsets[i],
                        "duration_sec": shot_durations[i],
                        "transition_in": getattr(sd, "transition_in", "cut"),
                        "target_duration_sec": getattr(sd, "target_duration_sec", None),
                    }
                    for i, sd in enumerate(shot_descriptions)
                ]
                Path(timeline_manifest_path).write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            except OSError:
                pass
            print(f"☑️ Concatenated videos, saved to {final_video_path}.")
            _emit_render_progress(progress, "concat_done", "Final video concatenated", {"path": final_video_path})

            # --- MULTI-CHARACTER AUDIO SYNTHESIS & NATIVE AUDIO / BGM MIXING ---
            try:
                provider = audio_model_provider(workspace_root).strip().lower()
                bgm = audio_bgm_file(workspace_root)
                bgm_resolved = None
                if bgm:
                    if os.path.exists(bgm):
                        bgm_resolved = bgm
                    else:
                        candidate_bgm = os.path.join(str(workspace_root), bgm)
                        if os.path.exists(candidate_bgm):
                            bgm_resolved = candidate_bgm
                vol = audio_bgm_volume(workspace_root)
                speech_segments: List[Tuple[float, str]] = []

                if provider in {"native", "none", "off", "disabled", "video"}:
                    _emit_render_progress(progress, "audio_native", "Keeping native video audio track", {"provider": provider})
                    if bgm_resolved:
                        _emit_render_progress(progress, "audio_mix", "Mixing BGM onto native video audio", {"bgm": bgm_resolved})
                        mix_multitrack_video_and_audio(
                            video_path=final_video_path,
                            output_path=final_video_path,
                            speech_segments=[],
                            bgm_path=bgm_resolved,
                            bgm_volume=vol,
                            preserve_video_audio=True,
                        )
                        _emit_render_progress(progress, "audio_done", "Native audio and BGM mixed successfully")
                else:
                    char_voice_map = audio_character_voices(workspace_root)
                    fallback_offset = 0.0

                    _emit_render_progress(progress, "audio_start", "Processing multi-character voiceovers", {"provider": provider})

                    for i, sd in enumerate(shot_descriptions):
                        dialogues = extract_all_dialogues_from_shot(
                            getattr(sd, 'audio_desc', ''),
                            getattr(sd, 'visual_desc', '')
                        )
                        shot_len = shot_durations[i] if shot_durations and i < len(shot_durations) else 8.0
                        shot_base_offset = (
                            shot_start_offsets[i]
                            if shot_start_offsets and i < len(shot_start_offsets)
                            else fallback_offset
                        )
                        if dialogues:
                            spacing = (shot_len - 0.8) / len(dialogues) if len(dialogues) > 1 else 0.25
                            for d_idx, (speaker, spoken) in enumerate(dialogues):
                                voice_to_use, rate, pitch = resolve_voice_and_prosody(
                                    speaker=speaker,
                                    characters=characters,
                                    provider=provider,
                                    default_voice=audio_voice(workspace_root),
                                    character_voices=char_voice_map,
                                )
                                shot_voice_file = os.path.join(self.working_dir, "shots", f"{sd.idx}", f"speech_{d_idx}.mp3")
                                await synthesize_speech(
                                    text=spoken,
                                    output_path=shot_voice_file,
                                    provider=provider,
                                    api_key=audio_api_key(workspace_root),
                                    voice=voice_to_use,
                                    model=audio_model(workspace_root),
                                    base_url=audio_base_url(workspace_root),
                                    rate=rate,
                                    pitch=pitch,
                                )
                                diag_offset = shot_base_offset + (0.25 if len(dialogues) == 1 else 0.25 + d_idx * spacing)
                                speech_segments.append((diag_offset, shot_voice_file))

                        fallback_offset += shot_len

                    if speech_segments or bgm_resolved:
                        _emit_render_progress(progress, "audio_mix", "Mixing multi-character voiceover & BGM into video", {"dialogue_count": len(speech_segments)})
                        mix_multitrack_video_and_audio(
                            video_path=final_video_path,
                            output_path=final_video_path,
                            speech_segments=speech_segments,
                            bgm_path=bgm_resolved,
                            bgm_volume=vol,
                            preserve_video_audio=True,
                        )
                        _emit_render_progress(progress, "audio_done", "Multi-character voiceover and music mixed successfully")
            except Exception as e:
                print(f"⚠️ Audio synthesis/mixing warning: {e}")
                logging.warning(f"Audio synthesis/mixing warning: {e}")
                _emit_render_progress(progress, "audio_warning", f"Audio synthesis/mixing skipped: {e}", {"error": str(e)})

        # --- VIETSUB / MULTILINGUAL SUBTITLES GENERATION & BURN-IN ---
        if subtitles_enabled(workspace_root):
            try:
                sub_lang = subtitles_language(workspace_root).strip().lower()
                burn_in = subtitles_burn_in(workspace_root)
                font = subtitles_font(workspace_root)
                font_size = subtitles_font_size(workspace_root)
                suffix = "vietsub" if sub_lang in {"vi", "vn", "vietnamese"} else sub_lang
                srt_path = os.path.join(self.working_dir, f"final_video_{suffix}.srt")
                ass_path = os.path.join(self.working_dir, f"final_video_{suffix}.ass")
                marker_path = _subtitle_burn_marker_path(self.working_dir, suffix)
                ass_existed_before = os.path.exists(ass_path)
                cues = []

                if not (final_video_existed and os.path.exists(srt_path) and ass_existed_before):
                    effective_script = (
                        script
                        if isinstance(script, str)
                        else json.dumps(script, ensure_ascii=False, indent=2)
                        if script is not None
                        else ""
                    )
                    script_txt_path = os.path.join(self.working_dir, "script.txt")
                    if "vietsub:" not in effective_script.lower() and os.path.exists(script_txt_path):
                        effective_script = Path(script_txt_path).read_text(encoding="utf-8")

                    if sub_lang in {"vi", "vn", "vietnamese"}:
                        script_vietsubs = _resolve_script_vietsubs(shot_descriptions, effective_script)
                    else:
                        script_vietsubs = {}

                    pending_translations: Dict[int, str] = {}
                    for i, sd in enumerate(shot_descriptions):
                        sd_idx = getattr(sd, "idx", i)
                        if sd_idx in script_vietsubs or i in script_vietsubs:
                            continue
                        spoken_lines = extract_spoken_dialogues_for_subtitles(
                            getattr(sd, "audio_desc", ""),
                            getattr(sd, "visual_desc", ""),
                        )
                        if spoken_lines:
                            pending_translations[sd_idx] = " ".join(line for _, line in spoken_lines)

                    translated_dialogues: Dict[int, str] = {}
                    if pending_translations:
                        target_lang_name = "Vietnamese" if sub_lang in {"vi", "vn", "vietnamese"} else sub_lang
                        translated_dialogues = await translate_subtitles_with_chat_model(
                            self.chat_model,
                            pending_translations,
                            target_language=target_lang_name,
                        )

                    if shot_durations is None:
                        shot_durations = []
                        for sd in shot_descriptions:
                            shot_mp4 = os.path.join(self.working_dir, "shots", f"{sd.idx}", "video.mp4")
                            if os.path.exists(shot_mp4):
                                clip = None
                                try:
                                    clip = VideoFileClip(shot_mp4)
                                    shot_durations.append(float(clip.duration or 8.0))
                                except Exception:
                                    shot_durations.append(8.0)
                                finally:
                                    if clip is not None:
                                        clip.close()
                            else:
                                shot_durations.append(8.0)

                    cues = build_subtitle_cues(
                        shot_descriptions=shot_descriptions,
                        shot_durations=shot_durations or [8.0] * len(shot_descriptions),
                        script_vietsubs=script_vietsubs,
                        translated_dialogues=translated_dialogues,
                        shot_start_offsets=shot_start_offsets,
                    )
                    if cues:
                        write_srt_subtitles(cues, srt_path)
                        write_ass_subtitles(cues, ass_path, font=font, font_size=font_size)

                if os.path.exists(ass_path):
                    ass_sha256 = _file_sha256(ass_path)
                    cue_count = len(cues) if cues else sum(
                        1 for line in Path(ass_path).read_text(encoding="utf-8").splitlines()
                        if line.startswith("Dialogue:")
                    )
                    already_burned = False
                    if final_video_existed:
                        if marker_path.exists():
                            try:
                                marker_data = json.loads(marker_path.read_text(encoding="utf-8"))
                                if isinstance(marker_data, dict) and marker_data.get("ass_sha256") == ass_sha256:
                                    recorded_out = marker_data.get("output_path")
                                    if recorded_out:
                                        candidate_out = Path(self.working_dir) / recorded_out
                                        if candidate_out.is_file():
                                            if burn_in:
                                                final_video_path = str(candidate_out)
                                            already_burned = True
                                    elif os.path.exists(final_video_path):
                                        already_burned = True
                            except (OSError, ValueError):
                                pass
                        if not already_burned and ass_existed_before and os.path.exists(final_video_path):
                            fv_stat = Path(final_video_path).stat()
                            ass_stat = Path(ass_path).stat()
                            if fv_stat.st_mtime_ns > ass_stat.st_mtime_ns + 5_000_000_000:
                                already_burned = True

                    if burn_in and not already_burned:
                        _emit_render_progress(progress, "subtitles_burn_start", "Burning subtitles into final video", {"cue_count": cue_count, "ass_path": ass_path})
                        burned_path = await asyncio.to_thread(burn_subtitles_into_video, final_video_path, ass_path, final_video_path)
                        final_video_path = burned_path
                        marker_payload = {
                            "ass_sha256": ass_sha256,
                            "ass_path": Path(ass_path).name,
                            "output_path": Path(final_video_path).name,
                        }
                        marker_path.write_text(json.dumps(marker_payload, ensure_ascii=False, indent=2), encoding="utf-8")

                    _emit_render_progress(
                        progress,
                        "subtitles_done",
                        "Generated subtitles successfully",
                        {
                            "cue_count": cue_count,
                            "srt_path": srt_path,
                            "ass_path": ass_path,
                            "burned_in": burn_in,
                            "output_path": final_video_path,
                        },
                    )
            except Exception as e:
                print(f"⚠️ Subtitle generation/burn-in warning: {e}")
                logging.warning(f"Subtitle generation/burn-in warning: {e}")
                _emit_render_progress(progress, "subtitles_warning", f"Subtitle generation/burn-in skipped: {e}", {"error": str(e)})

        _emit_render_progress(progress, "render_done", "Script2video render complete", {"final_video_path": final_video_path})
        return final_video_path


    async def _resolve_continuity_reference(
        self,
        shot_descriptions: List[ShotDescription],
        shot_idx: int,
    ) -> Optional[Tuple[str, str]]:
        shot_by_idx = {sd.idx: sd for sd in shot_descriptions}
        shot_desc = shot_by_idx.get(shot_idx)
        if shot_desc is None or not bool(getattr(shot_desc, "continuity_from_prev", False)):
            return None
        prev_shot = _previous_shot_description(shot_descriptions, shot_idx)
        if prev_shot is None:
            return None
        prev_events = self.frame_events.get(prev_shot.idx, {})
        prev_needs_lf = shot_requires_last_frame(shot_descriptions, prev_shot)
        if prev_needs_lf and "last_frame" in prev_events:
            await prev_events["last_frame"].wait()
            candidate_path = os.path.join(self.working_dir, "shots", f"{prev_shot.idx}", "last_frame.png")
            prev_text = prev_shot.lf_desc or prev_shot.visual_desc
            if not os.path.exists(candidate_path):
                candidate_path = os.path.join(self.working_dir, "shots", f"{prev_shot.idx}", "first_frame.png")
                prev_text = prev_shot.ff_desc or prev_shot.visual_desc
        elif "first_frame" in prev_events:
            await prev_events["first_frame"].wait()
            candidate_path = os.path.join(self.working_dir, "shots", f"{prev_shot.idx}", "first_frame.png")
            prev_text = prev_shot.ff_desc or prev_shot.visual_desc
        else:
            return None
        if not os.path.exists(candidate_path):
            return None
        return (
            candidate_path,
            f"CONTINUOUS ACTION ANCHOR (exact ending keyframe from preceding Shot {prev_shot.idx}: {prev_text}). "
            f"You MUST preserve environment lighting, battered/torn clothing wear, spatial screen direction, and exact physical momentum into this opening frame, adapting camera scale/angle as specified in the target description.",
        )

    async def generate_frames_for_single_camera(
        self,
        camera: Camera,
        shot_descriptions: List[ShotDescription],
        characters: List[CharacterInScene],
        character_portraits_registry: Dict[str, Dict[str, Dict[str, str]]],
        priority_shot_idxs: List[int],
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
        shot_panel_refs: Optional[Dict[int, List[Tuple[str, str]]]] = None,
        allow_transition_video: bool = True,
    ):
        if not camera.active_shot_idxs:
            return
        shot_by_idx = {sd.idx: sd for sd in shot_descriptions}
        char_by_idx = {c.idx: c for c in characters}
        panel_refs_map = _normalize_shot_panel_refs(shot_panel_refs, require_exists=False) if shot_panel_refs else {}

        # 1. generate the first_frame of the first shot of the camera
        first_shot_idx = camera.active_shot_idxs[0]
        first_shot_desc = shot_by_idx[first_shot_idx]
        first_shot_ff_path = os.path.join(self.working_dir, "shots", f"{first_shot_idx}", "first_frame.png")
        _emit_render_progress(progress, "camera_frames_start", f"Generating frames for camera {camera.idx}", {"camera_idx": camera.idx, "active_shot_idxs": camera.active_shot_idxs})

        if os.path.exists(first_shot_ff_path):
            print(f"🚀 Skipped generating first_frame for shot {first_shot_idx}, already exists.")
            self.frame_events[first_shot_idx]["first_frame"].set()
            _emit_render_progress(progress, "frame_exists", f"First frame for shot {first_shot_idx} already exists", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "frame_type": "first_frame", "path": first_shot_ff_path})

        else:
            print(f"🖼️ Starting first_frame generation for shot {first_shot_idx}...")
            _emit_render_progress(progress, "frame_start", f"Generating first frame for shot {first_shot_idx}", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "frame_type": "first_frame"})
            available_image_path_and_text_pairs = []

            for character_idx in first_shot_desc.ff_vis_char_idxs:
                char_obj = char_by_idx.get(character_idx)
                if char_obj is None or not getattr(char_obj, "is_visible", True):
                    continue
                identifier_in_scene = char_obj.identifier_in_scene
                registry_item = character_portraits_registry.get(identifier_in_scene) or {}
                for view, item in registry_item.items():
                    available_image_path_and_text_pairs.append((item["path"], item["description"]))

            first_panel_pairs = list(panel_refs_map.get(first_shot_idx) or [])[:2]
            for panel_pair in first_panel_pairs:
                available_image_path_and_text_pairs.append(panel_pair)

            continuity_pair = await self._resolve_continuity_reference(shot_descriptions, first_shot_idx)
            new_camera_image_path = None
            if continuity_pair is not None:
                available_image_path_and_text_pairs.append(continuity_pair)
            elif (first_panel_pairs or not allow_transition_video) and camera.parent_shot_idx is not None:
                parent_shot_idx = camera.parent_shot_idx
                parent_ev = self.frame_events.get(parent_shot_idx, {}).get("first_frame")
                if parent_ev is not None:
                    await parent_ev.wait()
                parent_shot_ff_path = os.path.join(self.working_dir, "shots", f"{parent_shot_idx}", "first_frame.png")
                if os.path.exists(parent_shot_ff_path):
                    available_image_path_and_text_pairs.append(
                        (
                            parent_shot_ff_path,
                            f"SCENE LIGHTING & ANIME STYLE CONTINUITY REFERENCE from parent camera Shot {parent_shot_idx} "
                            "(match night/day lighting, weather, and anime palette while strictly following the ORIGINAL COMIC PANEL COMPOSITION & POSE REFERENCE and character portraits for this shot).",
                        )
                    )
            elif camera.parent_shot_idx is not None:
                # generate the first_frame based on the transition video
                parent_shot_idx = camera.parent_shot_idx
                await self.frame_events[parent_shot_idx]["first_frame"].wait()
                parent_shot_ff_path = os.path.join(self.working_dir, "shots", f"{parent_shot_idx}", "first_frame.png")
                transition_video_path = os.path.join(self.working_dir, "shots", f"{first_shot_idx}", f"transition_video_from_shot_{parent_shot_idx}.mp4")

                if os.path.exists(transition_video_path):
                    print(f"🚀 Skipped generating transition video for shot {first_shot_idx} from shot {parent_shot_idx}, already exists.")
                    _emit_render_progress(progress, "transition_video_exists", f"Transition video for shot {first_shot_idx} already exists", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "parent_shot_idx": parent_shot_idx, "path": transition_video_path})
                else:
                    print(f"🖼️ Starting transition video generation for shot {first_shot_idx} from shot {parent_shot_idx}...")
                    _emit_render_progress(progress, "transition_video_start", f"Generating transition video for shot {first_shot_idx}", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "parent_shot_idx": parent_shot_idx})
                    workspace_root = _resolve_workspace_root(self.working_dir)
                    suppress_subs = subtitles_enabled(workspace_root) and subtitles_burn_in(workspace_root)
                    parent_shot_desc = shot_by_idx[parent_shot_idx]
                    trans_kwargs: Dict[str, Any] = {
                        "first_shot_visual_desc": parent_shot_desc.visual_desc,
                        "second_shot_visual_desc": first_shot_desc.visual_desc,
                        "first_shot_ff_path": parent_shot_ff_path,
                        "progress": _scoped_progress(progress, camera_idx=camera.idx, shot_idx=first_shot_idx, parent_shot_idx=parent_shot_idx, artifact="transition_video"),
                    }
                    try:
                        sig = inspect.signature(self.camera_image_generator.generate_transition_video)
                        if "suppress_on_screen_subtitles" in sig.parameters:
                            trans_kwargs["suppress_on_screen_subtitles"] = suppress_subs
                    except (TypeError, ValueError):
                        pass
                    transition_video_output = await self.camera_image_generator.generate_transition_video(**trans_kwargs)
                    transition_video_output.save(transition_video_path)
                    print(f"☑️ Generated transition video for shot {first_shot_idx} from shot {parent_shot_idx}, saved to {transition_video_path}.")
                    _emit_render_progress(progress, "transition_video_done", f"Transition video for shot {first_shot_idx} generated", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "parent_shot_idx": parent_shot_idx, "path": transition_video_path})

                new_camera_image_path = os.path.join(self.working_dir, "shots", f"{first_shot_idx}", f"new_camera_{camera.idx}.png")
                if os.path.exists(new_camera_image_path):
                    print(f"🚀 Skipped generating new camera image for shot {first_shot_idx}, already exists.")
                    _emit_render_progress(progress, "new_camera_image_exists", f"New camera image for shot {first_shot_idx} already exists", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "path": new_camera_image_path})
                else:
                    print(f"🖼️ Starting new camera image generation for shot {first_shot_idx}...")
                    _emit_render_progress(progress, "new_camera_image_start", f"Extracting new camera image for shot {first_shot_idx}", {"camera_idx": camera.idx, "shot_idx": first_shot_idx})
                    new_camera_image = self.camera_image_generator.get_new_camera_image(transition_video_path)
                    new_camera_image.save(new_camera_image_path)
                    print(f"☑️ Generated new camera image for shot {first_shot_idx} (not completed), saved to {new_camera_image_path}.")
                    _emit_render_progress(progress, "new_camera_image_done", f"New camera image for shot {first_shot_idx} extracted", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "path": new_camera_image_path})

                available_image_path_and_text_pairs.append(
                    (
                        new_camera_image_path,
                        f"The composition and background are correct but some elements may be wrong. The wrong elements should be replaced.\nWrong elements: {camera.missing_info}.\nYou must select this image as the main reference and replace the characters in the image with the provided character portraits. Don't change the background."
                    )
                )

            if continuity_pair is not None or first_panel_pairs or camera.parent_shot_idx is None or not allow_transition_video or camera.missing_info is not None:
                ff_selector_output_path = os.path.join(self.working_dir, "shots", f"{first_shot_idx}", "first_frame_selector_output.json")
                if os.path.exists(ff_selector_output_path):
                    with open(ff_selector_output_path, 'r', encoding='utf-8') as f:
                        ff_selector_output = json.load(f)
                    print(f"🚀 Loaded existing reference image selection and prompt for first_frame of shot {first_shot_idx} from {ff_selector_output_path}.")
                    _emit_render_progress(progress, "frame_prompt_exists", f"First frame prompt for shot {first_shot_idx} already exists", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "frame_type": "first_frame", "path": ff_selector_output_path})
                else:
                    print(f"🔍 Selecting reference images and generating prompt for first_frame of shot {first_shot_idx}...")
                    _emit_render_progress(progress, "frame_prompt_start", f"Selecting references for first frame of shot {first_shot_idx}", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "frame_type": "first_frame"})
                    _ws_ref = _resolve_workspace_root(self.working_dir)
                    _ref_policy = quality_ref_policy(_ws_ref)
                    _max_ref = quality_max_reference_images(_ws_ref)
                    available_image_path_and_text_pairs = _dedupe_image_text_pairs(available_image_path_and_text_pairs)
                    _cand_cap = 4 if _ref_policy == "legacy" else max(4, _max_ref)
                    available_image_path_and_text_pairs = available_image_path_and_text_pairs[:_cand_cap]  # cap refs to avoid muse2api payload hang
                    ff_selector_output = await self.reference_image_selector.select_reference_images_and_generate_prompt(
                        available_image_path_and_text_pairs=available_image_path_and_text_pairs,
                        frame_description=first_shot_desc.ff_desc
                    )
                    os.makedirs(os.path.dirname(ff_selector_output_path), exist_ok=True)
                    with open(ff_selector_output_path, 'w', encoding='utf-8') as f:
                        json.dump(ff_selector_output, f, ensure_ascii=False, indent=4)

                    print(f"☑️ Selected reference images and generated prompt for first_frame of shot {first_shot_idx}, saved to {ff_selector_output_path}.")
                    _emit_render_progress(progress, "frame_prompt_done", f"Selected references for first frame of shot {first_shot_idx}", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "frame_type": "first_frame", "path": ff_selector_output_path})

                raw_pairs = ff_selector_output["reference_image_path_and_text_pairs"]
                prompt = ff_selector_output["text_prompt"]
                vis_chars = [
                    char_by_idx[idx]
                    for idx in first_shot_desc.ff_vis_char_idxs
                    if idx in char_by_idx and getattr(char_by_idx[idx], "is_visible", True)
                ]
                scene_anchor = None
                transition_in_val = (getattr(first_shot_desc, "transition_in", None) or "cut").lower()
                desc_text_lower = " ".join(
                    filter(None, [getattr(first_shot_desc, "ff_desc", ""), getattr(first_shot_desc, "visual_desc", "")])
                ).lower()
                is_alley_night_context = (
                    any(k in desc_text_lower for k in ("alley", "alleyway", "stone wall", "asphalt"))
                    and not any(k in desc_text_lower for k in ("pet shop", "shop interior", "store lighting", "sunset", "sunlit", "warm amber"))
                )
                if transition_in_val != "fade_black":
                    for prev_s in (5, 4, 3):
                        if prev_s < first_shot_idx:
                            cand_end = os.path.join(self.working_dir, "shots", str(prev_s), "video_end.png")
                            cand_p = cand_end if os.path.exists(cand_end) else os.path.join(self.working_dir, "shots", str(prev_s), "first_frame.png")
                            if os.path.exists(cand_p):
                                if is_alley_night_context:
                                    anchor_desc = (
                                        f"Master scene lighting, color grading & architectural anchor from Shot {prev_s}: "
                                        "maintain the exact same moody rainy navy night color grading and cold desaturated alley atmosphere, "
                                        "preserving tonal separation with cool silver-white 3/4 rim/directional highlights from the brighter alley exit "
                                        "against dark vertical stone wall textures, wet asphalt pavement, and subtle glowing cyan sparks. "
                                        "Strictly NO warm incandescent yellow color shifts, NO daytime lighting changes."
                                    )
                                else:
                                    anchor_desc = (
                                        f"Master scene lighting, color grading & environment anchor from Shot {prev_s}: "
                                        "maintain consistent anime palette, lighting, and location atmosphere."
                                    )
                                scene_anchor = (cand_p, anchor_desc)
                                break
                if scene_anchor is None and first_shot_idx != 0 and transition_in_val != "fade_black":
                    scene_shot0_ff = os.path.join(self.working_dir, "shots", "0", "first_frame.png")
                    if os.path.exists(scene_shot0_ff):
                        scene_anchor = (scene_shot0_ff, "Scene establishing environment anchor: maintain the exact same architectural location, background walls, floor/ground textures, time of day, and lighting atmosphere.")

                _ws_ref = _resolve_workspace_root(self.working_dir)
                _ref_policy = quality_ref_policy(_ws_ref)
                _ref_limit = 2 if _ref_policy == "legacy" else quality_max_reference_images(_ws_ref)
                _sheet_dir = None if _ref_policy == "legacy" else os.path.join(self.working_dir, "shots", str(first_shot_idx))
                reference_image_path_and_text_pairs = _ensure_character_portraits_and_spatial_anchors(
                    pairs=raw_pairs,
                    visible_characters=vis_chars,
                    character_portraits_registry=character_portraits_registry,
                    scene_anchor_pair=scene_anchor,
                    limit=_ref_limit,
                    sheet_dir=_sheet_dir,
                    sheet_tag="first_frame",
                )
                prefix_prompt = ""
                for i, (image_path, text) in enumerate(reference_image_path_and_text_pairs):
                    prefix_prompt += f"Image {i}: {text}\n"
                prompt = f"{prefix_prompt}\n{prompt}"
                prompt = _apply_series_enrichment(prompt, self.working_dir, vis_chars)
                reference_image_paths = [item[0] for item in reference_image_path_and_text_pairs]
                _write_refs_used(self.working_dir, first_shot_idx, "first_frame", reference_image_path_and_text_pairs, vis_chars, character_portraits_registry)
                ff_image: ImageOutput = await self.generate_and_select_best_image(
                    prompt=prompt,
                    reference_image_paths=reference_image_paths,
                    reference_image_path_and_text_pairs=reference_image_path_and_text_pairs,
                    target_description=first_shot_desc.ff_desc,
                    size="1600x900",
                    candidates_save_dir=os.path.join(self.working_dir, "shots", str(first_shot_idx), "first_frame_candidates"),
                    progress=_scoped_progress(progress, camera_idx=camera.idx, shot_idx=first_shot_idx, frame_type="first_frame"),
                )
                ff_image.save(first_shot_ff_path)
                self.frame_events[first_shot_idx]["first_frame"].set()
                print(f"☑️ Generated first_frame for shot {first_shot_idx}, saved to {first_shot_ff_path}.")
                _emit_render_progress(progress, "frame_done", f"Generated first frame for shot {first_shot_idx}", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "frame_type": "first_frame", "path": first_shot_ff_path})
            else:
                shutil.copy(new_camera_image_path, first_shot_ff_path)
                self.frame_events[first_shot_idx]["first_frame"].set()
                print(f"☑️ Generated first_frame for shot {first_shot_idx}, saved to {first_shot_ff_path}.")
                _emit_render_progress(progress, "frame_done", f"Generated first frame for shot {first_shot_idx}", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "frame_type": "first_frame", "path": first_shot_ff_path})


        # 2. generate the following frames of the camera in a single acyclic gather
        async def _build_subsequent_first_frame(target_idx: int):
            target_desc = shot_by_idx[target_idx]
            cont_pair = await self._resolve_continuity_reference(shot_descriptions, target_idx)
            extra_pairs: List[Tuple[str, str]] = []
            if cont_pair is not None:
                extra_pairs.append(cont_pair)
            for pair in (panel_refs_map.get(target_idx) or [])[:2]:
                if pair:
                    extra_pairs.append(pair)
            await self.generate_frame_for_single_shot(
                shot_idx=target_idx,
                frame_type="first_frame",
                first_shot_ff_path_and_text_pair=(first_shot_ff_path, first_shot_desc.ff_desc),
                frame_desc=target_desc.ff_desc,
                visible_characters=[
                    char_by_idx[idx]
                    for idx in target_desc.ff_vis_char_idxs
                    if idx in char_by_idx and getattr(char_by_idx[idx], "is_visible", True)
                ],
                character_portraits_registry=character_portraits_registry,
                progress=progress,
                extra_reference_pairs=extra_pairs or None,
            )

        async def _build_shot_last_frame(target_idx: int):
            target_desc = shot_by_idx[target_idx]
            await self.frame_events[target_idx]["first_frame"].wait()
            shot_ff_path = os.path.join(self.working_dir, "shots", f"{target_idx}", "first_frame.png")
            anchor_pair = (
                (shot_ff_path, target_desc.ff_desc)
                if os.path.exists(shot_ff_path)
                else (first_shot_ff_path, first_shot_desc.ff_desc)
            )
            lf_panel_pairs = list(panel_refs_map.get(target_idx) or [])[:2]
            await self.generate_frame_for_single_shot(
                shot_idx=target_idx,
                frame_type="last_frame",
                first_shot_ff_path_and_text_pair=anchor_pair,
                frame_desc=target_desc.lf_desc,
                visible_characters=[
                    char_by_idx[idx]
                    for idx in target_desc.lf_vis_char_idxs
                    if idx in char_by_idx and getattr(char_by_idx[idx], "is_visible", True)
                ],
                character_portraits_registry=character_portraits_registry,
                progress=progress,
                extra_reference_pairs=lf_panel_pairs or None,
            )

        priority_tasks = []
        normal_tasks = []

        if shot_requires_last_frame(shot_descriptions, first_shot_desc):
            normal_tasks.append(_build_shot_last_frame(first_shot_idx))

        for shot_idx in camera.active_shot_idxs[1:]:
            s_desc = shot_by_idx[shot_idx]
            first_frame_coro = _build_subsequent_first_frame(shot_idx)
            if shot_idx in priority_shot_idxs:
                priority_tasks.append(first_frame_coro)
            else:
                normal_tasks.append(first_frame_coro)

            if shot_requires_last_frame(shot_descriptions, s_desc):
                normal_tasks.append(_build_shot_last_frame(shot_idx))

        await asyncio.gather(*(priority_tasks + normal_tasks))
        _emit_render_progress(progress, "camera_frames_done", f"Frames for camera {camera.idx} ready", {"camera_idx": camera.idx, "active_shot_idxs": camera.active_shot_idxs})



    async def generate_video_for_single_shot(
        self,
        shot_description: ShotDescription,
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
    ):
        video_path = os.path.join(self.working_dir, "shots", f"{shot_description.idx}", "video.mp4")
        if os.path.exists(video_path):
            print(f"🚀 Skipped generating video for shot {shot_description.idx}, already exists.")
            _emit_render_progress(progress, "video_clip_exists", f"Video clip for shot {shot_description.idx} already exists", {"shot_idx": shot_description.idx, "path": video_path})
        else:
            _emit_render_progress(progress, "video_clip_waiting_for_frames", f"Waiting for frames before video clip {shot_description.idx}", {"shot_idx": shot_description.idx})
            await self.frame_events[shot_description.idx]["first_frame"].wait()
            shot_ev_map = self.frame_events.get(shot_description.idx, {})
            needs_last_frame = shot_requires_last_frame(
                getattr(self, "_active_shot_descriptions", None) or [shot_description],
                shot_description,
            )
            if needs_last_frame and "last_frame" in shot_ev_map:
                await shot_ev_map["last_frame"].wait()

            frame_paths = []
            frame_paths.append(os.path.join(self.working_dir, "shots", f"{shot_description.idx}", "first_frame.png"))
            last_frame_path = os.path.join(self.working_dir, "shots", f"{shot_description.idx}", "last_frame.png")
            if needs_last_frame and os.path.exists(last_frame_path):
                frame_paths.append(last_frame_path)

            print(f"🎬 Starting video generation for shot {shot_description.idx}...")
            _emit_render_progress(progress, "video_clip_start", f"Generating video clip for shot {shot_description.idx}", {"shot_idx": shot_description.idx, "frame_count": len(frame_paths)})
            workspace_root = _resolve_workspace_root(self.working_dir)
            suppress_subs = subtitles_enabled(workspace_root) and subtitles_burn_in(workspace_root)
            clean_audio_desc = re.sub(
                r"\(\s*Vietsub\s*:[^)]*\)",
                "",
                getattr(shot_description, "audio_desc", "") or "",
                flags=re.IGNORECASE,
            )
            clean_audio_desc = re.sub(
                r"^\s*Vietsub\s*:[^\n]*\n?",
                "",
                clean_audio_desc,
                flags=re.IGNORECASE | re.MULTILINE,
            ).strip()
            motion_clean = (shot_description.motion_desc or "").strip()
            from agents.shot_linter import sanitize_motion_prompt
            motion_clean = sanitize_motion_prompt(motion_clean)
            characters_data = []
            chars_json_path = os.path.join(self.working_dir, "characters.json")
            if os.path.exists(chars_json_path):
                try:
                    with open(chars_json_path, "r", encoding="utf-8") as f:
                        characters_data = json.load(f)
                except Exception:
                    characters_data = []

            has_characters = bool(getattr(shot_description, "ff_vis_char_idxs", None) or getattr(shot_description, "lf_vis_char_idxs", None))
            if not has_characters and characters_data:
                has_characters = bool(re.search(r"<[^>]+>", shot_description.visual_desc or ""))

            has_two_keyframes = len(frame_paths) >= 2
            is_back_view = False
            if not has_characters:
                keyframe_note = (
                    "Animate strictly starting from Image 0 (exact opening keyframe). Pure scenic, architectural, and environmental scene: "
                    "preserve the exact background composition, sky, houses, trees, lighting, and objects from Image 0 throughout the entire clip. "
                    "Strictly DO NOT add any characters, DO NOT introduce any people or humans into the scene."
                )
                facial_rule = ""
                aesthetic_rule = (
                    "[STYLIZED ANIME AESTHETIC]: Clean modern 2D Japanese anime environmental animation. "
                    "Focus on atmospheric night lighting, gentle wind, sky effects, and smooth cinematic pacing."
                )
            else:
                keyframe_note = (
                    "Animate seamlessly starting from Image 0 (exact opening frame) through the full movement to land precisely on Image 1 (exact final keyframe). Maintain consistent anime character proportions, wardrobe, battle wounds, bloodstains, lighting, and background across the whole clip."
                    if has_two_keyframes
                    else "Animate naturally starting directly from Image 0 (exact opening keyframe). Maintain strict visual consistency with Image 0: preserve the exact character appearance, outfit, hairstyle, battle wounds, bloodstains, and lighting throughout the entire clip without altering clothing, healing injuries, or hallucinating unintended inner garments."
                )
                desc_text = f"{getattr(shot_description, 'visual_desc', '')} {getattr(shot_description, 'motion_desc', '')}".lower()
                has_injuries = bool(re.search(
                    r"\b(wound|wounded|injury|injured|bleed|bleeding|blood|bloodied|bruise|bruised|scraped|scratched|cut|cuts|slashed|battle wear|battle damage|torn)\b",
                    desc_text,
                ))
                is_back_view = bool(re.search(
                    r"\b(seen from behind|from behind|back view|rear view|behind|head bowed|back turned|facing away|facing wall|over-the-shoulder|back of|looking down|obscured face|face obscured|insert shot|tightly focused on.*?hand|focused on.*?hand|grounded hand|close-up on.*?hand|hand pressed|knuckles|foot|feet|boots)\b",
                    desc_text,
                ))

                injury_persistence_rule = (
                    "[BATTLE INJURY & DAMAGE PERSISTENCE]: Strictly preserve all visible battle wounds, cuts, scrapes, bruises, "
                    "blood streaks, and torn battle-damaged clothing from Image 0 consistently across all intermediate frames. "
                    "DO NOT magically heal, clean, erase, or smooth over any wounds or bloodstains."
                    if has_injuries
                    else ""
                )

                if is_back_view:
                    facial_rule = (
                        "[STRICT FRAMING & POSE LOCK]: Locked static perspective directly matching Image 0. "
                        "The character remains completely motionless facing away with head bowed down toward the ground throughout the shot. "
                        "Strictly preserve the exact rear framing, body posture, hands, and attire from Image 0 throughout the entire clip."
                    )
                else:
                    facial_rule = (
                        "[MANDATORY CHARACTER CONSISTENCY]: Preserve exact character proportions, facial features, "
                        "hair, battle injuries, and clothing strictly matching Image 0 reference across all animated frames."
                    )
                _series_style = ""
                if quality_generic_linter(_resolve_workspace_root(self.working_dir)):
                    _series_style = load_configured_series(_resolve_workspace_root(self.working_dir)).style_prompt()
                aesthetic_rule = _series_style or (
                    "[STYLIZED 2D ANIME AESTHETIC]: Clean 2D anime / manhwa style with crisp digital ink outlines and smooth flat "
                    "cel shading. Strictly NO 3D rendering, NO photorealism, NO painterly brush noise. Persistent stylized 2D battle "
                    "damage. Directly generate the visual animation starting seamlessly from Image 0."
                )

            has_spoken_audio = bool(re.search(r"\[(?:Speaker|Thought|Inner\s*Monologue)\]", clean_audio_desc, re.IGNORECASE))
            timing_note = (
                "\n(Spoken Delivery Pacing: start speaking promptly near 0.3s and complete every word cleanly near 70% of the clip duration, followed by at least 1.5 seconds of lingering ambient silence, breathing, and reaction before the shot ends.)"
                if has_spoken_audio
                else ""
            )
            formatted_audio = _format_video_prompt_audio(clean_audio_desc, characters_data, timing_note, is_back_view=is_back_view)
            if suppress_subs:
                visual_body = "\n".join(part for part in (keyframe_note, motion_clean, injury_persistence_rule, facial_rule, aesthetic_rule) if part).strip()
                sections = []
                if visual_body:
                    sections.append(
                        f"[Pure Visual Scene - Clean Textless Borderless Animation Only]\n{visual_body}"
                    )
                if formatted_audio:
                    sections.append(formatted_audio)
                video_prompt = "\n\n".join(sections).strip()
            else:
                visual_body = "\n".join(part for part in (keyframe_note, motion_clean, injury_persistence_rule, facial_rule, aesthetic_rule) if part).strip()
                video_prompt = "\n\n".join(part for part in (visual_body, formatted_audio) if part).strip()
            gen_kwargs: Dict[str, Any] = {
                "prompt": video_prompt,
                "reference_image_paths": frame_paths,
                "progress": _scoped_progress(progress, shot_idx=shot_description.idx, artifact="video_clip"),
            }
            target_dur = getattr(shot_description, "target_duration_sec", None)
            if target_dur is not None:
                try:
                    dur_val = float(target_dur)
                    if dur_val > 0:
                        gen_kwargs["duration"] = dur_val
                except (ValueError, TypeError):
                    pass
            try:
                sig = inspect.signature(self.video_generator.generate_single_video)
                params = sig.parameters
                if "suppress_on_screen_subtitles" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                    gen_kwargs["suppress_on_screen_subtitles"] = suppress_subs
                if "duration" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                    if target_dur is not None:
                        gen_kwargs["duration"] = target_dur
            except (TypeError, ValueError):
                pass
            video_output = await self.video_generator.generate_single_video(**gen_kwargs)
            video_output.save(video_path)
            from utils.video import conform_shot_video_duration
            conform_shot_video_duration(video_path, shot_description)

            # Per-shot burn-in of cinema-grade anime Vietsub if enabled
            if suppress_subs:
                try:
                    from utils.audio import burn_shot_subtitle
                    burn_shot_subtitle(
                        video_path=video_path,
                        shot_description=shot_description,
                        font=subtitles_font(workspace_root),
                        font_size=subtitles_font_size(workspace_root),
                    )
                except Exception as exc:
                    logging.warning(f"Per-shot subtitle burn-in skipped for shot {shot_description.idx}: {exc}")

            print(f"☑️ Generated video for shot {shot_description.idx}, saved to {video_path}.")
            _emit_render_progress(progress, "video_clip_done", f"Generated video clip for shot {shot_description.idx}", {"shot_idx": shot_description.idx, "path": video_path})

    async def generate_and_select_best_image(
        self,
        prompt: str,
        reference_image_paths: List[str],
        reference_image_path_and_text_pairs: List[Tuple[str, str]],
        target_description: str,
        size: str = "1600x900",
        num_candidates: int | None = None,
        candidates_save_dir: str | None = None,
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
    ) -> ImageOutput:
        count = validate_image_candidate_count(self.num_image_candidates if num_candidates is None else num_candidates)
        if count == 1:
            gen_kwargs: Dict[str, Any] = {
                "prompt": prompt + " Generate a single static image only. No video, no animation, no motion, no GIF. Output a static PNG image.",
                "reference_image_paths": reference_image_paths,
                "size": size,
            }
            if progress is not None:
                try:
                    sig = inspect.signature(self.image_generator.generate_single_image)
                    params = sig.parameters
                    if "progress" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                        gen_kwargs["progress"] = progress
                except (TypeError, ValueError):
                    pass
            return await self.image_generator.generate_single_image(**gen_kwargs)

        save_dir = Path(candidates_save_dir) if candidates_save_dir else Path(self.working_dir) / "image_candidates" / uuid4().hex
        save_dir.mkdir(parents=True, exist_ok=True)
        selection_path = save_dir / "selection.json"
        references = []
        for path, description in reference_image_path_and_text_pairs:
            stat = os.stat(path)
            references.append((path, description, stat.st_size, stat.st_mtime_ns))
        fingerprint = hashlib.sha256(json.dumps({
            "prompt": prompt, "size": size, "references": references,
            "reference_image_paths": reference_image_paths, "target_description": target_description,
            "generator": type(self.image_generator).__name__,
            "model": str(getattr(self.image_generator, "model", "")),
        }, sort_keys=True).encode()).hexdigest()
        previous = {}
        if selection_path.exists():
            try:
                previous = json.loads(selection_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        reuse = isinstance(previous, dict) and previous.get("request_fingerprint") == fingerprint
        reusable_indices = {
            item["index"] for item in previous.get("candidates", [])
            if isinstance(item, dict) and item.get("status") == "ready" and "index" in item
        } if reuse else set()
        selection = {"request_fingerprint": fingerprint, "requested_candidates": count, "status": "generating", "candidates": []}

        def save_selection():
            temporary = selection_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(selection_path)

        save_selection()
        _emit_render_progress(progress, "image_candidates_start", f"Generating {count} image candidates", {"count": count})

        async def generate_candidate(index):
            path = save_dir / f"candidate_{index}.png"
            record = {"index": index, "path": path.name, "status": "generating"}
            selection["candidates"].append(record)
            try:
                if index in reusable_indices and path.is_file():
                    record["reused"] = True
                else:
                    output = await self.image_generator.generate_single_image(
                        prompt=prompt + " Generate a single static image only. No video, no animation, no motion, no GIF. Output a static PNG image.", reference_image_paths=reference_image_paths, size=size,
                    )
                    if output is None:
                        raise RuntimeError("Image provider returned no image")
                    await asyncio.to_thread(output.save, str(path))
                with Image.open(path) as image:
                    width, height = image.size
                    output = ImageOutput(fmt="pil", ext="png", data=image.copy())
                record.update(width=width, height=height)
                if width <= height:
                    raise ValueError("Candidate is not landscape")
                record["status"] = "ready"
                save_selection()
                _emit_render_progress(progress, "image_candidate_done", f"Image candidate {index + 1}/{count} ready", {"index": index, "count": count})
                return index, output, str(path)
            except Exception as exc:
                record.update(status="error", error_type=type(exc).__name__)
                save_selection()
                _emit_render_progress(progress, "image_candidate_failed", f"Image candidate {index + 1}/{count} failed", {"index": index, "error_type": type(exc).__name__})
                raise

        tasks = [asyncio.create_task(generate_candidate(index)) for index in range(count)]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            cancelled = next((result for result in results if isinstance(result, asyncio.CancelledError)), None)
            if cancelled is not None:
                raise cancelled
            valid = [result for result in results if not isinstance(result, Exception)]
            if not valid:
                selection["status"] = "generation_failed"
                save_selection()
                raise RuntimeError(f"All {count} image candidates failed: {results[0]}") from results[0]

            if len(valid) == 1:
                selected = valid[0]
                reason = "Only one valid landscape candidate; VLM comparison skipped."
                selection["selection_method"] = "single_valid_candidate"
            else:
                selection["status"] = "selecting"
                save_selection()
                _emit_render_progress(progress, "image_selection_start", f"VLM selecting from {len(valid)} candidates", {"count": len(valid)})
                try:
                    response = await self.best_image_selector.select(
                        reference_image_path_and_text_pairs=reference_image_path_and_text_pairs,
                        target_description=target_description,
                        candidate_image_paths=[item[2] for item in valid],
                    )
                except Exception as exc:
                    selection.update(status="selection_failed", error_type=type(exc).__name__)
                    save_selection()
                    raise
                selected = valid[response.best_image_index]
                reason = response.reason
                selection["selection_method"] = "vlm"
            selection.update(status="selected", selected_candidate_index=selected[0], reason=reason)
            save_selection()
            _emit_render_progress(progress, "image_selection_done", f"Selected image candidate {selected[0] + 1}", {"index": selected[0], "count": len(valid), "selection_method": selection["selection_method"]})
            return selected[1]
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            selection["status"] = "cancelled"
            save_selection()
            raise


    async def generate_frame_for_single_shot(
        self,
        shot_idx: int,
        frame_type: Literal["first_frame", "last_frame"],
        first_shot_ff_path_and_text_pair: Tuple[str, str],
        frame_desc: str,
        visible_characters: List[CharacterInScene],
        character_portraits_registry: Dict[str, Dict[str, Dict[str, str]]],
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
        extra_reference_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> ImageOutput:

        frame_image_path = os.path.join(self.working_dir, "shots", f"{shot_idx}", f"{frame_type}.png")

        if os.path.exists(frame_image_path):
            print(f"🚀 Skipped generating {frame_type} for shot {shot_idx}, already exists.")
            _emit_render_progress(progress, "frame_exists", f"{frame_type} for shot {shot_idx} already exists", {"shot_idx": shot_idx, "frame_type": frame_type, "path": frame_image_path})

        else:
            print(f"🖼️ Starting {frame_type} generation for shot {shot_idx}...")
            _emit_render_progress(progress, "frame_start", f"Generating {frame_type} for shot {shot_idx}", {"shot_idx": shot_idx, "frame_type": frame_type})
            available_image_path_and_text_pairs = []
            for visible_character in visible_characters:
                if not getattr(visible_character, "is_visible", True):
                    continue
                identifier_in_scene = visible_character.identifier_in_scene
                registry_item = character_portraits_registry.get(identifier_in_scene) or {}
                for view, item in registry_item.items():
                    available_image_path_and_text_pairs.append((item["path"], item["description"]))

            available_image_path_and_text_pairs.append(first_shot_ff_path_and_text_pair)
            if extra_reference_pairs:
                for pair in extra_reference_pairs:
                    if pair and pair[0] != first_shot_ff_path_and_text_pair[0]:
                        available_image_path_and_text_pairs.append(pair)

            selector_output_path = os.path.join(self.working_dir, "shots", f"{shot_idx}", f"{frame_type}_selector_output.json")
            if os.path.exists(selector_output_path):
                with open(selector_output_path, 'r', encoding='utf-8') as f:
                    selector_output = json.load(f)
                print(f"🚀 Loaded existing reference image selection and prompt for {frame_type} frame of shot {shot_idx} from {selector_output_path}.")
                _emit_render_progress(progress, "frame_prompt_exists", f"Prompt for {frame_type} of shot {shot_idx} already exists", {"shot_idx": shot_idx, "frame_type": frame_type, "path": selector_output_path})
            else:
                print(f"🔍 Selecting reference images and generating prompt for {frame_type} frame of shot {shot_idx}...")
                _emit_render_progress(progress, "frame_prompt_start", f"Selecting references for {frame_type} of shot {shot_idx}", {"shot_idx": shot_idx, "frame_type": frame_type})
                _ws_ref2 = _resolve_workspace_root(self.working_dir)
                _ref_policy2 = quality_ref_policy(_ws_ref2)
                _max_ref2 = quality_max_reference_images(_ws_ref2)
                available_image_path_and_text_pairs = _dedupe_image_text_pairs(available_image_path_and_text_pairs)
                _cand_cap2 = 4 if _ref_policy2 == "legacy" else max(4, _max_ref2)
                available_image_path_and_text_pairs = available_image_path_and_text_pairs[:_cand_cap2]  # cap refs to avoid muse2api payload hang
                selector_output = await self.reference_image_selector.select_reference_images_and_generate_prompt(
                    available_image_path_and_text_pairs=available_image_path_and_text_pairs,
                    frame_description=frame_desc
                )
                os.makedirs(os.path.dirname(selector_output_path), exist_ok=True)
                with open(selector_output_path, 'w', encoding='utf-8') as f:
                    json.dump(selector_output, f, ensure_ascii=False, indent=4)
                print(f"☑️ Selected reference images and generated prompt for {frame_type} frame of shot {shot_idx}, saved to {selector_output_path}.")
                _emit_render_progress(progress, "frame_prompt_done", f"Selected references for {frame_type} of shot {shot_idx}", {"shot_idx": shot_idx, "frame_type": frame_type, "path": selector_output_path})

            raw_pairs = selector_output["reference_image_path_and_text_pairs"]
            prompt = selector_output["text_prompt"]
            _ws_ref3 = _resolve_workspace_root(self.working_dir)
            _ref_policy3 = quality_ref_policy(_ws_ref3)
            _ref_limit3 = 2 if _ref_policy3 == "legacy" else quality_max_reference_images(_ws_ref3)
            _sheet_dir3 = None if _ref_policy3 == "legacy" else os.path.join(self.working_dir, "shots", str(shot_idx))
            reference_image_path_and_text_pairs = _ensure_character_portraits_and_spatial_anchors(
                pairs=raw_pairs,
                visible_characters=visible_characters,
                character_portraits_registry=character_portraits_registry,
                scene_anchor_pair=first_shot_ff_path_and_text_pair,
                limit=_ref_limit3,
                sheet_dir=_sheet_dir3,
                sheet_tag=frame_type,
            )
            prefix_prompt = ""
            for i, (image_path, text) in enumerate(reference_image_path_and_text_pairs):
                prefix_prompt += f"Image {i}: {text}\n"
            prompt = f"{prefix_prompt}\n{prompt}"
            prompt = _apply_series_enrichment(prompt, self.working_dir, visible_characters)
            reference_image_paths = [item[0] for item in reference_image_path_and_text_pairs]
            _write_refs_used(self.working_dir, shot_idx, frame_type, reference_image_path_and_text_pairs, visible_characters, character_portraits_registry)

            frame_image: ImageOutput = await self.generate_and_select_best_image(
                prompt=prompt,
                reference_image_paths=reference_image_paths,
                reference_image_path_and_text_pairs=reference_image_path_and_text_pairs,
                target_description=frame_desc,
                size="1600x900",
                candidates_save_dir=os.path.join(self.working_dir, "shots", str(shot_idx), f"{frame_type}_candidates"),
                progress=_scoped_progress(progress, shot_idx=shot_idx, frame_type=frame_type),
            )
            frame_image.save(frame_image_path)
            print(f"☑️ Generated {frame_type} frame for shot {shot_idx}, saved to {frame_image_path}.")
            _emit_render_progress(progress, "frame_done", f"Generated {frame_type} for shot {shot_idx}", {"shot_idx": shot_idx, "frame_type": frame_type, "path": frame_image_path})


        self.frame_events[shot_idx][frame_type].set()
        return frame_image_path


    async def construct_camera_tree(
        self,
        shot_descriptions: List[ShotDescription],
        quiet: bool = False,
    ):
        camera_tree_path = os.path.join(self.working_dir, "camera_tree.json")

        if os.path.exists(camera_tree_path):
            with open(camera_tree_path, "r", encoding="utf-8") as f:
                camera_tree = json.load(f)
            camera_tree = [Camera.model_validate(camera) for camera in camera_tree]
            _pipeline_print(quiet, f"🚀 Loaded {len(camera_tree)} cameras from existing file.")
            return camera_tree

        shot_descriptions = _normalize_model_list(shot_descriptions, ShotDescription, "shot_descriptions")
        cameras = _group_shots_into_cameras(shot_descriptions)

        camera_tree = await self.camera_image_generator.construct_camera_tree(cameras=cameras, shot_descs=shot_descriptions)
        camera_tree = _normalize_model_list(camera_tree, Camera, "camera_tree")
        with open(camera_tree_path, "w", encoding="utf-8") as f:
            json.dump([camera.model_dump() for camera in camera_tree], f, ensure_ascii=False, indent=4)
        _pipeline_print(quiet, f"✅ Constructed camera tree and saved to {camera_tree_path}.")
        return camera_tree




    async def extract_characters(
        self,
        script: str,
        quiet: bool = False,
    ):
        save_path = os.path.join(self.working_dir, "characters.json")

        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                characters = json.load(f)
            characters = [CharacterInScene.model_validate(character) for character in characters]
            _pipeline_print(quiet, f"🚀 Loaded {len(characters)} characters from existing file.")
        else:
            characters = await self.character_extractor.extract_characters(script)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump([character.model_dump() for character in characters], f, ensure_ascii=False, indent=4)
            _pipeline_print(quiet, f"✅ Extracted {len(characters)} characters from script and saved to {save_path}.")

        for character in characters:
            self.character_portrait_events[character.idx] = asyncio.Event()

        return characters


    async def generate_character_portraits(
        self,
        characters: List[CharacterInScene],
        character_portraits_registry: Optional[Dict[str, Dict[str, Dict[str, str]]]],
        style: str,
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
        character_ref_paths: Optional[Dict[str, List[str]]] = None,
    ):
        character_portraits_registry_path = os.path.join(self.working_dir, "character_portraits_registry.json")
        if character_portraits_registry is None:
            if os.path.exists(character_portraits_registry_path):
                with open(character_portraits_registry_path, 'r', encoding='utf-8') as f:
                    character_portraits_registry = json.load(f)
            else:
                character_portraits_registry = {}

        ref_map = character_ref_paths or {}
        tasks = []
        for character in characters:
            if character.identifier_in_scene in character_portraits_registry or not character.is_visible:
                continue
            refs = [str(p).strip() for p in (ref_map.get(character.identifier_in_scene) or []) if str(p).strip()][:4]
            if refs:
                tasks.append(
                    self.generate_portraits_for_single_character(
                        character,
                        style,
                        progress=progress,
                        reference_image_paths=refs,
                    )
                )
            else:
                tasks.append(
                    self.generate_portraits_for_single_character(character, style, progress=progress)
                )
        if tasks:
            for future in asyncio.as_completed(tasks):
                character_portraits_registry.update(await future)
                with open(character_portraits_registry_path, 'w', encoding='utf-8') as f:
                    json.dump(character_portraits_registry, f, ensure_ascii=False, indent=4)

            print(f"✅ Completed character portrait generation for {len(characters)} characters.")
            _emit_render_progress(progress, "character_portraits_done", "Completed character portrait generation", {"character_count": len(characters)})
        else:
            print("🚀 All characters already have portraits, skipping portrait generation.")
            _emit_render_progress(progress, "character_portraits_exist", "All character portraits already exist", {"character_count": len(characters)})
        return character_portraits_registry


    async def generate_portraits_for_single_character(
        self,
        character: CharacterInScene,
        style: str,
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
        reference_image_paths: Optional[List[str]] = None,
    ):
        character_dir = os.path.join(self.working_dir, "character_portraits", f"{character.idx}_{character.identifier_in_scene}")
        os.makedirs(character_dir, exist_ok=True)
        _emit_render_progress(progress, "character_portrait_start", f"Generating portraits for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene})

        front_portrait_path = os.path.join(character_dir, "front.png")
        if os.path.exists(front_portrait_path):
            pass
        else:
            _emit_render_progress(progress, "character_portrait_front_start", f"Generating front portrait for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene})
            if reference_image_paths:
                front_portrait_output = await self.character_portraits_generator.generate_front_portrait(
                    character, style, reference_image_paths=reference_image_paths
                )
            else:
                front_portrait_output = await self.character_portraits_generator.generate_front_portrait(character, style)
            front_portrait_output.save(front_portrait_path)
            _emit_render_progress(progress, "character_portrait_front_done", f"Generated front portrait for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene, "path": front_portrait_path})


        side_portrait_path = os.path.join(character_dir, "side.png")
        if os.path.exists(side_portrait_path):
            pass
        else:
            _emit_render_progress(progress, "character_portrait_side_start", f"Generating side portrait for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene})
            try:
                side_portrait_output = await self.character_portraits_generator.generate_side_portrait(character, front_portrait_path, style=style)
                side_portrait_output.save(side_portrait_path)
            except Exception as exc:
                print(f"⚠️ Side portrait generation failed for {character.identifier_in_scene} after retries ({exc}); reusing front portrait as fallback.")
                shutil.copy(front_portrait_path, side_portrait_path)
            _emit_render_progress(progress, "character_portrait_side_done", f"Generated side portrait for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene, "path": side_portrait_path})

        back_portrait_path = os.path.join(character_dir, "back.png")
        if os.path.exists(back_portrait_path):
            pass
        else:
            _emit_render_progress(progress, "character_portrait_back_start", f"Generating back portrait for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene})
            try:
                back_portrait_output = await self.character_portraits_generator.generate_back_portrait(character, front_portrait_path, style=style)
                back_portrait_output.save(back_portrait_path)
            except Exception as exc:
                print(f"⚠️ Back portrait generation failed for {character.identifier_in_scene} after retries ({exc}); reusing front portrait as fallback.")
                shutil.copy(front_portrait_path, back_portrait_path)
            _emit_render_progress(progress, "character_portrait_back_done", f"Generated back portrait for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene, "path": back_portrait_path})

        self.character_portrait_events.setdefault(character.idx, asyncio.Event()).set()

        print(f"☑️ Completed character portrait generation for {character.identifier_in_scene}.")
        _emit_render_progress(progress, "character_portrait_done", f"Portraits for {character.identifier_in_scene} ready", {"character_idx": character.idx, "identifier": character.identifier_in_scene})

        return {
            character.identifier_in_scene: {
                "front": {
                    "path": front_portrait_path,
                    "description": f"A front view portrait of {character.identifier_in_scene}.",
                },
                "side": {
                    "path": side_portrait_path,
                    "description": f"A side view portrait of {character.identifier_in_scene}.",
                },
                "back": {
                    "path": back_portrait_path,
                    "description": f"A back view portrait of {character.identifier_in_scene}.",
                },
            }
        }



    async def design_storyboard(
        self,
        script: str,
        characters: List[CharacterInScene],
        user_requirement: str,
        quiet: bool = False,
        grounded: bool = False,
        force: bool = False,
    ):
        storyboard_path = os.path.join(self.working_dir, "storyboard.json")
        if os.path.exists(storyboard_path) and not force:
            with open(storyboard_path, 'r', encoding='utf-8') as f:
                storyboard = json.load(f)
            storyboard = [ShotBriefDescription.model_validate(shot) for shot in storyboard]
            _pipeline_print(quiet, f"🚀 Loaded {len(storyboard)} shot brief descriptions from existing file.")
        else:
            _pipeline_print(quiet, f"🔍 Designing storyboard...")
            storyboard = await self.storyboard_artist.design_storyboard(
                script=script,
                characters=characters,
                user_requirement=user_requirement,
                retry_timeout=240,
                grounded=grounded,
            )
            storyboard = _normalize_model_list(storyboard, ShotBriefDescription, "storyboard")
            with open(storyboard_path, 'w', encoding='utf-8') as f:
                json.dump([shot.model_dump() for shot in storyboard], f, ensure_ascii=False, indent=4)
            _pipeline_print(quiet, f"✅ Designed storyboard and saved to {storyboard_path}.")

        for shot_brief_description in storyboard:
            self.shot_desc_events[shot_brief_description.idx] = asyncio.Event()

        return storyboard



    async def decompose_visual_descriptions(
        self,
        shot_brief_descriptions: List[ShotBriefDescription],
        characters: List[CharacterInScene],
        quiet: bool = False,
    ):
        gate = getattr(self.chat_model, "_gate", None)
        gate_limit = max(1, int(getattr(gate, "limit", 1))) if gate is not None else None
        sem = asyncio.Semaphore(gate_limit) if gate_limit is not None else None

        async def _run_one(shot_brief_description: ShotBriefDescription):
            if sem is None:
                return await self.decompose_visual_description_for_single_shot_brief_description(
                    shot_brief_description, characters, quiet=quiet
                )
            async with sem:
                return await self.decompose_visual_description_for_single_shot_brief_description(
                    shot_brief_description, characters, quiet=quiet
                )

        tasks = [
            _run_one(shot_brief_description)
            for shot_brief_description in shot_brief_descriptions
        ]

        shot_descriptions = await asyncio.gather(*tasks)
        self._active_shot_descriptions = shot_descriptions
        self.frame_events = {}
        for sd in shot_descriptions:
            ev_map = {"first_frame": asyncio.Event()}
            if shot_requires_last_frame(shot_descriptions, sd):
                ev_map["last_frame"] = asyncio.Event()
            self.frame_events[sd.idx] = ev_map
        return shot_descriptions


    async def decompose_visual_description_for_single_shot_brief_description(
        self,
        shot_brief_description: ShotBriefDescription,
        characters: List[CharacterInScene],
        quiet: bool = False,
    ):
        shot_description_path = os.path.join(self.working_dir, "shots", f"{shot_brief_description.idx}", "shot_description.json")
        os.makedirs(os.path.dirname(shot_description_path), exist_ok=True)

        if os.path.exists(shot_description_path):
            with open(shot_description_path, 'r', encoding='utf-8') as f:
                shot_description = ShotDescription.model_validate(json.load(f))
            _pipeline_print(quiet, f"🚀 Loaded shot {shot_brief_description.idx} description from existing file.")
        else:
            shot_description = await self.storyboard_artist.decompose_visual_description(
                shot_brief_desc=shot_brief_description,
                characters=characters,
                retry_timeout=180,
            )
            shot_description = _normalize_model_list([shot_description], ShotDescription, "shot_description")[0]
            with open(shot_description_path, 'w', encoding='utf-8') as f:
                json.dump(shot_description.model_dump(), f, ensure_ascii=False, indent=4)
            _pipeline_print(quiet, f"✅ Decomposed visual description for shot {shot_brief_description.idx} and saved to {shot_description_path}.")

        self.shot_desc_events.setdefault(shot_brief_description.idx, asyncio.Event()).set()

        needs_lf = shot_requires_last_frame([shot_description], shot_description)
        ev_map = {"first_frame": asyncio.Event()}
        if needs_lf:
            ev_map["last_frame"] = asyncio.Event()
        self.frame_events[shot_brief_description.idx] = ev_map

        return shot_description
