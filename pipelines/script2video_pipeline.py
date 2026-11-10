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
    subtitles_burn_in,
    subtitles_enabled,
    subtitles_font,
    subtitles_font_size,
    subtitles_language,
)
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

    if getattr(shot, "variation_type", "small") in {"medium", "large"}:
        return True
    if bool(getattr(shot, "continuity_from_prev", False)):
        return True
    target_dur = getattr(shot, "target_duration_sec", None)
    if isinstance(target_dur, (int, float)) and not isinstance(target_dur, bool) and float(target_dur) <= 5.5:
        return True
    for i, candidate in enumerate(shot_descriptions):
        if candidate.idx == shot.idx:
            if i + 1 < len(shot_descriptions) and bool(getattr(shot_descriptions[i + 1], "continuity_from_prev", False)):
                return True
            break
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


def _resolve_character_acoustic_profile(speaker: str, characters_data: List[Any]) -> Dict[str, str]:
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

    is_narrator = any(w in spk_clean.lower() or w in desc for w in ["narrator", "người dẫn chuyện", "narration"])
    is_woman = any(w in spk_clean.lower() or w in desc for w in ["woman", "mother", "female", "girl", "mẹ", "bà", "cô"])
    is_child = (not is_woman) and (any(w in desc for w in ["elementary", "child", "small young", "6-7", "kid"]) or (bool(re.search(r"\bhojin\b", spk_clean.lower())) and not is_woman))
    is_teen = (not is_woman) and (not is_child) and (any(w in desc for w in ["teen", "high school", "16", "17", "young boy"]) or "jiwoo" in spk_clean.lower())
    is_man = (not is_woman) and (not is_child) and (not is_teen) and (any(w in desc for w in ["man", "male", "athletic build", "trench coat"]) or "kayden" in spk_clean.lower())

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
    elif is_man or "kayden" in spk_clean.lower():
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


def _format_video_prompt_audio(
    clean_audio_desc: str,
    characters_data: List[Any],
    timing_note: str = "",
) -> str:
    """Format dialogue into structured, hallucination-proof audio & vocal profile specifications."""
    if not clean_audio_desc or not clean_audio_desc.strip():
        return ""

    sfx_match = re.findall(r"\[Sound\s*Effect\]\s*([^\[\n]+)", clean_audio_desc, re.IGNORECASE)
    sfx_text = ", ".join(s.strip() for s in sfx_match if s.strip())

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
        prof = _resolve_character_acoustic_profile(speaker_block, characters_data)
        out = [
            "[Audio & Spoken Japanese Dialogue Track ONLY - Studio Anime Voice Acting, No On-screen Subtitles]",
            f"- Active Speaker: {prof['clean_speaker']}" + (f" ({emotion})" if emotion else ""),
            f"- Vocal Acoustic Profile: {prof['profile']}",
            f"- Delivery Mode: Spoken dialogue (expressive, natural dialogue pacing, lips move in sync).",
            f"- Acoustic Rules: {prof['negatives']} Only {prof['clean_speaker']} speaks. All other characters in the shot remain completely silent.",
            f"- Spoken Line: '{line}'",
        ]
        if sfx_text:
            out.append(f"- Sound Effects: {sfx_text}")
        if timing_note:
            out.append(timing_note.strip())
        return "\n".join(out)

    if sfx_text or clean_audio_desc:
        out = [
            "[Audio & Sound Effects Track ONLY - No Spoken Dialogue, No On-screen Subtitles]",
            f"- Sound Effects & Ambient: {clean_audio_desc}",
            "- Acoustic Rules: Pure Foley and environmental sound effects. No random speech, no human vocal chatter.",
        ]
        return "\n".join(out)

    return ""


def _ensure_character_portraits_and_spatial_anchors(
    pairs: List[Tuple[str, str]],
    visible_characters: List[Any],
    character_portraits_registry: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None,
    scene_anchor_pair: Optional[Tuple[str, str]] = None,
    limit: int = 8,
) -> List[Tuple[str, str]]:
    """Guarantees that canonical portraits for all visible characters and the scene establishing anchor are present."""
    pairs = list(pairs or [])
    existing_paths = {p[0] for p in pairs}

    if character_portraits_registry and visible_characters:
        for char in visible_characters:
            if not getattr(char, "is_visible", True):
                continue
            ident = getattr(char, "identifier_in_scene", "")
            reg = character_portraits_registry.get(ident) or {}
            char_portrait_paths = {v["path"] for v in reg.values() if isinstance(v, dict) and "path" in v}
            if not char_portrait_paths.intersection(existing_paths):
                chosen = (
                    reg.get("front")
                    or reg.get("front_three_quarter")
                    or reg.get("side")
                    or next(iter(reg.values()), None)
                )
                if chosen and isinstance(chosen, dict) and os.path.exists(chosen.get("path", "")):
                    p_path = chosen["path"]
                    p_desc = f"Canonical character portrait reference for <{ident}>: maintain exact facial features, hairstyle, hair color, eye shape, and costume."
                    pairs.insert(0, (p_path, p_desc))
                    existing_paths.add(p_path)

    if scene_anchor_pair and scene_anchor_pair[0] and os.path.exists(scene_anchor_pair[0]):
        if scene_anchor_pair[0] not in existing_paths:
            insert_idx = min(len(visible_characters or []), len(pairs))
            pairs.insert(insert_idx, scene_anchor_pair)
            existing_paths.add(scene_anchor_pair[0])

    return _dedupe_image_text_pairs(pairs, limit=limit)


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
        from agents.anime_director_guard import AnimeDirectorGuard
        storyboard = AnimeDirectorGuard.lint_and_auto_correct_storyboard(storyboard, characters=characters)
        _emit_render_progress(progress, "storyboard_ready", "Storyboard ready", {"shot_count": len(storyboard)})

        # decompose visual descriptions of shots
        _emit_render_progress(progress, "load_shot_descriptions", "Loading or decomposing shot descriptions", {"shot_count": len(storyboard)})
        shot_descriptions = await self.decompose_visual_descriptions(
            shot_brief_descriptions=storyboard,
            characters=characters,
            quiet=quiet,
        )
        shot_descriptions = [AnimeDirectorGuard.lint_and_auto_correct_shot_description(sd, characters=characters) for sd in shot_descriptions]
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
        for sd in shot_descriptions:
            ev_map = self.frame_events.setdefault(sd.idx, {"first_frame": asyncio.Event()})
            if shot_requires_last_frame(shot_descriptions, sd):
                ev_map.setdefault("last_frame", asyncio.Event())

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
                )
                if effective_panel_refs
                else self.generate_frames_for_single_camera(
                    camera=camera,
                    shot_descriptions=shot_descriptions,
                    characters=characters,
                    character_portraits_registry=character_portraits_registry,
                    priority_shot_idxs=priority_shot_idxs,
                    progress=progress,
                )
            )
            for camera in camera_tree
        ]

        if stop_after == "frames":
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
        if "last_frame" in prev_events:
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
            elif first_panel_pairs and camera.parent_shot_idx is not None:
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

            if continuity_pair is not None or first_panel_pairs or camera.parent_shot_idx is None or camera.missing_info is not None:
                ff_selector_output_path = os.path.join(self.working_dir, "shots", f"{first_shot_idx}", "first_frame_selector_output.json")
                if os.path.exists(ff_selector_output_path):
                    with open(ff_selector_output_path, 'r', encoding='utf-8') as f:
                        ff_selector_output = json.load(f)
                    print(f"🚀 Loaded existing reference image selection and prompt for first_frame of shot {first_shot_idx} from {ff_selector_output_path}.")
                    _emit_render_progress(progress, "frame_prompt_exists", f"First frame prompt for shot {first_shot_idx} already exists", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "frame_type": "first_frame", "path": ff_selector_output_path})
                else:
                    print(f"🔍 Selecting reference images and generating prompt for first_frame of shot {first_shot_idx}...")
                    _emit_render_progress(progress, "frame_prompt_start", f"Selecting references for first frame of shot {first_shot_idx}", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "frame_type": "first_frame"})
                    available_image_path_and_text_pairs = _dedupe_image_text_pairs(available_image_path_and_text_pairs)
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
                scene_shot0_ff = os.path.join(self.working_dir, "shots", "0", "first_frame.png")
                if os.path.exists(scene_shot0_ff) and first_shot_idx != 0:
                    scene_anchor = (scene_shot0_ff, "Scene establishing environment anchor: maintain the exact same architectural location, background walls, floor/ground textures, time of day, and lighting atmosphere.")

                reference_image_path_and_text_pairs = _ensure_character_portraits_and_spatial_anchors(
                    pairs=raw_pairs,
                    visible_characters=vis_chars,
                    character_portraits_registry=character_portraits_registry,
                    scene_anchor_pair=scene_anchor,
                    limit=8,
                )
                prefix_prompt = ""
                for i, (image_path, text) in enumerate(reference_image_path_and_text_pairs):
                    prefix_prompt += f"Image {i}: {text}\n"
                prompt = f"{prefix_prompt}\n{prompt}"
                reference_image_paths = [item[0] for item in reference_image_path_and_text_pairs]
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
            needs_last_frame = (
                "last_frame" in shot_ev_map
                or shot_requires_last_frame(getattr(self, "_active_shot_descriptions", None) or [shot_description], shot_description)
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
            has_two_keyframes = len(frame_paths) >= 2
            keyframe_note = (
                "Animate seamlessly starting from Image 0 (exact opening frame) through the full movement to land precisely on Image 1 (exact final keyframe). Maintain consistent anime character proportions, battle wear, lighting, and background across the whole clip."
                if has_two_keyframes
                else ""
            )
            has_spoken_audio = bool(re.search(r"\[(?:Speaker|Thought|Inner\s*Monologue)\]", clean_audio_desc, re.IGNORECASE))
            timing_note = (
                "\n(Spoken Delivery Pacing: start speaking promptly near 0.3s and complete every word cleanly with natural phrasing before the shot ends.)"
                if has_spoken_audio
                else ""
            )
            characters_data = []
            chars_json_path = os.path.join(self.working_dir, "characters.json")
            if os.path.exists(chars_json_path):
                try:
                    with open(chars_json_path, "r", encoding="utf-8") as f:
                        characters_data = json.load(f)
                except Exception:
                    characters_data = []

            formatted_audio = _format_video_prompt_audio(clean_audio_desc, characters_data, timing_note)
            if suppress_subs:
                visual_body = "\n".join(part for part in (keyframe_note, motion_clean) if part).strip()
                sections = []
                if visual_body:
                    sections.append(
                        f"[Pure Visual Scene - Clean Textless Borderless Animation Only]\n{visual_body}"
                    )
                if formatted_audio:
                    sections.append(formatted_audio)
                video_prompt = "\n\n".join(sections).strip()
            else:
                visual_body = "\n".join(part for part in (keyframe_note, motion_clean) if part).strip()
                video_prompt = "\n\n".join(part for part in (visual_body, formatted_audio) if part).strip()
            gen_kwargs: Dict[str, Any] = {
                "prompt": video_prompt,
                "reference_image_paths": frame_paths,
                "progress": _scoped_progress(progress, shot_idx=shot_description.idx, artifact="video_clip"),
            }
            try:
                sig = inspect.signature(self.video_generator.generate_single_video)
                params = sig.parameters
                if "suppress_on_screen_subtitles" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                    gen_kwargs["suppress_on_screen_subtitles"] = suppress_subs
            except (TypeError, ValueError):
                pass
            video_output = await self.video_generator.generate_single_video(**gen_kwargs)
            video_output.save(video_path)
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
                "prompt": prompt,
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
                        prompt=prompt, reference_image_paths=reference_image_paths, size=size,
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
                available_image_path_and_text_pairs = _dedupe_image_text_pairs(available_image_path_and_text_pairs)
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
            reference_image_path_and_text_pairs = _ensure_character_portraits_and_spatial_anchors(
                pairs=raw_pairs,
                visible_characters=visible_characters,
                character_portraits_registry=character_portraits_registry,
                scene_anchor_pair=first_shot_ff_path_and_text_pair,
                limit=8,
            )
            prefix_prompt = ""
            for i, (image_path, text) in enumerate(reference_image_path_and_text_pairs):
                prefix_prompt += f"Image {i}: {text}\n"
            prompt = f"{prefix_prompt}\n{prompt}"
            reference_image_paths = [item[0] for item in reference_image_path_and_text_pairs]

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
    ):
        storyboard_path = os.path.join(self.working_dir, "storyboard.json")
        if os.path.exists(storyboard_path):
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
        for sd in shot_descriptions:
            ev_map = self.frame_events.setdefault(sd.idx, {"first_frame": asyncio.Event()})
            if shot_requires_last_frame(shot_descriptions, sd):
                ev_map.setdefault("last_frame", asyncio.Event())
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

        if shot_description.variation_type in ["medium", "large"]:
            self.frame_events[shot_brief_description.idx] = {
                "first_frame": asyncio.Event(),
                "last_frame": asyncio.Event(),
            }
        else:
            self.frame_events[shot_brief_description.idx] = {
                "first_frame": asyncio.Event(),
            }

        return shot_description
