import os
import re
import json
import math
import time
import shutil
import asyncio
import logging
import subprocess
import requests
from pathlib import Path
from typing import Optional, List, Tuple, Any, Dict
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip
from utils.robust_json_parser import (
    iter_json_blocks,
    strip_markdown_json_wrapper,
    strip_trailing_commas,
)

logger = logging.getLogger(__name__)

async def generate_speech_edge_tts(
    text: str,
    output_path: str,
    voice: str = "vi-VN-NamMinhNeural",
    rate: str = "+0%",
    pitch: str = "+0Hz"
) -> str:
    """Generate high-quality speech using Edge-TTS (Microsoft Neural Voice)."""
    import edge_tts
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    communicator = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicator.save(output_path)
    return output_path

def generate_speech_elevenlabs(
    text: str,
    output_path: str,
    api_key: str,
    voice_id: str = "21m00Tcm4TlvDq8ikWAM",
    model_id: str = "eleven_multilingual_v2",
    base_url: str = "https://api.elevenlabs.io/v1",
) -> str:
    """Generate premium expressive speech using ElevenLabs API."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    voice_id = voice_id.strip() or "21m00Tcm4TlvDq8ikWAM"
    url = f"{base_url.rstrip('/')}/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key.strip(),
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }
    payload = {
        "text": text,
        "model_id": model_id.strip() or "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
        }
    }
    response = requests.post(url, json=payload, headers=headers, timeout=90)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)
    return output_path

async def synthesize_speech(
    text: str,
    output_path: str,
    provider: str = "elevenlabs",
    api_key: str = "",
    voice: str = "",
    model: str = "eleven_multilingual_v2",
    base_url: str = "https://api.elevenlabs.io/v1",
    rate: str = "+0%",
    pitch: str = "+0Hz",
) -> str:
    """Synthesize speech using either ElevenLabs or Edge-TTS based on config."""
    text = (text or "").strip()
    if not text:
        return ""

    provider = (provider or "elevenlabs").lower().strip()
    if provider == "elevenlabs" and api_key.strip():
        voice_id = voice.strip() or "21m00Tcm4TlvDq8ikWAM"
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            generate_speech_elevenlabs,
            text,
            output_path,
            api_key.strip(),
            voice_id,
            model.strip() or "eleven_multilingual_v2",
            base_url.strip() or "https://api.elevenlabs.io/v1"
        )
    else:
        edge_voice = voice.strip() if "Neural" in (voice or "") else "vi-VN-NamMinhNeural"
        return await generate_speech_edge_tts(text, output_path, voice=edge_voice, rate=rate, pitch=pitch)

DEFAULT_FEMALE_VOICE_ELEVENLABS = "EXAVITQu4vr4xnSDxMaL"
DEFAULT_MALE_VOICE_ELEVENLABS = "pNInz6obpgDQGcFmaJgB"
DEFAULT_FEMALE_VOICE_EDGE_TTS = "vi-VN-HoaiMyNeural"
DEFAULT_MALE_VOICE_EDGE_TTS = "vi-VN-NamMinhNeural"

def _strip_vietsub_annotations(text: str) -> str:
    cleaned = re.sub(r"\(\s*Vietsub\s*:[^)]*\)", "", text or "", flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*Vietsub\s*:[^\n]*\n?", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    return cleaned.strip()


def extract_all_dialogues_from_shot(audio_desc: str = "", visual_desc: str = "") -> List[Tuple[str, str]]:
    """Extract all character speakers and dialogue lines from storyboard shot descriptions."""
    audio_desc = _strip_vietsub_annotations(audio_desc or "")
    visual_desc = visual_desc or ""
    dialogues = []

    # 1. Match all [Speaker] / [Thought] / [Inner Monologue] Name (Emotion): Dialogue text up to next tag or end
    pattern = r'\[(?:Speaker|Thought|Inner\s*Monologue)\]\s*([^:\(\[\n]+?)(?:\s*\([^)]*\))?\s*:\s*(.+?)(?=\s*\[(?:Speaker|Thought|Inner\s*Monologue|Sound\s*Effect)\]|$)'
    for m in re.finditer(pattern, audio_desc, re.IGNORECASE | re.DOTALL):
        spk = m.group(1).strip().strip('<>"\'')
        txt = m.group(2).strip().strip('"\'')
        txt = re.sub(r'\[.*?\]', '', txt).strip()
        if spk and txt:
            dialogues.append((spk, txt))

    if dialogues:
        return dialogues

    # 2. Match character speech in visual_desc: <Alice> says: "..."
    char_says_matches = re.finditer(r'<([^>]+)>[^:]*says?:\s*["\']([^"\']+)["\']', visual_desc, re.IGNORECASE)
    for m in char_says_matches:
        spk = m.group(1).strip()
        txt = m.group(2).strip()
        if spk and txt:
            dialogues.append((spk, txt))

    if dialogues:
        return dialogues

    # 2b. Match general Character says in visual_desc: Charlie says: "..."
    gen_says_matches = re.finditer(r'(?:^|[\n.,;]\s*)([A-Za-z0-9_\s]{2,20})\s+says?:\s*["\']([^"\']+)["\']', visual_desc, re.IGNORECASE)
    for m in gen_says_matches:
        spk = m.group(1).strip()
        txt = m.group(2).strip()
        if spk and txt:
            dialogues.append((spk, txt))

    if dialogues:
        return dialogues

    # 3. Match quotes in audio_desc
    quote_match = re.search(r'["\']([^"\']{4,})["\']', audio_desc)
    if quote_match:
        return [("", quote_match.group(1).strip())]

    # 4. Filter out [Sound Effect]
    cleaned = re.sub(r'\[Sound Effect\][^\n]*', '', audio_desc, flags=re.IGNORECASE).strip()
    if len(cleaned) > 5 and not cleaned.lower().startswith('none'):
        return [("", cleaned)]

    return []

def extract_dialogue_from_shot(audio_desc: str = "", visual_desc: str = "") -> Tuple[str, str]:
    """Extract single character speaker and dialogue line (backwards compatibility)."""
    all_d = extract_all_dialogues_from_shot(audio_desc, visual_desc)
    if all_d:
        return all_d[0]
    return "", ""

def extract_speech_from_shot(audio_desc: str = "", visual_desc: str = "") -> str:
    """Backwards-compatible helper returning just the dialogue line."""
    _, text = extract_dialogue_from_shot(audio_desc, visual_desc)
    return text

def resolve_voice_for_speaker(
    speaker: str,
    characters: Optional[List[Any]] = None,
    provider: str = "elevenlabs",
    default_voice: str = "",
    character_voices: Optional[dict] = None,
) -> str:
    """Resolve the specific Voice ID for a given character speaker."""
    speaker = (speaker or "").strip()
    provider = (provider or "elevenlabs").lower().strip()
    character_voices = character_voices or {}

    # 1. Exact or case-insensitive match in character_voices dictionary
    if speaker:
        for name, v_id in character_voices.items():
            if name.strip().lower() == speaker.lower() and v_id:
                return v_id.strip()

    # 1b. Specific Anime / Eleceed character voice mapping for authentic Japanese audio
    spk_low = (speaker or "").strip().lower()
    if "jiwoo" in spk_low:
        return "ja-JP-KeitaNeural" if provider == "edge-tts" else DEFAULT_MALE_VOICE_ELEVENLABS
    if "hojin" in spk_low:
        return "ja-JP-NanamiNeural" if provider == "edge-tts" else DEFAULT_FEMALE_VOICE_ELEVENLABS
    if "kayden" in spk_low:
        return "ja-JP-DaichiNeural" if provider == "edge-tts" else DEFAULT_MALE_VOICE_ELEVENLABS

    # 2. Check character description/gender from characters metadata
    is_female = False
    is_male = False

    female_keywords = {"female", "woman", "girl", "nữ", "cô", "nàng", "bà", "chị", "em", "phụ nữ", "ai", "trợ lý"}
    male_keywords = {"male", "man", "boy", "nam", "chàng", "ông", "anh", "đàn ông", "sếp", "huy"}

    if speaker and characters:
        for char in characters:
            ident = getattr(char, "identifier_in_scene", "") or getattr(char, "name", "")
            if speaker.lower() in ident.lower() or ident.lower() in speaker.lower():
                static_feat = (getattr(char, "static_features", "") or "").lower()
                dynamic_feat = (getattr(char, "dynamic_features", "") or "").lower()
                combined = f"{ident.lower()} {static_feat} {dynamic_feat}"
                if any(kw in combined for kw in female_keywords):
                    is_female = True
                    break
                elif any(kw in combined for kw in male_keywords):
                    is_male = True
                    break

    if not is_female and not is_male and speaker:
        lower_sp = speaker.lower()
        if any(kw in lower_sp for kw in female_keywords):
            is_female = True
        elif any(kw in lower_sp for kw in male_keywords):
            is_male = True

    if is_female:
        if provider == "elevenlabs":
            return character_voices.get("female") or character_voices.get("Female") or DEFAULT_FEMALE_VOICE_ELEVENLABS
        else:
            return character_voices.get("female") or character_voices.get("Female") or DEFAULT_FEMALE_VOICE_EDGE_TTS

    if is_male:
        if provider == "elevenlabs":
            return character_voices.get("male") or character_voices.get("Male") or DEFAULT_MALE_VOICE_ELEVENLABS
        else:
            return character_voices.get("male") or character_voices.get("Male") or DEFAULT_MALE_VOICE_EDGE_TTS

    # 3. Fallback to default configured voice
    return default_voice.strip() or (DEFAULT_MALE_VOICE_ELEVENLABS if provider == "elevenlabs" else DEFAULT_MALE_VOICE_EDGE_TTS)

def resolve_voice_and_prosody(
    speaker: str,
    characters: Optional[List[Any]] = None,
    provider: str = "edge-tts",
    default_voice: str = "",
    character_voices: Optional[dict] = None,
) -> Tuple[str, str, str]:
    """Returns (voice_id, rate_str, pitch_str) tailored to the character role."""
    spk_lower = (speaker or "").strip().lower()
    voice = resolve_voice_for_speaker(speaker, characters, provider, default_voice, character_voices)

    if "jiwoo" in spk_lower:
        return "ja-JP-KeitaNeural", "+3%", "+0Hz"
    if "hojin" in spk_lower:
        return "ja-JP-NanamiNeural", "+8%", "+10Hz"
    if "kayden" in spk_lower:
        return "ja-JP-DaichiNeural", "-5%", "-5Hz"
    if any(kw in spk_lower for kw in ["sếp", "sep", "boss", "manager", "quản lý"]):
        return voice, "-5%", "-5Hz"
    if any(kw in spk_lower for kw in ["huy", "dev", "coder", "developer"]):
        return voice, "+8%", "+0Hz"
    if any(kw in spk_lower for kw in ["ai", "assistant", "trợ lý", "bot", "synthetic"]):
        ai_voice = (character_voices or {}).get("AI trợ lý") or (character_voices or {}).get("AI") or DEFAULT_FEMALE_VOICE_EDGE_TTS
        return ai_voice, "+0%", "+0Hz"

    return voice, "+0%", "+0Hz"

def mix_multitrack_video_and_audio(
    video_path: str,
    output_path: str,
    speech_segments: List[Tuple[float, str]],  # [(start_offset_seconds, audio_file_path), ...]
    bgm_path: Optional[str] = None,
    bgm_volume: float = 0.15,
    preserve_video_audio: bool = True,
    video_audio_volume: float = 1.0,
) -> str:
    """Mix multiple character speech clips timed across the video timeline, preserving native audio when requested."""
    if not os.path.exists(video_path):
        return video_path

    has_speech = any(voice_path and os.path.exists(voice_path) for _, voice_path in (speech_segments or []))
    has_bgm = bool(bgm_path and os.path.exists(bgm_path))
    if not has_speech and not has_bgm:
        if video_path != output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            shutil.copyfile(video_path, output_path)
        return output_path

    video = VideoFileClip(video_path)
    audio_clips = []

    if preserve_video_audio and video.audio is not None:
        try:
            native_clip = (
                video.audio
                if video_audio_volume == 1.0
                else video.audio.with_volume_scaled(video_audio_volume)
            )
            audio_clips.append(native_clip)
        except Exception as e:
            logger.warning(f"Error preserving native video audio: {e}")

    # 1. Load each character's speech clip positioned at its start time
    for start_time, voice_path in speech_segments:
        if voice_path and os.path.exists(voice_path):
            try:
                clip = AudioFileClip(voice_path)
                if start_time < video.duration:
                    clip = clip.with_start(start_time)
                    audio_clips.append(clip)
                else:
                    clip.close()
            except Exception as e:
                logger.warning(f"Error loading speech clip {voice_path} at t={start_time}: {e}")

    # 2. Load background music
    if has_bgm:
        try:
            from moviepy import concatenate_audioclips
            bgm_clip = AudioFileClip(bgm_path).with_volume_scaled(bgm_volume)
            if bgm_clip.duration < video.duration:
                repeats = math.ceil(video.duration / bgm_clip.duration)
                bgm_clip = concatenate_audioclips([bgm_clip] * repeats)
            bgm_clip = bgm_clip.subclipped(0, video.duration)
            audio_clips.append(bgm_clip)
        except Exception as e:
            logger.warning(f"Error loading BGM {bgm_path}: {e}")

    if not audio_clips:
        video.close()
        if video_path != output_path:
            shutil.copyfile(video_path, output_path)
        return output_path

    try:
        final_audio = CompositeAudioClip(audio_clips)
        final_video = video.with_audio(final_audio)

        temp_target = output_path + ".tmp.mp4"
        final_video.write_videofile(
            temp_target,
            codec="libx264",
            audio_codec="aac",
            preset="fast",
            logger=None
        )

        final_video.close()
        for clip in audio_clips:
            if clip is not video.audio:
                clip.close()
        video.close()

        _replace_file_with_retry(Path(temp_target), Path(output_path))
        return output_path
    except Exception as e:
        logger.error(f"Error mixing audio into video: {e}")
        for clip in audio_clips:
            if clip is not video.audio:
                try:
                    clip.close()
                except Exception:
                    pass
        video.close()
        return video_path


def extract_spoken_dialogues_for_subtitles(audio_desc: str = "", visual_desc: str = "") -> List[Tuple[str, str]]:
    """Extract spoken/thought character dialogue lines while ignoring [Sound Effect] quote snippets."""
    audio_no_sfx = re.sub(
        r'\[Sound\s*Effect\].*?(?=\s*\[(?:Speaker|Thought|Inner\s*Monologue)\]|\n|$)',
        '',
        audio_desc or "",
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    return extract_all_dialogues_from_shot(audio_desc=audio_no_sfx, visual_desc=visual_desc)


def extract_vietsubs_from_script(script_text: str) -> Dict[int, str]:
    """Parse per-shot (Vietsub: ...) lines from script.txt and normalize shot indices to 0-based."""
    if not (script_text or "").strip():
        return {}

    shot_header_re = re.compile(r'^\s*Shot\s+(\d+)\b', re.IGNORECASE)
    vietsub_re = re.compile(r'^\s*\(?\s*Vietsub\s*:\s*(.+?)\s*\)?\s*$', re.IGNORECASE)

    parsed_by_shot: Dict[int, List[str]] = {}
    all_shot_nums: List[int] = []
    current_shot: Optional[int] = None

    for raw_line in script_text.splitlines():
        line = raw_line.strip()
        header_m = shot_header_re.match(line)
        if header_m:
            current_shot = int(header_m.group(1))
            all_shot_nums.append(current_shot)
            continue
        if current_shot is not None:
            sub_m = vietsub_re.match(line)
            if sub_m:
                sub_text = sub_m.group(1).strip()
                if sub_text.endswith(")") and "(" not in sub_text:
                    sub_text = sub_text[:-1].strip()
                if sub_text:
                    parsed_by_shot.setdefault(current_shot, []).append(sub_text)

    if not parsed_by_shot:
        return {}

    one_based = bool(all_shot_nums) and 0 not in all_shot_nums and min(all_shot_nums) >= 1
    normalized: Dict[int, str] = {}
    for shot_num, parts in parsed_by_shot.items():
        idx = shot_num - 1 if one_based else shot_num
        normalized[idx] = " ".join(parts)
    return normalized


def extract_speaker_and_dialogue_info(audio_desc: str) -> Dict[str, Any]:
    """Generic extractor for speaker name, delivery mode (spoken vs thought), emotion, and clean speaker label."""
    if not audio_desc:
        return {"speaker": "", "is_thought": False, "emotion": "", "clean_speaker": ""}

    # 1. Narrator
    narrator_match = re.search(
        r"\[(?:Speaker|Off-screen.*?Voiceover)\]\s*(?:Người Dẫn Chuyện|Narrator)\s*(?:\([^)]*\))?\s*:",
        audio_desc,
        re.IGNORECASE,
    )
    if narrator_match:
        return {
            "speaker": "Narrator",
            "is_thought": False,
            "emotion": "",
            "clean_speaker": "Người dẫn chuyện",
        }

    # 2. Thought / Inner monologue patterns
    thought_match = re.search(
        r"\[(?:Thought|Inner\s*Monologue)\]\s*([^:\n]+?)\s*:",
        audio_desc,
        re.IGNORECASE,
    )
    if not thought_match:
        thought_match = re.search(
            r"\[Speaker\]\s*([^:\n]+?)\s*\(([^)]*inner\s*monologue[^)]*)\)\s*:",
            audio_desc,
            re.IGNORECASE,
        )
    if not thought_match:
        thought_match = re.search(
            r"\[Off-screen\s+Japanese\s+Inner\s+Monologue\s+Voiceover\s+of\s+([^:\]\n]+?)\]\s*:",
            audio_desc,
            re.IGNORECASE,
        )
    if thought_match:
        block = thought_match.group(1).strip()
        clean_speaker = re.sub(r"\s*\([^)]*\)", "", block).strip()
        em = re.findall(r"\(([^)]+)\)", block)
        raw_em = em[-1].strip() if em else ""
        emotion = "" if re.search(r"\b(cat|human|boy|girl|man|woman)\b", raw_em.lower()) else raw_em
        return {
            "speaker": block,
            "is_thought": True,
            "emotion": emotion,
            "clean_speaker": clean_speaker,
        }

    # 3. Spoken dialogue: [Speaker] Name (Emotion):
    spoken_match = re.search(
        r"\[Speaker\]\s*([^:\n]+?)\s*:",
        audio_desc,
        re.IGNORECASE,
    )
    if spoken_match:
        block = spoken_match.group(1).strip()
        clean_speaker = re.sub(r"\s*\([^)]*\)", "", block).strip()
        em = re.findall(r"\(([^)]+)\)", block)
        raw_em = em[-1].strip() if em else ""
        emotion = "" if re.search(r"\b(cat|human|boy|girl|man|woman)\b", raw_em.lower()) else raw_em
        return {
            "speaker": block,
            "is_thought": False,
            "emotion": emotion,
            "clean_speaker": clean_speaker,
        }

    return {"speaker": "", "is_thought": False, "emotion": "", "clean_speaker": ""}


def format_attributed_subtitle(text: str, audio_desc: str) -> str:
    """Format subtitle line with clean speaker attribution to avoid dialogue confusion across characters."""
    clean_text = (text or "").strip()
    if not clean_text:
        return ""
    if re.match(r"^(\([^)]+\):|[A-Za-z0-9_\s\.\-'\u00C0-\u1EF9]+:)", clean_text):
        return clean_text

    info = extract_speaker_and_dialogue_info(audio_desc)
    spk = info.get("clean_speaker", "").strip()
    if not spk:
        return clean_text

    inner = clean_text.strip("\"'“”")
    if info.get("is_thought"):
        return f"({spk}): \"{inner}\""
    else:
        return f"{spk}: \"{inner}\""


def extract_vietsubs_from_shots(shot_descriptions: List[Any], include_attribution: bool = True) -> Dict[int, str]:
    """Extract inline `(Vietsub: ...)` lines stored inside ShotDescription.audio_desc with speaker attribution."""
    extracted: Dict[int, str] = {}
    inline_re = re.compile(r"\(\s*Vietsub\s*:\s*([^)]+?)\s*\)", re.IGNORECASE)
    line_re = re.compile(r"^\s*Vietsub\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
    for i, sd in enumerate(shot_descriptions or []):
        audio_desc = str(getattr(sd, "audio_desc", "") or "")
        parts = [m.group(1).strip() for m in inline_re.finditer(audio_desc) if m.group(1).strip()]
        if not parts:
            parts = [m.group(1).strip() for m in line_re.finditer(audio_desc) if m.group(1).strip()]
        if parts:
            sd_idx = getattr(sd, "idx", i)
            raw_sub = " ".join(parts)
            if include_attribution:
                extracted[int(sd_idx)] = format_attributed_subtitle(raw_sub, audio_desc)
            else:
                extracted[int(sd_idx)] = raw_sub
    return extracted


def _split_subtitle_text(text: str, max_chars: int = 75) -> List[str]:
    clean = " ".join((text or "").split()).strip()
    if not clean:
        return []
    if len(clean) <= max_chars:
        return [clean]

    midpoint = len(clean) // 2
    min_chunk = 14
    candidates: List[int] = []

    for delim in ("... ", "! ", "? ", ". ", ", "):
        start = 0
        while True:
            pos = clean.find(delim, start)
            if pos == -1:
                break
            split_at = pos + len(delim.rstrip())
            if split_at >= min_chunk and len(clean) - split_at >= min_chunk:
                candidates.append(split_at)
            start = pos + len(delim)
        if candidates:
            break

    if not candidates:
        for i, ch in enumerate(clean):
            if ch == " " and i >= min_chunk and len(clean) - i >= min_chunk:
                candidates.append(i)

    if not candidates:
        return [clean]

    best = min(candidates, key=lambda idx: abs(idx - midpoint))
    first = clean[:best].strip()
    second = clean[best:].strip()
    if first and second:
        return [first, second]
    return [clean]


def build_subtitle_cues(
    shot_descriptions: List[Any],
    shot_durations: List[float],
    script_vietsubs: Optional[Dict[int, str]] = None,
    translated_dialogues: Optional[Dict[int, str]] = None,
    max_chars_per_cue: int = 75,
    shot_start_offsets: Optional[List[float]] = None,
) -> List[Tuple[float, float, str]]:
    """Build (start_sec, end_sec, text) cues aligned to rendered shot timeline offsets and durations."""
    script_vietsubs = script_vietsubs or {}
    translated_dialogues = translated_dialogues or {}
    cues: List[Tuple[float, float, str]] = []
    current_offset = 0.0

    for i, sd in enumerate(shot_descriptions):
        shot_len = float(shot_durations[i]) if i < len(shot_durations) else 8.0
        base_offset = float(shot_start_offsets[i]) if (shot_start_offsets is not None and i < len(shot_start_offsets)) else current_offset
        sd_idx = getattr(sd, "idx", i)
        text = (
            script_vietsubs.get(sd_idx)
            or script_vietsubs.get(i)
            or translated_dialogues.get(sd_idx)
            or translated_dialogues.get(i)
            or ""
        ).strip()
        if text:
            audio_desc = str(getattr(sd, "audio_desc", "") or "")
            text = format_attributed_subtitle(text, audio_desc)

        if text and shot_len > 0.25:
            lead = min(0.20, shot_len * 0.1)
            tail = min(0.15, shot_len * 0.05)
            usable = max(0.20, shot_len - lead - tail)
            t0 = base_offset + lead
            t_end = t0 + usable
            if shot_start_offsets is not None and i + 1 < len(shot_start_offsets):
                next_start = float(shot_start_offsets[i + 1])
                if next_start > t0 + 0.06:
                    gap_margin = 0.04 if next_start - t0 > 0.25 else 0.02
                    t_end = min(t_end, next_start - gap_margin)
                    usable = max(0.04, t_end - t0)
            parts = _split_subtitle_text(text, max_chars=max_chars_per_cue)
            if len(parts) == 1:
                cues.append((t0, t0 + usable, parts[0]))
            elif len(parts) >= 2:
                gap = min(0.08, usable * 0.05)
                avail = usable - gap
                total_len = max(1, sum(len(p) for p in parts[:2]))
                dur1 = avail * (len(parts[0]) / total_len)
                min_part = 0.8 if avail >= 2.0 else avail * 0.35
                dur1 = max(min_part, min(avail - min_part, dur1))
                cues.append((t0, t0 + dur1, parts[0]))
                cues.append((t0 + dur1 + gap, t0 + usable, parts[1]))

        current_offset += shot_len

    return cues


def format_srt_timestamp(seconds: float) -> str:
    ms_total = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(ms_total, 3600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_ass_timestamp(seconds: float) -> str:
    cs_total = max(0, int(round(seconds * 100)))
    hours, rem = divmod(cs_total, 360_000)
    minutes, rem = divmod(rem, 6000)
    secs, centis = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def write_srt_subtitles(cues: List[Tuple[float, float, str]], output_path: str) -> str:
    blocks = []
    for idx, (start, end, text) in enumerate(cues, 1):
        blocks.append(f"{idx}\n{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n{text.strip()}")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")
    return str(out)


def _escape_ass_text(text: str) -> str:
    return (
        text.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\r\n", "\n")
        .replace("\n", r"\N")
    )


def write_ass_subtitles(
    cues: List[Tuple[float, float, str]],
    output_path: str,
    font: str = "Arial",
    font_size: int = 34,
    play_res_x: int = 1280,
    play_res_y: int = 720,
) -> str:
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {play_res_x}\n"
        f"PlayResY: {play_res_y}\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font},{font_size},&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2.4,1.2,2,48,48,30,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = [
        f"Dialogue: 0,{format_ass_timestamp(start)},{format_ass_timestamp(end)},Default,,0,0,0,,{_escape_ass_text(text.strip())}"
        for start, end, text in cues
    ]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + "\n".join(events) + ("\n" if events else ""), encoding="utf-8")
    return str(out)


def _ffmpeg_supports_ass(exe: str) -> bool:
    try:
        proc = subprocess.run(
            [exe, "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        return proc.returncode == 0 and bool(re.search(r"\bass\b", proc.stdout))
    except Exception:
        return False


def _replace_file_with_retry(
    src: Path,
    dst: Path,
    *,
    attempts: int = 8,
    initial_delay: float = 0.15,
) -> Path:
    last_exc: Optional[BaseException] = None
    delay = initial_delay
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return dst
        except (PermissionError, OSError) as exc:
            last_exc = exc
            if attempt + 1 >= attempts:
                break
            time.sleep(delay)
            delay = min(1.5, delay * 2.0)
    assert last_exc is not None
    raise last_exc


def _resolve_ffmpeg_exe(require_ass: bool = False) -> str:
    candidates: List[str] = []
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            candidates.append(exe)
    except Exception:
        pass
    which_exe = shutil.which("ffmpeg")
    if which_exe and which_exe not in candidates:
        candidates.append(which_exe)

    if not candidates:
        raise RuntimeError("No ffmpeg executable found via imageio_ffmpeg or PATH")

    if require_ass:
        for exe in candidates:
            if _ffmpeg_supports_ass(exe):
                return exe
    return candidates[0]


def burn_subtitles_into_video(video_path: str, ass_path: str, output_path: Optional[str] = None) -> str:
    """Burn an ASS subtitle file into video_path while preserving audio track."""
    src = Path(video_path).resolve()
    ass = Path(ass_path).resolve()
    dst = Path(output_path).resolve() if output_path else src
    if not src.exists():
        raise FileNotFoundError(f"Video not found for subtitle burn-in: {src}")
    if not ass.exists():
        raise FileNotFoundError(f"ASS subtitle file not found: {ass}")

    ffmpeg = _resolve_ffmpeg_exe(require_ass=True)
    work_dir = dst.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    temp_ass: Optional[Path] = None
    if ass.parent != work_dir:
        temp_ass = work_dir / f".__sub_{os.getpid()}.ass"
        shutil.copyfile(ass, temp_ass)
        ass_name = temp_ass.name
    else:
        ass_name = ass.name

    temp_out = work_dir / f".__burn_{dst.stem}.tmp.mp4"
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-vf",
        f"ass={ass_name}",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        str(temp_out),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg subtitle burn-in failed (exit {proc.returncode}): {proc.stderr[-1000:]}")
        try:
            _replace_file_with_retry(temp_out, dst)
            return str(dst)
        except PermissionError:
            fallback_dst = dst.with_name(f"{dst.stem}_subtitled{dst.suffix}")
            _replace_file_with_retry(temp_out, fallback_dst)
            logger.warning(f"Destination {dst} locked by another process; saved subtitled video to {fallback_dst}")
            return str(fallback_dst)
    finally:
        if temp_ass is not None and temp_ass.exists():
            temp_ass.unlink(missing_ok=True)
        if temp_out.exists():
            temp_out.unlink(missing_ok=True)


async def translate_subtitles_with_chat_model(
    chat_model: Any,
    lines: Dict[int, str],
    target_language: str = "Vietnamese",
) -> Dict[int, str]:
    """Translate spoken shot dialogue lines into target_language via chat_model."""
    if not lines or chat_model is None:
        return {}
    if not (hasattr(chat_model, "ainvoke") or hasattr(chat_model, "invoke")):
        return dict(lines)

    payload = {str(k): v for k, v in lines.items()}
    prompt = (
        f"Translate the following numbered anime/film dialogue lines naturally into {target_language} for professional on-screen subtitles. "
        "Follow standard anime fansubbing/localization norms: translate idioms naturally into idiomatic phrasing rather than word-for-word literal phrasing (e.g. translate urgent rescue/panic phrases dynamically and naturally). "
        "Keep them concise, vivid, punchy, and conversational. Preserve exact JSON keys and return ONLY valid JSON.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    try:
        if hasattr(chat_model, "ainvoke"):
            response = await chat_model.ainvoke(prompt)
        else:
            response = chat_model.invoke(prompt)
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
        raw = str(content).strip()
        unwrapped = strip_markdown_json_wrapper(raw)
        candidates: List[str] = []
        seen: set[str] = set()
        for candidate in (raw, strip_trailing_commas(raw), unwrapped, strip_trailing_commas(unwrapped)):
            if candidate and candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
        for source in (unwrapped, raw):
            for block in iter_json_blocks(source):
                for variant in (block, strip_trailing_commas(block)):
                    if variant and variant not in seen:
                        seen.add(variant)
                        candidates.append(variant)

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                result = {int(k): str(v).strip() for k, v in parsed.items() if str(k).lstrip("-").isdigit() and str(v).strip()}
                if result:
                    return result
    except Exception as exc:
        logger.warning(f"Subtitle translation fallback failed ({exc}); keeping original lines.")
    return dict(lines)
