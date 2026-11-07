import os
import re
import asyncio
import logging
import requests
from typing import Optional, List, Tuple, Any, Dict
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip

logger = logging.getLogger(__name__)

async def generate_speech_edge_tts(text: str, output_path: str, voice: str = "vi-VN-NamMinhNeural", rate: str = "+0%") -> str:
    """Generate high-quality speech using Edge-TTS (Microsoft Neural Voice)."""
    import edge_tts
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    communicator = edge_tts.Communicate(text, voice, rate=rate)
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
        # Default fallback to high quality Edge-TTS
        edge_voice = voice.strip() if "Neural" in (voice or "") else "vi-VN-NamMinhNeural"
        return await generate_speech_edge_tts(text, output_path, voice=edge_voice)

DEFAULT_FEMALE_VOICE_ELEVENLABS = "EXAVITQu4vr4xnSDxMaL"  # Bella (Expressive Female)
DEFAULT_MALE_VOICE_ELEVENLABS = "pNInz6obpgDQGcFmaJgB"    # Adam (Deep Male)
DEFAULT_FEMALE_VOICE_EDGE_TTS = "vi-VN-HoaiMyNeural"      # Hoài My (Nữ)
DEFAULT_MALE_VOICE_EDGE_TTS = "vi-VN-NamMinhNeural"        # Nam Minh (Nam)

def extract_dialogue_from_shot(audio_desc: str = "", visual_desc: str = "") -> Tuple[str, str]:
    """Extract character speaker and dialogue line from storyboard shot descriptions.
    
    Returns (speaker_name, dialogue_text). If no specific speaker is identified, speaker is empty.
    """
    audio_desc = audio_desc or ""
    visual_desc = visual_desc or ""

    # 1. Match [Speaker] tag: [Speaker] Alice (Happy): Hello!
    speaker_tag_match = re.search(r'\[Speaker\]\s*([^:\(]+?)(?:\s*\([^)]*\))?\s*:\s*(.+)', audio_desc, re.IGNORECASE)
    if speaker_tag_match:
        speaker = speaker_tag_match.group(1).strip().strip('<>"\'')
        text = speaker_tag_match.group(2).strip().strip('"\'')
        if text:
            return speaker, text

    # 2. Match character speech in visual_desc: <Alice> (female...) says: "..."
    char_says_match = re.search(r'<([^>]+)>[^:]*says?:\s*["\']([^"\']+)["\']', visual_desc, re.IGNORECASE)
    if char_says_match:
        speaker = char_says_match.group(1).strip()
        text = char_says_match.group(2).strip()
        if text:
            return speaker, text

    # 3. Match general Character says: Alice says: "..."
    gen_says_match = re.search(r'([A-Za-z0-9_\s]{2,20})\s+says?:\s*["\']([^"\']+)["\']', visual_desc, re.IGNORECASE)
    if gen_says_match:
        speaker = gen_says_match.group(1).strip()
        text = gen_says_match.group(2).strip()
        if text:
            return speaker, text

    # 4. Match quotes in audio_desc: "Hello world"
    quote_match = re.search(r'["\']([^"\']{4,})["\']', audio_desc)
    if quote_match:
        return "", quote_match.group(1).strip()

    # 5. Filter out [Sound Effect] tag
    cleaned = re.sub(r'\[Sound Effect\][^\n]*', '', audio_desc, flags=re.IGNORECASE).strip()
    if len(cleaned) > 5 and not cleaned.lower().startswith('none'):
        return "", cleaned

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

    # 2. Check character description/gender from characters metadata
    is_female = False
    is_male = False

    female_keywords = {"female", "woman", "girl", "nữ", "cô", "nàng", "bà", "chị", "em", "phụ nữ"}
    male_keywords = {"male", "man", "boy", "nam", "chàng", "ông", "anh", "đàn ông"}

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

def mix_multitrack_video_and_audio(
    video_path: str,
    output_path: str,
    speech_segments: List[Tuple[float, str]],  # [(start_offset_seconds, audio_file_path), ...]
    bgm_path: Optional[str] = None,
    bgm_volume: float = 0.15,
) -> str:
    """Mix multiple character speech clips timed across the video timeline, along with BGM."""
    if not os.path.exists(video_path):
        return video_path

    video = VideoFileClip(video_path)
    audio_clips = []

    # 1. Load each character's speech clip positioned at its start time
    for start_time, voice_path in speech_segments:
        if voice_path and os.path.exists(voice_path):
            try:
                clip = AudioFileClip(voice_path)
                if start_time < video.duration:
                    clip = clip.with_start(start_time)
                    audio_clips.append(clip)
            except Exception as e:
                logger.warning(f"Error loading speech clip {voice_path} at t={start_time}: {e}")

    # 2. Load background music
    if bgm_path and os.path.exists(bgm_path):
        try:
            import math
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
            import shutil
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
            clip.close()
        video.close()

        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(temp_target, output_path)
        return output_path
    except Exception as e:
        logger.error(f"Error mixing audio into video: {e}")
        video.close()
        for clip in audio_clips:
            clip.close()
        return video_path

def mix_video_and_audio(
    video_path: str,
    output_path: str,
    voice_path: Optional[str] = None,
    bgm_path: Optional[str] = None,
    bgm_volume: float = 0.15,
) -> str:
    """Mix single voiceover and BGM into a video file (backwards compatibility)."""
    segments = [(0.0, voice_path)] if voice_path else []
    return mix_multitrack_video_and_audio(
        video_path=video_path,
        output_path=output_path,
        speech_segments=segments,
        bgm_path=bgm_path,
        bgm_volume=bgm_volume,
    )

def _unused_mix_video_and_audio(
    video_path: str,
    output_path: str,
    voice_path: Optional[str] = None,
    bgm_path: Optional[str] = None,
    bgm_volume: float = 0.15,
) -> str:
    """Mix generated voiceover and optional background music into a video file."""
    if not os.path.exists(video_path):
        return video_path

    video = VideoFileClip(video_path)
    audio_clips = []

    if voice_path and os.path.exists(voice_path):
        try:
            voice_clip = AudioFileClip(voice_path)
            if voice_clip.duration > video.duration:
                voice_clip = voice_clip.subclipped(0, video.duration)
            audio_clips.append(voice_clip)
        except Exception as e:
            logger.warning(f"Error loading voiceover {voice_path}: {e}")

    if bgm_path and os.path.exists(bgm_path):
        try:
            bgm_clip = AudioFileClip(bgm_path).with_volume_scaled(bgm_volume)
            if bgm_clip.duration < video.duration:
                import math
                repeats = math.ceil(video.duration / bgm_clip.duration)
                from moviepy import concatenate_audioclips
                bgm_clip = concatenate_audioclips([bgm_clip] * repeats)
            bgm_clip = bgm_clip.subclipped(0, video.duration)
            audio_clips.append(bgm_clip)
        except Exception as e:
            logger.warning(f"Error loading BGM {bgm_path}: {e}")

    if not audio_clips:
        video.close()
        if video_path != output_path:
            import shutil
            shutil.copyfile(video_path, output_path)
        return output_path

    try:
        final_audio = CompositeAudioClip(audio_clips) if len(audio_clips) > 1 else audio_clips[0]
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
            clip.close()
        video.close()

        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(temp_target, output_path)
        return output_path
    except Exception as e:
        logger.error(f"Error mixing audio into video: {e}")
        video.close()
        for clip in audio_clips:
            clip.close()
        return video_path
