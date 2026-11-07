import os
import re
import asyncio
import logging
import requests
from typing import Optional, List, Tuple
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

def extract_speech_from_shot(audio_desc: str = "", visual_desc: str = "") -> str:
    """Extract spoken lines or dialogue from storyboard shot descriptions."""
    audio_desc = audio_desc or ""
    visual_desc = visual_desc or ""

    # 1. Match [Speaker] tag: [Speaker] Alice (Happy): Hello!
    speaker_match = re.search(r'\[Speaker\][^:]*:\s*(.+)', audio_desc, re.IGNORECASE)
    if speaker_match:
        text = speaker_match.group(1).strip().strip('"\'')
        if text:
            return text

    # 2. Match quotes in audio_desc: "Hello world"
    quote_match = re.search(r'["\']([^"\']{4,})["\']', audio_desc)
    if quote_match:
        return quote_match.group(1).strip()

    # 3. Match character speech in visual_desc: says: "..."
    says_match = re.search(r'says?:\s*["\']([^"\']+)["\']', visual_desc, re.IGNORECASE)
    if says_match:
        return says_match.group(1).strip()

    # 4. Filter out [Sound Effect] tag
    cleaned = re.sub(r'\[Sound Effect\][^\n]*', '', audio_desc, flags=re.IGNORECASE).strip()
    if len(cleaned) > 5 and not cleaned.lower().startswith('none'):
        return cleaned

    return ""

def mix_video_and_audio(
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
