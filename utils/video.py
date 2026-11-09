import logging
import os
import uuid
import requests
from typing import Any, List, Optional, Tuple
from moviepy import CompositeVideoClip, VideoFileClip, afx, concatenate_videoclips, vfx
from utils.retry import download_retry


@download_retry
def download_video(url, save_path):
    try:
        logging.info(f"Downloading video from {url} to {save_path}")

        response = requests.get(url, stream=True, timeout=(10, 300))
        response.raise_for_status()  # Check request succeeded
    
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logging.info(f"Video downloaded successfully to {save_path}")
    
    except Exception as e:
        logging.error(f"Error downloading video: {e}")
        raise e


def _valid_clip_size(clip) -> tuple[int, int] | None:
    size = getattr(clip, "size", None)
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        return None
    width, height = size
    if isinstance(width, bool) or isinstance(height, bool):
        return None
    if isinstance(width, (int, float)) and isinstance(height, (int, float)) and width > 0 and height > 0:
        return (int(width), int(height))
    return None


def normalize_clips_for_concat(clips, target_size: tuple[int, int] | None = None):
    """Ensure all video clips share one (width, height) before MoviePy concatenation.

    MoviePy's default ``method="chain"`` assumes every clip emits identical raw
    frame dimensions; mixing e.g. 1248x704 and 1280x720 clips corrupts pixels
    and desynchronizes audio across shot boundaries.
    """
    clip_list = list(clips)
    valid_sizes = [s for s in (_valid_clip_size(clip) for clip in clip_list) if s is not None]
    if target_size is None:
        unique_sizes = set(valid_sizes)
        if len(unique_sizes) <= 1:
            return clip_list, []
        target_size = max(unique_sizes, key=lambda s: (s[0] * s[1], s[0], s[1]))

    normalized = []
    extra_clips = []
    for clip in clip_list:
        size = _valid_clip_size(clip)
        if size is None or size == target_size:
            normalized.append(clip)
            continue
        resize_fn = getattr(clip, "resized", None) or getattr(clip, "resize", None)
        if resize_fn is None:
            normalized.append(clip)
            continue
        resized_clip = resize_fn(target_size)
        normalized.append(resized_clip)
        extra_clips.append(resized_clip)
    return normalized, extra_clips


def _apply_audio_effects(clip: Any, effects: List[Any], extra_clips: List[Any]) -> Any:
    audio = getattr(clip, "audio", None)
    if audio is None or not effects:
        return clip
    faded_audio = audio.with_effects(effects)
    extra_clips.append(faded_audio)
    with_audio = clip.with_audio(faded_audio)
    extra_clips.append(with_audio)
    return with_audio


def compose_timeline_clips(
    clips: List[Any],
    shot_descriptions: List[Any],
    spoken_shot_mask: Optional[List[bool]] = None,
    fade_black_sec: float = 0.35,
    crossfade_sec: float = 0.45,
    max_speed_factor: float = 2.5,
) -> Tuple[Any, List[float], List[float], List[Any]]:
    """Compose final video timeline with speed-paced silent action and editorial transitions.

    - Normalizes clip resolutions upfront.
    - Applies ``vfx.MultiplySpeed(final_duration=...)`` ONLY on non-dialogue shots
      whose ``target_duration_sec`` is shorter than the raw clip, keeping both
      ``first_frame`` and ``last_frame`` intact while accelerating anime combat tempo.
    - Supports ``transition_in`` values: ``"cut"``, ``"fade_black"``, and ``"crossfade"``.
    - Returns ``(final_clip, shot_start_offsets, shot_durations, extra_clips_to_close)``.
    """
    clip_list = list(clips)
    if not clip_list:
        raise ValueError("Cannot compose an empty clip list.")

    normalized_clips, extra_clips = normalize_clips_for_concat(clip_list)
    target_size = next((s for s in (_valid_clip_size(c) for c in normalized_clips) if s is not None), None)

    paced_clips: List[Any] = []
    durations: List[float] = []
    for i, clip in enumerate(normalized_clips):
        sd = shot_descriptions[i] if i < len(shot_descriptions) else None
        raw_dur = float(getattr(clip, "duration", None) or 8.0)
        has_speech = bool(spoken_shot_mask[i]) if spoken_shot_mask is not None and i < len(spoken_shot_mask) else False
        target_dur = getattr(sd, "target_duration_sec", None) if sd is not None else None

        active_clip = clip
        if (
            not has_speech
            and target_dur is not None
            and isinstance(target_dur, (int, float))
            and not isinstance(target_dur, bool)
            and float(target_dur) >= 0.8
            and raw_dur > 0.8
            and float(target_dur) < raw_dur - 0.15
        ):
            clamped_dur = max(float(target_dur), raw_dur / max_speed_factor, 0.8)
            if clamped_dur < raw_dur - 0.05:
                active_clip = clip.with_effects([vfx.MultiplySpeed(final_duration=clamped_dur)])
                extra_clips.append(active_clip)
        paced_clips.append(active_clip)
        durations.append(float(getattr(active_clip, "duration", None) or raw_dur))

    n = len(paced_clips)
    transitions: List[str] = ["cut"]
    for i in range(1, n):
        sd = shot_descriptions[i] if i < len(shot_descriptions) else None
        t_in = str(getattr(sd, "transition_in", "cut") or "cut").strip().lower()
        if t_in not in {"cut", "crossfade", "fade_black"}:
            t_in = "cut"
        transitions.append(t_in)

    video_effects: List[List[Any]] = [[] for _ in range(n)]
    audio_effects: List[List[Any]] = [[] for _ in range(n)]
    offsets: List[float] = [0.0] * n
    has_crossfade = False

    for i in range(1, n):
        prev_start = offsets[i - 1]
        prev_dur = durations[i - 1]
        curr_dur = durations[i]
        t_in = transitions[i]

        if t_in == "fade_black":
            fade_dur = min(fade_black_sec, prev_dur * 0.2, curr_dur * 0.2)
            if fade_dur >= 0.05:
                video_effects[i - 1].append(vfx.FadeOut(fade_dur))
                audio_effects[i - 1].append(afx.AudioFadeOut(fade_dur))
                video_effects[i].append(vfx.FadeIn(fade_dur))
                audio_effects[i].append(afx.AudioFadeIn(fade_dur))
            offsets[i] = prev_start + prev_dur
        elif t_in == "crossfade":
            max_safe_overlap = max(0.0, prev_dur - 0.25)
            xfade = min(crossfade_sec, prev_dur * 0.25, curr_dur * 0.25, max_safe_overlap)
            if xfade >= 0.05:
                has_crossfade = True
                audio_effects[i - 1].append(afx.AudioFadeOut(xfade))
                video_effects[i].append(vfx.CrossFadeIn(xfade))
                audio_effects[i].append(afx.AudioFadeIn(xfade))
                offsets[i] = prev_start + prev_dur - xfade
            else:
                offsets[i] = prev_start + prev_dur
        else:
            offsets[i] = prev_start + prev_dur

    processed_clips: List[Any] = []
    for i, clip in enumerate(paced_clips):
        current = clip
        if video_effects[i]:
            current = current.with_effects(video_effects[i])
            extra_clips.append(current)
        if audio_effects[i]:
            current = _apply_audio_effects(current, audio_effects[i], extra_clips)
        processed_clips.append(current)

    if has_crossfade:
        positioned: List[Any] = []
        for i, clip in enumerate(processed_clips):
            placed = clip.with_start(offsets[i])
            extra_clips.append(placed)
            positioned.append(placed)
        final_clip = CompositeVideoClip(positioned, size=target_size) if target_size else CompositeVideoClip(positioned)
    else:
        final_clip = concatenate_videoclips(processed_clips, method="compose")

    return final_clip, offsets, durations, extra_clips


def concatenate_video_files(
    video_paths,
    output_path,
    codec="libx264",
    preset="medium",
    audio_codec="aac",
    scene_fade_sec: float = 0.0,
):
    """Concatenate video files, releasing every ffmpeg reader even on failure.

    When ``scene_fade_sec > 0`` and multiple scene clips are joined, applies a
    smooth fade-out/fade-in across scene boundaries and writes atomically via a
    temporary MP4 before replacing ``output_path``.
    """
    clips = []
    extra_clips = []
    final = None
    out_dir = os.path.dirname(output_path) or "."
    out_stem, out_ext = os.path.splitext(os.path.basename(output_path))
    tmp_output = os.path.join(out_dir, f".{out_stem}.{uuid.uuid4().hex[:8]}.tmp{out_ext or '.mp4'}")
    try:
        for path in video_paths:
            clips.append(VideoFileClip(path))
        normalized_clips, extra_clips = normalize_clips_for_concat(clips)
        active_clips = normalized_clips
        if scene_fade_sec > 0 and len(normalized_clips) > 1:
            active_clips = []
            for i, clip in enumerate(normalized_clips):
                dur = float(getattr(clip, "duration", None) or 0.0)
                fade_cap = dur * 0.2 if dur > 0 else float(scene_fade_sec)
                fade_dur = min(float(scene_fade_sec), fade_cap)
                current = clip
                if fade_dur >= 0.05:
                    v_fx = []
                    a_fx = []
                    if i > 0:
                        v_fx.append(vfx.FadeIn(fade_dur))
                        a_fx.append(afx.AudioFadeIn(fade_dur))
                    if i < len(normalized_clips) - 1:
                        v_fx.append(vfx.FadeOut(fade_dur))
                        a_fx.append(afx.AudioFadeOut(fade_dur))
                    if v_fx:
                        current = current.with_effects(v_fx)
                        extra_clips.append(current)
                    if a_fx:
                        current = _apply_audio_effects(current, a_fx, extra_clips)
                active_clips.append(current)
        final = concatenate_videoclips(active_clips, method="compose")
        final.write_videofile(tmp_output, codec=codec, audio_codec=audio_codec, preset=preset)
    finally:
        if final is not None:
            final.close()
        for clip in extra_clips:
            clip.close()
        for clip in clips:
            clip.close()
    if os.path.exists(tmp_output):
        os.replace(tmp_output, output_path)
    return output_path
