import logging
import os
import re
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


def resolve_anime_shot_pacing(sd: Any, raw_dur: float = 8.0) -> float:
    """Dynamically determine natural anime tempo and shot duration based on dialogue and action type."""
    explicit = getattr(sd, "target_duration_sec", None)
    if explicit is not None and isinstance(explicit, (int, float)) and float(explicit) >= 1.0:
        return float(explicit)

    audio = str(getattr(sd, "audio_desc", "") or "")
    vis = str(getattr(sd, "visual_desc", "") or "").lower()

    dialogue_match = re.search(
        r"\[(?:Speaker|Thought|Inner\s*Monologue)\]\s*([^:\n]+?)\s*:\s*(?:'|\")?([^'\"]+)(?:'|\")?",
        audio,
        re.IGNORECASE,
    )

    if not dialogue_match:
        if any(w in vis for w in ["speed", "sprint", "blitz", "dash", "lightning", "skid", "screech", "impact", "flash"]):
            return 2.5
        elif any(w in vis for w in ["establishing", "wide high-angle", "extreme wide", "quiet", "night sky"]):
            return 4.5
        else:
            return 3.5

    line = dialogue_match.group(2).strip()
    line = re.sub(r"\(Vietsub:[^)]*\)", "", line).strip()
    words = line.split()
    word_count = len(words)

    if word_count <= 4:
        return 3.2
    elif word_count <= 8:
        return 4.5
    elif word_count <= 13:
        return 5.8
    else:
        return 7.5


def resolve_and_apply_anime_fx(clip: Any, sd: Any, extra_clips: Optional[List[Any]] = None) -> Any:
    """Inspect shot description for anime visual effects tags and apply them in sequence.

    Supported anime VFX tags across visual_desc, motion_desc, audio_desc:
    - [AnimeFX: Flashback] or [Flashback] / [Hồi tưởng] / [Ký ức] -> apply_anime_flashback
    - [AnimeFX: DramaticFocus] or [DramaticFocus] / [Spotlight] -> apply_anime_dramatic_focus
    - [AnimeFX: Letterbox] or [Letterbox] / [CinemaBars] -> apply_anime_letterbox
    - [AnimeFX: WhiteFlash] or [WhiteFlash] / [Chớp trắng] -> apply_anime_white_flash
    - [AnimeFX: ActionBurst] or [ActionBurst] / [Blitz] -> apply_anime_action_burst
    - [AnimeFX: SlowMo] or [SlowMo] / [BulletTime] / [Chậm lại] -> apply_bullet_time_slowmo
    - [AnimeFX: Vanish] or [Vanish] / [Biến mất] / [Shunpo] / [Soru] -> apply_anime_vanish
    - [AnimeFX: Impact] or [Impact] / [ImpactFrame] / [Sakuga] -> apply_impact_frames
    - [AnimeFX: SpeedLines] or [SpeedLines] / [Tốc độ] / [RushLines] -> apply_anime_speed_lines
    - [AnimeFX: ScreenShake] or [ScreenShake] / [CameraShake] / [Rung lắc] -> apply_screen_shake
    """
    if sd is None or clip is None:
        return clip
    combined_desc = " ".join(
        str(getattr(sd, attr, "") or "")
        for attr in ("visual_desc", "motion_desc", "audio_desc")
    )
    if not combined_desc:
        return clip

    c = clip
    try:
        from utils.anime_fx import (
            apply_impact_frames,
            apply_anime_speed_lines,
            apply_screen_shake,
            apply_anime_vanish,
            apply_anime_frame_skip_streak,
            apply_bullet_time_slowmo,
            apply_anime_action_burst,
            apply_anime_flashback,
            apply_anime_dramatic_focus,
            apply_anime_letterbox,
            apply_anime_white_flash,
            apply_anime_ken_burns_drift,
            apply_anime_snap_zoom,
            apply_anime_chromatic_aberration,
            apply_anime_film_grain_and_halation,
            apply_anime_whip_pan,
        )
    except ImportError:
        return clip

    try:
        dur = float(getattr(c, "duration", 4.0) or 4.0)

        # 1. Snap Zoom (Sakuga crash zoom on shock / eye reaction)
        if re.search(r"\[(?:AnimeFX:\s*)?(?:SnapZoom|Snap_Zoom|CrashZoom|Cận\s*cảnh\s*sốc)\]", combined_desc, re.IGNORECASE):
            c = apply_anime_snap_zoom(c, trigger_time_sec=min(0.25, dur * 0.1), zoom_ratio=1.22, duration=0.22)
            if extra_clips is not None:
                extra_clips.append(c)

        # 2. Chromatic Aberration (RGB split on sonic booms / energy surge)
        if re.search(r"\[(?:AnimeFX:\s*)?(?:Chromatic|RGB_Split|Lệch\s*màu)\]", combined_desc, re.IGNORECASE):
            c = apply_anime_chromatic_aberration(c, trigger_time_sec=0.0, duration=0.35, max_shift=7)
            if extra_clips is not None:
                extra_clips.append(c)

        # 3. Flashback / Memory treatment
        if re.search(r"\[(?:AnimeFX:\s*)?(?:Flashback|Hồi\s*tưởng|Ký\s*ức|Memory)\]", combined_desc, re.IGNORECASE):
            tint = "monochrome" if re.search(r"monochrome|black[- ]and[- ]white|silver", combined_desc, re.IGNORECASE) else "sepia"
            c = apply_anime_flashback(c, tint_mode=tint, glow_intensity=0.35, vignette_strength=0.45)
            if extra_clips is not None:
                extra_clips.append(c)

        # 4. Dramatic Focus (Spotlight on character eyes/reaction)
        if re.search(r"\[(?:AnimeFX:\s*)?(?:DramaticFocus|Dramatic_Focus|Spotlight|Tập\s*trung)\]", combined_desc, re.IGNORECASE):
            c = apply_anime_dramatic_focus(c, focus_radius=0.38, darkness=0.60)
            if extra_clips is not None:
                extra_clips.append(c)

        # 5. Cinematic Letterbox bars
        if re.search(r"\[(?:AnimeFX:\s*)?(?:Letterbox|CinemaBars|Widescreen)\]", combined_desc, re.IGNORECASE):
            c = apply_anime_letterbox(c, bar_ratio=0.10)
            if extra_clips is not None:
                extra_clips.append(c)

        # 6. White Flash
        if re.search(r"\[(?:AnimeFX:\s*)?(?:WhiteFlash|White_Flash|Chớp\s*trắng)\]", combined_desc, re.IGNORECASE):
            c = apply_anime_white_flash(c, trigger_time_sec=0.0, duration=0.35, mode="in")
            if extra_clips is not None:
                extra_clips.append(c)

        # 7. Action Burst preset (if tagged)
        if re.search(r"\[(?:AnimeFX:\s*)?(?:ActionBurst|Blitz|SonicBurst)\]", combined_desc, re.IGNORECASE):
            c = apply_anime_action_burst(c, trigger_time_sec=min(0.2, dur * 0.1), with_impact=True, with_speed_lines=True, with_shake=True)
            if extra_clips is not None:
                extra_clips.append(c)

        # 8. Bullet-time Slow Motion
        if re.search(r"\[(?:AnimeFX:\s*)?(?:SlowMo|Slow_Mo|BulletTime|Thời\s*gian\s*chậm)\]", combined_desc, re.IGNORECASE):
            c = apply_bullet_time_slowmo(c, slow_start_sec=min(0.5, dur * 0.2), slow_dur_sec=min(1.2, dur * 0.4), speed_factor=0.35)
            if extra_clips is not None:
                extra_clips.append(c)
            dur = float(getattr(c, "duration", dur) or dur)

        # 8b. Frame Skip / Speed Streak Ellipsis (skips difficult physical motion with high-speed blur & energy trail)
        if re.search(r"\[(?:AnimeFX:\s*)?(?:FrameSkip|Frame_Skip|SpeedStreak|Speed_Streak|V[eệ]t(?:TocDo|HieuUng)|V[eệ]t\s*t[oố]c\s*đ[oộ])\]", combined_desc, re.IGNORECASE):
            t_skip = min(1.2, dur * 0.35)
            c = apply_anime_frame_skip_streak(c, trigger_time_sec=t_skip, duration=0.40)
            if extra_clips is not None:
                extra_clips.append(c)

        # 9. Vanish / Shunpo effect
        if re.search(r"\[(?:AnimeFX:\s*)?(?:Vanish|Bién\s*mất|Biến\s*mất|Shunpo|Soru)\]", combined_desc, re.IGNORECASE):
            t_vanish = min(1.0, dur * 0.35)
            c = apply_anime_vanish(c, vanish_time_sec=t_vanish, duration=0.35)
            if extra_clips is not None:
                extra_clips.append(c)

        # 10. Sakuga Impact Frames
        if re.search(r"\[(?:AnimeFX:\s*)?(?:Impact|ImpactFrames?|Sakuga)\]", combined_desc, re.IGNORECASE):
            t_impact = min(0.3, dur * 0.1)
            c = apply_impact_frames(c, trigger_time_sec=t_impact, num_frames=3, style="inverted")
            if extra_clips is not None:
                extra_clips.append(c)

        # 11. Speed Lines (Horizontal or Radial)
        if re.search(r"\[(?:AnimeFX:\s*)?(?:SpeedLines|Speed_Lines|Tốc\s*độ|RushLines)\]", combined_desc, re.IGNORECASE):
            line_type = "radial" if re.search(r"radial|focus|zoom", combined_desc, re.IGNORECASE) else "horizontal"
            c = apply_anime_speed_lines(c, start_sec=0.0, end_sec=dur, line_type=line_type, intensity=0.5)
            if extra_clips is not None:
                extra_clips.append(c)

        # 12. Screen Shake
        if re.search(r"\[(?:AnimeFX:\s*)?(?:ScreenShake|Screen_Shake|CameraShake|Rung\s*lắc|Recoil)\]", combined_desc, re.IGNORECASE):
            t_shake = min(0.15, dur * 0.1)
            c = apply_screen_shake(c, trigger_time_sec=t_shake, duration=0.35, max_displacement=16)
            if extra_clips is not None:
                extra_clips.append(c)

        # 13. Dynamic Camera Drift (Ken Burns) for non-action conversational/scenery shots
        # Eliminates frozen static frames across the entire episode!
        is_action_beat = bool(re.search(r"ActionBurst|Vanish|FrameSkip|SpeedStreak|SpeedLines|SnapZoom|ScreenShake", combined_desc, re.IGNORECASE))
        if not is_action_beat:
            # Alternating subtle drift creates natural cinematic flow
            drift_dir = "push_in" if getattr(sd, "idx", 0) % 2 == 0 else "pan_right"
            c = apply_anime_ken_burns_drift(c, direction=drift_dir, zoom_ratio=1.04)
            if extra_clips is not None:
                extra_clips.append(c)

        # 14. Organic 2D Anime Cel Texture & Halation
        c = apply_anime_film_grain_and_halation(c, grain_intensity=0.05, halation_warmth=0.08)
        if extra_clips is not None:
            extra_clips.append(c)

    except Exception as e:
        logging.warning(f"Error applying anime FX to shot: {e}")

    return c


def compose_timeline_clips(
    clips: List[Any],
    shot_descriptions: List[Any],
    spoken_shot_mask: Optional[List[bool]] = None,
    fade_black_sec: float = 0.35,
    crossfade_sec: float = 0.45,
    max_speed_factor: float = 2.5,
) -> Tuple[Any, List[float], List[float], List[Any]]:
    """Compose final video timeline with full cinematic duration and authentic anime effects.

    - Retains full clip durations so dialogue, soundscapes, and cinematic atmosphere
      never get prematurely cut off.
    - Applies AnimeFX (Vanish, SpeedLines, ScreenShake, Flashback, DramaticFocus, etc.)
      dynamically per shot description.
    - Supports transitions: 'cut', 'crossfade', 'fade_black', 'white_flash'.
    - Returns (final_clip, shot_start_offsets, shot_durations, extra_clips_to_close).
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

        # Keep full natural duration - do NOT trim or cut clips in half!
        active_clip = clip
        active_clip = resolve_and_apply_anime_fx(active_clip, sd, extra_clips)
        paced_clips.append(active_clip)
        durations.append(float(getattr(active_clip, "duration", None) or raw_dur))

    n = len(paced_clips)
    transitions: List[str] = ["cut"]
    for i in range(1, n):
        sd = shot_descriptions[i] if i < len(shot_descriptions) else None
        t_in = str(getattr(sd, "transition_in", "cut") or "cut").strip().lower()
        if t_in not in {"cut", "crossfade", "fade_black", "white_flash", "whip_pan"}:
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
        elif t_in == "white_flash":
            flash_dur = min(0.35, prev_dur * 0.15, curr_dur * 0.15)
            if flash_dur >= 0.05:
                from utils.anime_fx import apply_anime_white_flash
                paced_clips[i - 1] = apply_anime_white_flash(paced_clips[i - 1], trigger_time_sec=max(0.0, prev_dur - flash_dur), duration=flash_dur, mode="out")
                paced_clips[i] = apply_anime_white_flash(paced_clips[i], trigger_time_sec=0.0, duration=flash_dur, mode="in")
                extra_clips.append(paced_clips[i - 1])
                extra_clips.append(paced_clips[i])
            offsets[i] = prev_start + prev_dur
        elif t_in == "whip_pan":
            whip_dur = min(0.25, prev_dur * 0.15, curr_dur * 0.15)
            if whip_dur >= 0.05:
                from utils.anime_fx import apply_anime_whip_pan
                paced_clips[i - 1] = apply_anime_whip_pan(paced_clips[i - 1], trigger_time_sec=max(0.0, prev_dur - whip_dur), duration=whip_dur, direction="right")
                paced_clips[i] = apply_anime_whip_pan(paced_clips[i], trigger_time_sec=0.0, duration=whip_dur, direction="right")
                extra_clips.append(paced_clips[i - 1])
                extra_clips.append(paced_clips[i])
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
