"""Anime Visual Effects & Directing Toolkit (AnimeFX).

Provides authentic Japanese anime directing & post-processing effects:
- Flashback / Memory Treatment (Sepia/Monochrome tint, Dreamy Bloom, Soft Vignette)
- Anime White Flash Transition (Awakening / Impact / Memory snap transition)
- Dramatic Spotlight / Iris Vignette (High-contrast focus on character's eyes/reaction)
- Cinematic Letterbox (2.39:1 widescreen dramatic bars)
- Impact Frames (Sakuga 2-3 frame monochrome/inverted flashes)
- High-Speed Vanish / Flicker Disappear (Shunpo / Soru with cyan electricity streaks)
- Anime Speed Lines Overlay (Radial focus lines & Horizontal rush streaks)
- Dynamic Screen Shake (Damped harmonic camera impact recoil)
- Bullet-Time Slow Motion (Time freeze / dilation before explosive action)
- Action Burst Preset (Integrated anime super-speed blitz composite)
"""

from __future__ import annotations

import math
import random
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np


def _create_radial_speed_lines(
    h: int, w: int, center: Tuple[int, int] | None = None, num_lines: int = 50, seed: int = 42
) -> np.ndarray:
    """Generate a single mask of radial anime speed lines radiating outward from focus center."""
    rng = np.random.RandomState(seed)
    mask = np.zeros((h, w), dtype=np.uint8)
    cx = center[0] if center else w // 2
    cy = center[1] if center else h // 2
    diag = int(math.hypot(w, h))

    angles = rng.uniform(0, 2 * math.pi, num_lines)
    thicknesses = rng.randint(2, 6, num_lines)
    inner_radii = rng.uniform(min(w, h) * 0.15, min(w, h) * 0.35, num_lines)

    for ang, th, r_in in zip(angles, thicknesses, inner_radii):
        x1 = int(cx + r_in * math.cos(ang))
        y1 = int(cy + r_in * math.sin(ang))
        x2 = int(cx + diag * math.cos(ang))
        y2 = int(cy + diag * math.sin(ang))
        cv2.line(mask, (x1, y1), (x2, y2), 255, int(th), lineType=cv2.LINE_AA)

    return mask


def _create_horizontal_speed_lines(h: int, w: int, num_lines: int = 40, seed: int = 42) -> np.ndarray:
    """Generate a single mask of horizontal anime speed lines rushing across screen."""
    rng = np.random.RandomState(seed)
    mask = np.zeros((h, w), dtype=np.uint8)
    y_coords = rng.randint(0, h, num_lines)
    lengths = rng.randint(w // 4, w, num_lines)
    x_starts = rng.randint(-w // 4, w // 2, num_lines)
    thicknesses = rng.randint(1, 4, num_lines)

    for y, l, x, th in zip(y_coords, lengths, x_starts, thicknesses):
        cv2.line(mask, (int(x), int(y)), (int(x + l), int(y)), 255, int(th), lineType=cv2.LINE_AA)

    return mask


def apply_anime_flashback(
    clip: Any,
    tint_mode: str = "sepia",
    glow_intensity: float = 0.35,
    vignette_strength: float = 0.40,
    soft_border: bool = True,
) -> Any:
    """Apply authentic Japanese anime flashback/memory aesthetic.

    - Warm nostalgic sepia or cool monochrome memory color grading.
    - Ethereal soft bloom/glow on highlights.
    - Gentle edge vignette and dreamy border softening.
    """
    w, h = clip.size

    # Precompute radial vignette mask
    y_indices, x_indices = np.indices((h, w))
    cx, cy = w / 2.0, h / 2.0
    max_dist = math.hypot(cx, cy)
    dist = np.hypot(x_indices - cx, y_indices - cy) / max_dist
    vignette_mask = np.clip(1.0 - (dist ** 1.8) * vignette_strength, 0.0, 1.0)[:, :, np.newaxis]

    # Precompute sepia transformation matrix
    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131],
    ])

    def _flashback_transform(get_frame, t: float):
        frame = get_frame(t)  # RGB uint8
        float_frame = frame.astype(np.float32)

        if tint_mode == "sepia":
            # Blend 65% sepia + 35% original color to maintain character recognizability
            sepia = cv2.transform(frame, sepia_matrix)
            graded = float_frame * 0.35 + sepia.astype(np.float32) * 0.65
        elif tint_mode == "monochrome":
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            gray_3c = np.repeat(gray[:, :, np.newaxis], 3, axis=2).astype(np.float32)
            # Slight cool tint (anime silver-grey memory)
            graded = gray_3c * np.array([0.92, 0.96, 1.05], dtype=np.float32)
        else:
            graded = float_frame

        # Soft ethereal bloom
        if glow_intensity > 0.05:
            small = cv2.resize(graded, (w // 4, h // 4))
            glow = cv2.GaussianBlur(small, (15, 15), 0)
            glow_large = cv2.resize(glow, (w, h))
            graded = graded * (1.0 - glow_intensity * 0.5) + glow_large * glow_intensity

        # Vignette darkening
        graded = graded * vignette_mask

        # Soft border dreaminess
        if soft_border:
            border_mask = np.clip((dist - 0.7) * 3.3, 0.0, 1.0)[:, :, np.newaxis]
            blurred_frame = cv2.GaussianBlur(graded, (11, 11), 0)
            graded = graded * (1.0 - border_mask * 0.5) + blurred_frame * (border_mask * 0.5)

        return np.clip(graded, 0, 255).astype(np.uint8)

    return clip.transform(_flashback_transform)


def apply_anime_dramatic_focus(
    clip: Any,
    focus_center: Optional[Tuple[int, int]] = None,
    focus_radius: float = 0.35,
    darkness: float = 0.65,
) -> Any:
    """Spotlight / Iris vignette focusing intensely on character reaction or eye-shot."""
    w, h = clip.size
    cx = focus_center[0] if focus_center else w // 2
    cy = focus_center[1] if focus_center else int(h * 0.45)
    max_dim = max(w, h)
    r_px = max_dim * focus_radius

    y_indices, x_indices = np.indices((h, w))
    dist = np.hypot(x_indices - cx, y_indices - cy)
    focus_mask = np.clip((dist - r_px * 0.6) / (r_px * 0.8), 0.0, 1.0)[:, :, np.newaxis]
    multiplier = 1.0 - focus_mask * darkness

    def _focus_transform(get_frame, t: float):
        frame = get_frame(t)
        out = frame.astype(np.float32) * multiplier
        return np.clip(out, 0, 255).astype(np.uint8)

    return clip.transform(_focus_transform)


def apply_anime_letterbox(clip: Any, bar_ratio: float = 0.10) -> Any:
    """Add cinematic widescreen 2.39:1 letterbox black bars for dramatic / combat tension."""
    w, h = clip.size
    bar_h = int(h * bar_ratio)
    if bar_h <= 0:
        return clip

    def _letterbox_transform(get_frame, t: float):
        frame = get_frame(t).copy()
        frame[:bar_h, :] = 0
        frame[h - bar_h :, :] = 0
        return frame

    return clip.transform(_letterbox_transform)


def apply_anime_white_flash(
    clip: Any,
    trigger_time_sec: float = 0.0,
    duration: float = 0.40,
    mode: str = "in",
) -> Any:
    """Sakuga white flash transition (awakening, explosive memory, or flash cut).

    - 'in': starts at pure white at trigger_time_sec and fades down into normal footage.
    - 'out': ramps from normal footage up to blinding white at trigger_time_sec + duration.
    - 'flash': quickly peaks to white at midpoint and settles.
    """
    end_time = trigger_time_sec + duration

    def _flash_transform(get_frame, t: float):
        frame = get_frame(t)
        if trigger_time_sec <= t < end_time:
            rel = (t - trigger_time_sec) / duration
            if mode == "in":
                alpha = max(0.0, 1.0 - rel ** 0.8)
            elif mode == "out":
                alpha = min(1.0, rel ** 1.2)
            else:  # 'flash'
                alpha = 1.0 - abs(rel - 0.5) * 2.0
                alpha = max(0.0, min(1.0, alpha ** 0.7))

            white = np.ones_like(frame, dtype=np.float32) * 255.0
            out = frame.astype(np.float32) * (1.0 - alpha) + white * alpha
            return np.clip(out, 0, 255).astype(np.uint8)
        return frame

    return clip.transform(_flash_transform)


def apply_impact_frames(
    clip: Any,
    trigger_time_sec: float,
    num_frames: int = 3,
    style: str = "inverted",
) -> Any:
    """Apply classic 2-3 frame anime sakuga impact flash at trigger_time_sec."""
    fps = getattr(clip, "fps", None) or 24
    frame_dur = 1.0 / fps
    end_time = trigger_time_sec + num_frames * frame_dur

    def _impact_transform(get_frame, t: float):
        frame = get_frame(t)
        if trigger_time_sec <= t < end_time:
            rel_f = int((t - trigger_time_sec) * fps)
            if rel_f == 0:
                # Frame 1: Inverted black-and-white high contrast sketch
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
                return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
            elif rel_f == 1:
                # Frame 2: Saturated white/cyan flash
                flash = frame.astype(np.float32) * 1.8 + 60
                return np.clip(flash, 0, 255).astype(np.uint8)
            else:
                # Frame 3: Inverted full color
                return (255 - frame).astype(np.uint8)
        return frame

    return clip.transform(_impact_transform)


def apply_anime_speed_lines(
    clip: Any,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    line_type: str = "radial",
    intensity: float = 0.45,
    color: Tuple[int, int, int] = (255, 255, 255),
    center: Optional[Tuple[int, int]] = None,
) -> Any:
    """Overlay animated anime speed lines across the specified time window."""
    fps = getattr(clip, "fps", None) or 24
    w, h = clip.size

    def _speed_line_transform(get_frame, t: float):
        frame = get_frame(t)
        max_t = end_sec if end_sec is not None else clip.duration
        if start_sec <= t <= max_t:
            f_idx = int(t * fps)
            seed = (f_idx * 17) % 1000
            if line_type == "horizontal":
                mask = _create_horizontal_speed_lines(h, w, num_lines=45, seed=seed)
            else:
                mask = _create_radial_speed_lines(h, w, center=center, num_lines=50, seed=seed)

            # Smooth mask blur
            mask_blur = cv2.GaussianBlur(mask, (5, 5), 0) / 255.0 * intensity
            mask_3c = np.repeat(mask_blur[:, :, np.newaxis], 3, axis=2)

            col_layer = np.zeros_like(frame, dtype=np.float32)
            col_layer[:, :] = color
            out = frame.astype(np.float32) * (1.0 - mask_3c) + col_layer * mask_3c
            return np.clip(out, 0, 255).astype(np.uint8)
        return frame

    return clip.transform(_speed_line_transform)


def apply_screen_shake(
    clip: Any,
    trigger_time_sec: float,
    duration: float = 0.35,
    max_displacement: int = 14,
    frequency: float = 28.0,
) -> Any:
    """Apply high-frequency damped harmonic camera recoil/shake on sudden impacts or sonic booms."""
    w, h = clip.size

    def _shake_transform(get_frame, t: float):
        if trigger_time_sec <= t < trigger_time_sec + duration:
            rel = t - trigger_time_sec
            decay = math.exp(-rel * 9.0)  # fast exponential damping
            dx = int(max_displacement * decay * math.sin(2 * math.pi * frequency * rel))
            dy = int((max_displacement * 0.7) * decay * math.cos(2 * math.pi * frequency * 1.3 * rel))

            frame = get_frame(t)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            shaken = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
            return shaken
        return get_frame(t)

    return clip.transform(_shake_transform)


def apply_anime_vanish(
    clip: Any,
    vanish_time_sec: float,
    duration: float = 0.35,
    direction: str = "right",
    trail_color: Tuple[int, int, int] = (0, 240, 255),
) -> Any:
    """Classic Shunpo / Vanish effect: character streaks with cyan energy motion blur and vanishes from frame."""
    w, h = clip.size
    fps = getattr(clip, "fps", None) or 24
    end_time = vanish_time_sec + duration

    def _vanish_transform(get_frame, t: float):
        frame = get_frame(t)
        if vanish_time_sec <= t < end_time:
            progress = (t - vanish_time_sec) / duration
            ksize = int(progress * 65) | 1
            if ksize >= 3:
                kernel = np.zeros((ksize, ksize))
                kernel[ksize // 2, :] = 1.0 / ksize
                blurred = cv2.filter2D(frame, -1, kernel)

                edges = cv2.Canny(frame, 80, 180)
                edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
                edge_mask = (edges > 0)[:, :, np.newaxis]

                trail = np.zeros_like(frame)
                trail[:, :] = trail_color

                glow = np.where(edge_mask, trail, blurred)
                alpha = max(0.0, 1.0 - progress * 1.4)
                out = glow.astype(np.float32) * alpha + frame.astype(np.float32) * (1.0 - alpha)
                return np.clip(out, 0, 255).astype(np.uint8)
        elif t >= end_time and t < end_time + 0.2:
            res_t = (t - end_time) / 0.2
            spark_mask = _create_horizontal_speed_lines(h, w, num_lines=15, seed=int(t * 100))
            spark_blur = cv2.GaussianBlur(spark_mask, (3, 3), 0) / 255.0 * (1.0 - res_t) * 0.4
            mask_3c = np.repeat(spark_blur[:, :, np.newaxis], 3, axis=2)
            trail = np.zeros_like(frame, dtype=np.float32)
            trail[:, :] = trail_color
            out = frame.astype(np.float32) * (1.0 - mask_3c) + trail * mask_3c
            return np.clip(out, 0, 255).astype(np.uint8)
        return frame

    return clip.transform(_vanish_transform)


def apply_anime_frame_skip_streak(
    clip: Any,
    trigger_time_sec: float,
    duration: float = 0.40,
    streak_color: Tuple[int, int, int] = (0, 245, 255),
    intensity: float = 0.85,
) -> Any:
    """Anime Ellipsis & Frame Skip Shorthand:
    Instead of rendering awkward continuous physical contact/carrying,
    the scene erupts into high-speed electric blur streaks, a 2-frame impact flash,
    and residual speed lines, leaving the contact area instantly cleared with lingering energy trails.
    """
    w, h = clip.size
    end_time = trigger_time_sec + duration
    impact_flash_time = trigger_time_sec + duration * 0.45

    def _skip_transform(get_frame, t: float):
        frame = get_frame(t)
        if trigger_time_sec <= t < end_time:
            progress = (t - trigger_time_sec) / duration
            # Phase 1: Sudden Sakuga Impact Flash in the middle of the streak
            if abs(t - impact_flash_time) < 0.08:
                # 2-frame monochrome negative impact burst
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                inv = 255 - gray
                flash_frame = np.stack([inv, inv, np.clip(inv.astype(np.int32) + 60, 0, 255).astype(np.uint8)], axis=2)
                return flash_frame

            # Phase 2: Blinding directional speed streak & electric trail
            ksize = int(25 + progress * 70) | 1
            kernel = np.zeros((ksize, ksize))
            kernel[ksize // 2, :] = 1.0 / ksize
            streaked = cv2.filter2D(frame, -1, kernel)

            # High-speed horizontal rush streaks
            rush_mask = _create_horizontal_speed_lines(h, w, num_lines=35, seed=int(t * 120))
            rush_blur = cv2.GaussianBlur(rush_mask, (5, 5), 0) / 255.0 * intensity

            # Energy color wash
            streak_layer = np.zeros_like(frame, dtype=np.float32)
            streak_layer[:, :] = streak_color

            # Compose streak trail
            rush_3c = np.repeat(rush_blur[:, :, np.newaxis], 3, axis=2)
            blended = streaked.astype(np.float32) * (1.0 - rush_3c * 0.7) + streak_layer * (rush_3c * 0.7)
            alpha = max(0.0, 1.0 - progress * 1.2)
            out = blended * alpha + frame.astype(np.float32) * (1.0 - alpha)
            return np.clip(out, 0, 255).astype(np.uint8)

        elif end_time <= t < end_time + 0.35:
            # Residual speed lines dissolving outward
            residual_prog = (t - end_time) / 0.35
            res_mask = _create_horizontal_speed_lines(h, w, num_lines=18, seed=int(t * 80))
            res_blur = cv2.GaussianBlur(res_mask, (3, 3), 0) / 255.0 * (1.0 - residual_prog) * 0.45 * intensity
            res_3c = np.repeat(res_blur[:, :, np.newaxis], 3, axis=2)
            streak_layer = np.zeros_like(frame, dtype=np.float32)
            streak_layer[:, :] = streak_color
            out = frame.astype(np.float32) * (1.0 - res_3c) + streak_layer * res_3c
            return np.clip(out, 0, 255).astype(np.uint8)

        return frame

    return clip.transform(_skip_transform)


def apply_bullet_time_slowmo(
    clip: Any,
    slow_start_sec: float,
    slow_dur_sec: float = 1.2,
    speed_factor: float = 0.3,
) -> Any:
    """Bullet-time time dilation: slows down high-stakes action beat before explosive snap resolution."""
    if speed_factor <= 0 or speed_factor >= 1.0:
        return clip

    def _bullet_time_mapping(t: float) -> float:
        if t < slow_start_sec:
            return t
        extra = t - slow_start_sec
        dilated_span = slow_dur_sec / speed_factor
        if extra < dilated_span:
            return slow_start_sec + extra * speed_factor
        orig_elapsed = slow_dur_sec
        post_extra = extra - dilated_span
        return slow_start_sec + orig_elapsed + post_extra

    new_duration = clip.duration + slow_dur_sec * (1.0 / speed_factor - 1.0)
    return clip.time_transform(_bullet_time_mapping).with_duration(new_duration)


def apply_anime_action_burst(
    clip: Any,
    trigger_time_sec: float = 0.2,
    with_impact: bool = True,
    with_speed_lines: bool = True,
    with_shake: bool = True,
    with_vanish: bool = False,
) -> Any:
    """Master preset for anime high-speed action beats, sonic blitzes, and superpower activations."""
    c = clip
    if with_shake:
        c = apply_screen_shake(c, trigger_time_sec=trigger_time_sec, duration=0.35, max_displacement=16)
    if with_impact:
        c = apply_impact_frames(c, trigger_time_sec=trigger_time_sec, num_frames=3, style="inverted")
    if with_speed_lines:
        c = apply_anime_speed_lines(
            c, start_sec=trigger_time_sec, end_sec=trigger_time_sec + 0.8, line_type="horizontal", intensity=0.55
        )
    if with_vanish:
        c = apply_anime_vanish(c, vanish_time_sec=trigger_time_sec, duration=0.3)
    return c


def apply_anime_ken_burns_drift(
    clip: Any,
    direction: str = "push_in",
    zoom_ratio: float = 1.05,
    pan_shift_px: int = 15,
) -> Any:
    """Apply silky-smooth cinematic 2D camera drift (Ken Burns) to eliminate static frozen frames.

    - 'push_in': subtle continuous camera zoom-in toward center (creates dramatic focus).
    - 'pull_out': subtle continuous camera pull-back (reveals scale/atmosphere).
    - 'pan_right': gentle horizontal tracking pan toward the right.
    - 'pan_left': gentle horizontal tracking pan toward the left.
    """
    w, h = clip.size
    dur = float(getattr(clip, "duration", 4.0) or 4.0)
    cx, cy = w / 2.0, h / 2.0

    def _drift_transform(get_frame, t: float):
        frame = get_frame(t)
        progress = max(0.0, min(1.0, t / max(dur, 0.1)))

        if direction == "push_in":
            scale = 1.0 + (zoom_ratio - 1.0) * progress
            dx, dy = 0.0, 0.0
        elif direction == "pull_out":
            scale = zoom_ratio - (zoom_ratio - 1.0) * progress
            dx, dy = 0.0, 0.0
        elif direction == "pan_right":
            scale = 1.02
            dx = pan_shift_px * (progress - 0.5) * 2.0
            dy = 0.0
        elif direction == "pan_left":
            scale = 1.02
            dx = -pan_shift_px * (progress - 0.5) * 2.0
            dy = 0.0
        else:
            scale = 1.0 + (zoom_ratio - 1.0) * progress
            dx, dy = 0.0, 0.0

        M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
        M[0, 2] += dx
        M[1, 2] += dy
        warped = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        return warped

    return clip.transform(_drift_transform)


def apply_anime_snap_zoom(
    clip: Any,
    trigger_time_sec: float = 0.2,
    zoom_ratio: float = 1.25,
    duration: float = 0.22,
) -> Any:
    """Instant Sakuga / Shonen Crash Zoom snapping onto character's shock or eye reaction."""
    w, h = clip.size
    cx, cy = w / 2.0, h * 0.45
    end_time = trigger_time_sec + duration

    def _snap_transform(get_frame, t: float):
        frame = get_frame(t)
        if t < trigger_time_sec:
            return frame
        elif trigger_time_sec <= t < end_time:
            # Fast cubic ease-out zoom snap
            rel = (t - trigger_time_sec) / duration
            ease = 1.0 - (1.0 - rel) ** 3
            scale = 1.0 + (zoom_ratio - 1.0) * ease
        else:
            scale = zoom_ratio

        M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
        return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    return clip.transform(_snap_transform)


def apply_anime_chromatic_aberration(
    clip: Any,
    trigger_time_sec: float = 0.0,
    duration: float = 0.35,
    max_shift: int = 8,
) -> Any:
    """RGB color channel separation split on sonic booms, energy surges, or psychological shocks."""
    end_time = trigger_time_sec + duration

    def _chroma_transform(get_frame, t: float):
        frame = get_frame(t)
        if trigger_time_sec <= t < end_time:
            rel = (t - trigger_time_sec) / duration
            # Exponential decay of the color displacement
            shift = int(max_shift * math.exp(-rel * 6.0))
            if shift > 0:
                b, g, r = cv2.split(frame)
                r_shifted = np.roll(r, shift, axis=1)
                b_shifted = np.roll(b, -shift, axis=1)
                return cv2.merge([b_shifted, g, r_shifted])
        return frame

    return clip.transform(_chroma_transform)


def apply_anime_film_grain_and_halation(
    clip: Any,
    grain_intensity: float = 0.08,
    halation_warmth: float = 0.12,
) -> Any:
    """Give diffusion videos organic 2D anime cel texture, eliminating plastic AI diffusion gloss."""
    w, h = clip.size

    def _grain_transform(get_frame, t: float):
        frame = get_frame(t).astype(np.float32)

        # Soft halation on bright highlights
        if halation_warmth > 0.02:
            bright_mask = np.clip((frame - 180.0) / 75.0, 0.0, 1.0)
            warm_tint = np.array([1.10, 1.03, 0.92], dtype=np.float32)
            frame = frame * (1.0 - bright_mask * halation_warmth) + (frame * warm_tint) * (bright_mask * halation_warmth)

        # Organic film grain
        if grain_intensity > 0.01:
            seed = int(t * 100) % 500
            rng = np.random.RandomState(seed)
            noise = rng.normal(0, grain_intensity * 255.0, (h, w, 1)).astype(np.float32)
            frame = frame + noise

        return np.clip(frame, 0, 255).astype(np.uint8)

    return clip.transform(_grain_transform)


def apply_anime_whip_pan(
    clip: Any,
    trigger_time_sec: float = 0.0,
    duration: float = 0.25,
    direction: str = "right",
) -> Any:
    """High-speed swish/whip pan blur connecting different angles or camera positions."""
    w, h = clip.size
    end_time = trigger_time_sec + duration

    def _whip_transform(get_frame, t: float):
        frame = get_frame(t)
        if trigger_time_sec <= t < end_time:
            rel = (t - trigger_time_sec) / duration
            # Bell curve blur intensity peaking at midpoint
            intensity = math.sin(rel * math.pi)
            ksize = int(intensity * 45) | 1
            if ksize >= 3:
                kernel = np.zeros((ksize, ksize), dtype=np.float32)
                kernel[ksize // 2, :] = 1.0 / ksize
                blurred = cv2.filter2D(frame, -1, kernel)
                shift = int(w * 0.15 * intensity) if direction == "right" else -int(w * 0.15 * intensity)
                M = np.float32([[1, 0, shift], [0, 1, 0]])
                shifted = cv2.warpAffine(blurred, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
                return shifted
        return frame

    return clip.transform(_whip_transform)

