import logging
import requests
from moviepy import VideoFileClip, concatenate_videoclips
from utils.retry import download_retry


@download_retry
def download_video(url, save_path):
    try:
        logging.info(f"Downloading video from {url} to {save_path}")

        response = requests.get(url, stream=True, timeout=(10, 300))
        response.raise_for_status()  # 检查请求是否成功
    
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


def concatenate_video_files(video_paths, output_path, codec="libx264", preset="medium"):
    """Concatenate video files, releasing every ffmpeg reader even on failure.

    Each VideoFileClip keeps an ffmpeg subprocess and file handle open until
    closed; leaking them exhausts file descriptors on long multi-scene runs.
    """
    clips = []
    extra_clips = []
    final = None
    try:
        for path in video_paths:
            clips.append(VideoFileClip(path))
        normalized_clips, extra_clips = normalize_clips_for_concat(clips)
        final = concatenate_videoclips(normalized_clips, method="compose")
        final.write_videofile(output_path, codec=codec, preset=preset)
    finally:
        if final is not None:
            final.close()
        for clip in extra_clips:
            clip.close()
        for clip in clips:
            clip.close()
    return output_path
