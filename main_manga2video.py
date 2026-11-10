from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from agent_runtime.config import image_num_candidates
from agent_runtime.vimax_adapters import (
    _UnavailableGenerator,
    _build_chat_model,
    _build_image_generator,
    _build_video_generator,
)
from pipelines.manga2video_pipeline import Manga2VideoPipeline

DEFAULT_ANIME_STYLE = (
    "2D modern Japanese anime film style, crisp cel shading, expressive eyes, atmospheric lighting"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapt manga, manhwa, or webtoon chapters into canon-faithful anime episodes or full multi-episode series."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Input folders, .cbz/.zip archives, .pdf files, or LINE Webtoon URLs (episode viewer or series /list URL).",
    )
    parser.add_argument(
        "--working-dir",
        default=".working_dir/manga2video_cli",
        help="Working directory where comic_manifest.json, character portraits, and episode renders are written.",
    )
    parser.add_argument(
        "--chapter-range",
        default=None,
        help="Chapter subset to adapt, e.g. '1-5', '1,3,5', or 'all'.",
    )
    parser.add_argument(
        "--chapters-per-episode",
        type=int,
        default=1,
        help="Number of chapters per anime episode (>=1 enables full-series mode under episodes/ep_XXX/; 0 runs single-episode mode).",
    )
    parser.add_argument(
        "--episode-range",
        default=None,
        help="Optional episode subset in series mode, e.g. '1-3' or '2'.",
    )
    parser.add_argument(
        "--scene-range",
        default=None,
        help="Optional scene subset in episode mode, e.g. '0' or '0-2'.",
    )
    parser.add_argument(
        "--style",
        default=DEFAULT_ANIME_STYLE,
        help="Target anime visual style prompt.",
    )
    parser.add_argument(
        "--user-requirement",
        default="",
        help="Extra adaptation instructions (pacing, tone, duration hints, etc.).",
    )
    parser.add_argument(
        "--series-title",
        default="",
        help="Custom series or episode title (defaults to detected comic title).",
    )
    parser.add_argument(
        "--character-refs-json",
        default=None,
        help="Optional JSON file mapping character names to reference image paths.",
    )
    parser.add_argument(
        "--upload-images",
        nargs="*",
        default=None,
        help="Optional character reference image files (filename stem matches character name).",
    )
    parser.add_argument(
        "--stop-after",
        choices=("portraits", "frames", "full"),
        default="full",
        help="Stop rendering after turnaround portraits, keyframes, or full video concatenation.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Ingest sources and plan character roster/storyboards/shots only (skip image/video generation).",
    )
    parser.add_argument(
        "--force-replan",
        action="store_true",
        help="Ignore cached episode/series planning JSONs and re-plan from source pages.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip previously rendered episodes in series mode.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="Number of comic pages sent per VLM reading batch (default: 3).",
    )
    parser.add_argument(
        "--allow-private-urls",
        action="store_true",
        help="Allow fetching comic URLs on private/loopback addresses (for local mirrors/tests).",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Emit structured JSONL progress events to stdout.",
    )
    return parser.parse_args(argv)


def _load_character_refs(path_str: str | None) -> dict[str, Any] | None:
    if not path_str:
        return None
    p = Path(path_str).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Character refs JSON not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"--character-refs-json must contain a JSON object: {p}")
    return data


def _build_progress_printer(jsonl: bool):
    def _emit(stage: str, message: str, metadata: dict[str, Any] | None = None) -> None:
        if jsonl:
            print(
                json.dumps(
                    {"stage": stage, "message": message, "metadata": metadata or {}},
                    ensure_ascii=False,
                ),
                flush=True,
            )
        else:
            print(f"[{stage}] {message}", flush=True)

    return _emit


async def run_manga2video_cli(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    if args.chapters_per_episode < 0:
        raise ValueError("--chapters-per-episode must be >= 0")

    work_dir = Path(args.working_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    character_refs = _load_character_refs(args.character_refs_json)
    progress = _build_progress_printer(args.jsonl)

    chat_model = _build_chat_model()
    unavailable = _UnavailableGenerator()
    if args.plan_only:
        image_generator: Any = unavailable
        video_generator: Any = unavailable
    else:
        image_generator = _build_image_generator()
        if args.stop_after == "full":
            video_generator = _build_video_generator()
        elif args.stop_after == "frames":
            try:
                video_generator = _build_video_generator()
            except RuntimeError:
                video_generator = unavailable
        else:
            video_generator = unavailable

    num_candidates = image_num_candidates()
    pipeline = Manga2VideoPipeline(
        chat_model=chat_model,
        image_generator=image_generator,
        video_generator=video_generator,
        working_dir=work_dir,
        num_image_candidates=num_candidates,
    )

    if args.chapters_per_episode >= 1:
        series_plan = await pipeline.plan_series(
            sources=args.sources,
            chapter_range=args.chapter_range,
            chapters_per_episode=args.chapters_per_episode,
            episode_range=args.episode_range,
            character_refs=character_refs,
            upload_image_paths=args.upload_images,
            style=args.style,
            user_requirement=args.user_requirement,
            series_title=args.series_title,
            batch_size=args.batch_size,
            force_replan=args.force_replan,
            allow_private_urls=args.allow_private_urls,
            progress=progress,
            quiet=args.jsonl,
        )
        summary: dict[str, Any] = {
            "mode": "series",
            "working_dir": str(work_dir),
            "series_title": series_plan.series_title,
            "total_chapters": series_plan.total_chapters,
            "chapters_per_episode": series_plan.chapters_per_episode,
            "episode_count": len(series_plan.episodes),
            "plan_only": bool(args.plan_only),
        }
        if not args.plan_only:
            render_out = await pipeline.render_series(
                style=series_plan.style or args.style,
                user_requirement=args.user_requirement,
                stop_after=args.stop_after,
                episode_range=args.episode_range,
                scene_range=args.scene_range,
                resume=not args.no_resume,
                progress=progress,
                quiet=args.jsonl,
            )
            if args.stop_after == "full":
                summary["completed_episodes"] = render_out.completed_episodes
                summary["final_video_paths"] = [
                    ep.final_video_path for ep in render_out.episodes if ep.final_video_path
                ]
            else:
                summary["stop_after"] = args.stop_after
                summary["stopped_output_count"] = len(render_out) if isinstance(render_out, (list, dict)) else 0
        return summary

    ep_plan = await pipeline.plan_episode(
        sources=args.sources,
        chapter_range=args.chapter_range,
        character_refs=character_refs,
        upload_image_paths=args.upload_images,
        style=args.style,
        user_requirement=args.user_requirement,
        episode_title=args.series_title,
        batch_size=args.batch_size,
        force_replan=args.force_replan,
        allow_private_urls=args.allow_private_urls,
        progress=progress,
        quiet=args.jsonl,
    )
    summary = {
        "mode": "single_episode",
        "working_dir": str(work_dir),
        "episode_title": ep_plan.episode_title,
        "chapters": ep_plan.chapters,
        "scene_count": len(ep_plan.scenes),
        "total_beats": ep_plan.total_beats,
        "plan_only": bool(args.plan_only),
    }
    if not args.plan_only:
        render_out = await pipeline.render_episode(
            style=ep_plan.style or args.style,
            user_requirement=args.user_requirement,
            stop_after=args.stop_after,
            scene_range=args.scene_range,
            progress=progress,
            quiet=args.jsonl,
        )
        if args.stop_after == "full":
            summary["final_video_path"] = str(render_out)
        else:
            summary["stop_after"] = args.stop_after
            summary["stopped_output_count"] = len(render_out) if isinstance(render_out, (list, dict)) else 0
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    try:
        summary = asyncio.run(run_manga2video_cli(argv))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "ok", **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
