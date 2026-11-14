from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image, ImageOps
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from agents.shot_linter import detect_context_jump
from interfaces import (
    CharacterInScene,
    MangaBeat,
    MangaCharacterObservation,
    MangaChunkExtraction,
    MangaDialogueLine,
    MangaEpisodePlan,
    MangaEpisodeScene,
)
from tools.manga_source import ChapterPages, ComicManifest, MangaPage
from utils.image import image_path_to_b64
from utils.retry import after_func
from utils.robust_json_parser import TrailingCommaTolerantPydanticOutputParser as PydanticOutputParser
from utils.text import safe_path_component


system_prompt_template_read_comic = """
[Role]
You are an expert manga / manhwa / webtoon director and visual continuity archivist.

[Task]
Inspect the attached sequential comic page images carefully and extract a strictly canon, pixel-faithful structured JSON breakdown (`MangaChunkExtraction`) covering EVERY panel, character appearance, spoken line, internal thought/monologue, and sound effect visible in these pages.

[Strict Grounding Rules]
1. OBEY ONLY WHAT IS LITERALLY DRAWN OR WRITTEN ON THESE PAGES. Never guess plot points or character hair/clothing colors from outside memory!
2. EXACT VISUAL FEATURES: Record exact hair colors, eye colors, fur colors & tabby stripe patterns, clothes/school uniforms/hoodies/coats, injuries, rain/lighting, and posture as literally visible. If an injured man has black hair and dark clothing on page 0, record black hair and dark clothing — NOT silver/white hair unless white hair is actually drawn.
3. DISTINCT FORMS / SPECIES MUST BE SEPARATED: If one character has physically different forms (e.g. human form vs four-legged fat orange cat, or armored vs beast form), ALWAYS extract each physical form as a separate `MangaCharacterObservation` entry such as `"Alex (Human)"` vs `"Alex (Orange Tabby Cat)"` with `"human"` vs `"cat (four-legged quadruped)"` in `species_or_form` and explicit posture (`"walking on all four paws, NEVER bipedal"` for animals).
4. COMPLETE TEXT EXTRACTION:
   - Capture EVERY speech bubble (`line_type: "spoken"`), thought/narration box/monologue (`line_type: "thought"` or `"narration"`), and visible SFX text (`sfx`).
   - Use `previous_tail_summary` ONLY to recognize a half-bubble continuing from the end of the previous batch — do NOT re-emit already completed beats from earlier pages!
   - Stitch any bubble that spans across two adjacent `[Page N]` images within THIS batch into its coherent beat without dropping half the sentence.
5. JAPANESE ANIME AUDIO + VIETSUB FAITHFULNESS:
   - `line`: Adapt the comic bubble concisely into short natural Japanese anime dialogue/monologue (romaji or Japanese, 1 brief sentence/clause <= 12 words) suitable for a native Japanese seiyuu.
   - `vietsub`: Translate the EXACT meaning and tone of that original comic bubble into natural, faithful Vietnamese (preserving humor, slang, urgency, or inner thoughts).
   - `source_text`: Copy the ORIGINAL text inside the comic bubble verbatim (before any rewrite/translation), for fidelity checks. Leave empty only if unreadable.
6. NORMALIZED BOUNDING BOXES (`0..1000`):
   - Set `page_idx` to the exact integer page number shown in `[Page N]` where the character or panel appears.
   - Set `crop_box_norm` to `[ymin, xmin, ymax, xmax]` in integer coordinates `0..1000` tightly framing that panel (for beats) or that character's clean face/upper-body/full-body (for character observations, avoiding overlapping speech bubbles when possible).
7. SCENE BREAKS: Set `is_scene_break: true` ONLY when location, time of day, or continuity cleanly switches to a new scene at that beat.
8. CINEMATIC CAMERA COMPOSITION & PERSPECTIVE DYNAMICS (NO WEBTOON PAGE CHROME):
   - Capture the author's exact cinematic framing drawn in the panels: camera angles (extreme tilted Dutch angle, ground-level worm's eye looking diagonally up at walls/buildings, high overhead bird's eye, extreme close-up on eyes), lighting direction, and spatial depth.
   - Strip comic-page layout chrome from `setting`, `action_desc`, and `visual_details`: put spoken/thought lines ONLY in `dialogues` and sounds ONLY in `sfx` — NEVER instruct image/video generators to draw speech/thought bubbles, comic SFX text (`FLASH!`, `YAWN`, `... A CAT?`), webtoon gutter borders (e.g. dark maroon margins, white gradient strips, empty fade rectangles), or truncated corner crop fragments; replace plain white close-up voids inside a dark scene with the continuous dim scene atmosphere.
   - In high-altitude transitions, extract only what is literally drawn on the page without inventing crash explosions or extra dialogue not present in the panels.

[Output Format]
Return ONLY valid JSON matching:
{format_instructions}
"""

human_prompt_template_read_comic = """
Chapter: {chapter} ({title})
Previous Batch Continuity Tail: {previous_tail_summary}
Valid Page Indices in THIS Batch: {valid_page_indices}
Target Beats Style Context: {style_hint}

Analyze ONLY the images attached below in top-to-bottom reading order:
"""


class _RawChunkResponse(BaseModel):
    characters_observed: List[MangaCharacterObservation] = Field(default_factory=list)
    beats: List[MangaBeat] = Field(default_factory=list)
    chunk_tail_summary: str = Field(default="")


_CAT_FORM_TOKENS = (
    "cat",
    "tabby",
    "kitten",
    "feline",
    "quadruped",
    "four-legged",
    "all four paws",
    "meo",
    "mèo",
    "dog",
    "puppy",
    "wolf",
    "fox",
    "animal",
)
_HUMAN_FORM_TOKENS = (
    "human",
    "man",
    "woman",
    "boy",
    "girl",
    "teen",
    "student",
    "awakener",
    "person",
    "người",
    "nam",
    "nữ",
)
_GENERIC_UPLOAD_STEM_SKIP = re.compile(
    r"^(?:(?:ch(?:ap(?:ter)?)?[_\-\s]*\d+[_\-\s]+)?(?:p|page|img|scan|panel|raw)[_\-\s]*\d+|ch(?:ap(?:ter)?)?[_\-\s]*\d+)$",
    re.IGNORECASE,
)


def _clamp_box_norm(raw_box: Sequence[Any] | None) -> List[int]:
    if not raw_box or len(raw_box) != 4:
        return [0, 0, 1000, 1000]
    try:
        vals = [int(round(float(v))) for v in raw_box]
    except (TypeError, ValueError):
        return [0, 0, 1000, 1000]
    ymin, xmin, ymax, xmax = vals
    # Handle normalized 0..1 floats accidentally emitted by VLMs
    if all(0 <= float(v) <= 1.0 for v in raw_box) and max(float(v) for v in raw_box) <= 1.0 and max(vals) <= 1:
        ymin, xmin, ymax, xmax = [int(round(float(v) * 1000)) for v in raw_box]
    ymin = max(0, min(1000, ymin))
    xmin = max(0, min(1000, xmin))
    ymax = max(0, min(1000, ymax))
    xmax = max(0, min(1000, xmax))
    if ymax <= ymin or xmax <= xmin:
        return [0, 0, 1000, 1000]
    return [ymin, xmin, ymax, xmax]


def _form_category(species_or_form: str, name: str = "", appearance: str = "") -> str:
    combined = f"{species_or_form} {name}".lower()
    if any(tok in combined for tok in ("dog", "puppy", "wolf", "fox", "chó")):
        return "canine"
    if any(tok in combined for tok in ("cat", "tabby", "kitten", "feline", "mèo", "stray")):
        if any(h in combined for h in ("(human", "human form", "dạng người")):
            return "human"
        return "feline"
    if any(tok in combined for tok in ("quadruped", "four-legged", "animal", "beast", "bird")):
        return "animal"
    if any(tok in combined for tok in _HUMAN_FORM_TOKENS):
        return "human"
    app_l = appearance.lower()
    if any(tok in app_l for tok in ("four paws", "quadruped", "feline", "whiskers", "tabby fur")):
        return "feline"
    return "human"


def _strip_form_parenthetical(name: str) -> str:
    cleaned = re.sub(r"\s*\([^)]*\)\s*", " ", str(name or "")).strip()
    return " ".join(cleaned.split())


def _canonical_form_label(name: str, form_cat: str, species_or_form: str) -> str:
    base = _strip_form_parenthetical(name) or str(name or "").strip() or "Character"
    if "(" in name and ")" in name:
        return " ".join(name.split())
    if form_cat == "human":
        return f"{base} (Human)"
    detail = species_or_form.strip()
    if detail and detail.lower() not in {"animal", "cat", "feline"} and len(detail) <= 32:
        return f"{base} ({detail})"
    if form_cat == "feline":
        return f"{base} (Cat Form)"
    return f"{base} ({form_cat.title()})"


def _name_tokens(text: str) -> List[str]:
    return [tok for tok in re.findall(r"[a-z0-9]+", str(text or "").lower()) if len(tok) >= 2]


def _crop_from_normalized_box(
    src_image_path: str | Path,
    crop_box_norm: List[int],
    dest_path: str | Path,
    *,
    pad_norm: int = 30,
    min_box_norm: int = 120,
) -> str:
    out_path = Path(dest_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ymin, xmin, ymax, xmax = _clamp_box_norm(crop_box_norm)
    use_full = (ymax - ymin) < min_box_norm or (xmax - xmin) < min_box_norm
    if not use_full:
        ymin = max(0, ymin - pad_norm)
        xmin = max(0, xmin - pad_norm)
        ymax = min(1000, ymax + pad_norm)
        xmax = min(1000, xmax + pad_norm)
    else:
        ymin, xmin, ymax, xmax = 0, 0, 1000, 1000

    with Image.open(src_image_path) as raw_im:
        rgb = ImageOps.exif_transpose(raw_im).convert("RGB")
        try:
            w, h = rgb.size
            left = max(0, min(w - 1, int(round(w * xmin / 1000.0))))
            top = max(0, min(h - 1, int(round(h * ymin / 1000.0))))
            right = max(left + 1, min(w, int(round(w * xmax / 1000.0))))
            bottom = max(top + 1, min(h, int(round(h * ymax / 1000.0))))
            cropped = rgb.crop((left, top, right, bottom))
            try:
                cropped.save(out_path, format="JPEG", quality=92, optimize=True)
            finally:
                cropped.close()
        finally:
            rgb.close()
    return str(out_path)


def split_dialogue_to_word_budget(
    line: str,
    vietsub: str = "",
    *,
    max_words: int = 14,
) -> List[Tuple[str, str]]:
    """Split a long spoken line cleanly into <= max_words clauses, aligning proportional Vietsub splits."""
    clean_line = " ".join((line or "").split()).strip()
    clean_viet = " ".join((vietsub or "").split()).strip()
    words = clean_line.split()
    if len(words) <= max_words:
        return [(clean_line, clean_viet)]

    # Split original line into clauses under budget
    chunks: List[str] = []
    # Prefer punctuation delimiters first
    sentences = [s.strip() for s in re.split(r"(?<=[.!?…])\s+|(?<=,)\s+(?=.{8,})", clean_line) if s.strip()]
    buf: List[str] = []
    for seg in sentences or [clean_line]:
        seg_words = seg.split()
        if len(seg_words) > max_words:
            if buf:
                chunks.append(" ".join(buf))
                buf = []
            for idx in range(0, len(seg_words), max_words):
                chunks.append(" ".join(seg_words[idx : idx + max_words]))
        elif len(buf) + len(seg_words) <= max_words:
            buf.extend(seg_words)
        else:
            if buf:
                chunks.append(" ".join(buf))
            buf = list(seg_words)
    if buf:
        chunks.append(" ".join(buf))

    if len(chunks) <= 1 or not clean_viet:
        return [(c, clean_viet if i == 0 else "") for i, c in enumerate(chunks)]

    # Split Vietnamese translation across matching chunk count if multiple clauses exist
    v_sentences = [s.strip() for s in re.split(r"(?<=[.!?…])\s+|(?<=,)\s+", clean_viet) if s.strip()]
    v_chunks: List[str] = []
    if len(v_sentences) >= len(chunks):
        per = max(1, len(v_sentences) // len(chunks))
        cursor = 0
        for i in range(len(chunks)):
            end = len(v_sentences) if i == len(chunks) - 1 else min(len(v_sentences), cursor + per)
            v_chunks.append(" ".join(v_sentences[cursor:end]).strip())
            cursor = end
    else:
        v_words = clean_viet.split()
        total_orig = max(1, sum(len(c.split()) for c in chunks))
        cursor = 0
        for i, c in enumerate(chunks):
            if i == len(chunks) - 1:
                v_chunks.append(" ".join(v_words[cursor:]).strip())
            else:
                share = max(1, int(round(len(v_words) * (len(c.split()) / total_orig))))
                end = min(len(v_words), cursor + share)
                v_chunks.append(" ".join(v_words[cursor:end]).strip())
                cursor = end

    return [(chunks[i], v_chunks[i] if i < len(v_chunks) else "") for i in range(len(chunks))]


def _match_user_ref_to_character(
    char_name: str,
    form_cat: str,
    explicit_refs: Dict[str, Any],
    upload_paths: List[Path],
) -> List[str]:
    matched: List[str] = []
    seen: set[str] = set()
    char_tokens = set(_name_tokens(char_name))
    base_tokens = set(_name_tokens(_strip_form_parenthetical(char_name)))

    def _add(path_str: str) -> None:
        resolved = str(Path(path_str))
        if resolved not in seen and Path(resolved).is_file():
            seen.add(resolved)
            matched.append(resolved)

    # 1. Explicit character_refs dictionary
    char_l = char_name.strip().lower()
    base_l = _strip_form_parenthetical(char_name).lower()
    for key, val in (explicit_refs or {}).items():
        key_l = str(key or "").strip().lower()
        key_tokens = set(_name_tokens(key_l))
        paths = val if isinstance(val, (list, tuple)) else [val]
        if key_l == char_l or (key_tokens and key_tokens == char_tokens):
            for item in paths:
                _add(str(item))
        elif base_l and key_l == base_l:
            # Ensure human/animal form mismatch doesn't cross-contaminate when multiple forms share a name
            key_form = _form_category("", key_l, "")
            if key_form == form_cat or form_cat == "human":
                for item in paths:
                    _add(str(item))
        elif key_tokens and base_tokens and base_tokens.issubset(key_tokens):
            key_has_cat = any(tok in key_tokens for tok in ("cat", "meo", "tabby", "feline"))
            key_has_human = any(tok in key_tokens for tok in ("human", "nguoi", "man"))
            if (form_cat == "feline" and key_has_cat) or (form_cat == "human" and key_has_human) or (not key_has_cat and not key_has_human):
                for item in paths:
                    _add(str(item))

    # 2. Uploaded image files discovered inside <session>/uploads/
    for img_path in upload_paths:
        stem = img_path.stem
        if _GENERIC_UPLOAD_STEM_SKIP.match(stem):
            continue
        stem_tokens = set(_name_tokens(stem))
        if not stem_tokens or not (stem_tokens & base_tokens):
            continue
        stem_has_cat = any(tok in stem_tokens for tok in ("cat", "meo", "tabby", "feline", "fat"))
        stem_has_human = any(tok in stem_tokens for tok in ("human", "nguoi", "man", "person"))
        if form_cat in {"feline", "canine", "animal"}:
            if stem_has_human and not stem_has_cat:
                continue
            if not stem_has_cat and len(char_tokens) > len(base_tokens):
                # Require explicit animal tag if the character identifier was split into Human/Cat forms
                continue
        elif form_cat == "human" and stem_has_cat and not stem_has_human:
            continue
        _add(str(img_path))

    return matched


class MangaReader:
    def __init__(self, chat_model: Any) -> None:
        self.chat_model = chat_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=6),
        after=after_func,
        reraise=True,
    )
    async def _invoke_chunk_vision(
        self,
        *,
        chapter: int,
        title: str,
        batch: List[MangaPage],
        previous_tail_summary: str,
        style_hint: str,
    ) -> _RawChunkResponse:
        parser = PydanticOutputParser(pydantic_object=_RawChunkResponse)
        system_msg = SystemMessage(
            content=system_prompt_template_read_comic.format(
                format_instructions=parser.get_format_instructions()
            )
        )
        human_content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": human_prompt_template_read_comic.format(
                    chapter=chapter,
                    title=title or f"Chapter {chapter}",
                    previous_tail_summary=previous_tail_summary or "None (start of chapter).",
                    valid_page_indices=[p.page_idx for p in batch],
                    style_hint=style_hint or "2D Japanese anime",
                ),
            }
        ]
        for page in batch:
            human_content.append(
                {
                    "type": "text",
                    "text": f"[Page {page.page_idx}] (chapter {chapter}, size {page.width}x{page.height}):",
                }
            )
            human_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_path_to_b64(page.path, mime=True)},
                }
            )
        messages = [system_msg, HumanMessage(content=human_content)]
        chain = self.chat_model | parser
        return await chain.ainvoke(messages)

    async def read_chapter_pages(
        self,
        chapter_pages: ChapterPages,
        chunks_dir: str | Path,
        *,
        batch_size: int = 3,
        style_hint: str = "2D Japanese anime",
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
    ) -> List[MangaChunkExtraction]:
        batch_n = max(1, min(8, int(batch_size)))
        out_dir = Path(chunks_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pages = list(chapter_pages.pages)
        chunks: List[MangaChunkExtraction] = []
        prev_tail = ""
        beat_seq = 0

        for chunk_idx, start in enumerate(range(0, len(pages), batch_n)):
            batch = pages[start : start + batch_n]
            valid_indices = {p.page_idx for p in batch}
            default_page_idx = batch[0].page_idx
            cache_path = out_dir / f"ch{chapter_pages.chapter:02d}_chunk_{chunk_idx:03d}.json"

            extracted: Optional[MangaChunkExtraction] = None
            if cache_path.is_file():
                try:
                    cached = MangaChunkExtraction.model_validate(
                        json.loads(cache_path.read_text(encoding="utf-8"))
                    )
                    if cached.beats or cached.characters_observed or cached.chunk_tail_summary:
                        extracted = cached
                except (OSError, ValueError):
                    extracted = None

            if extracted is None:
                if progress is not None:
                    progress(
                        "manga_vision_chunk",
                        f"Reading chapter {chapter_pages.chapter} pages {batch[0].page_idx}-{batch[-1].page_idx}",
                        {"chapter": chapter_pages.chapter, "chunk_idx": chunk_idx, "pages": [p.page_idx for p in batch]},
                    )
                raw = await self._invoke_chunk_vision(
                    chapter=chapter_pages.chapter,
                    title=chapter_pages.title,
                    batch=batch,
                    previous_tail_summary=prev_tail,
                    style_hint=style_hint,
                )
                norm_chars: List[MangaCharacterObservation] = []
                for obs in raw.characters_observed:
                    p_idx = obs.page_idx if obs.page_idx in valid_indices else default_page_idx
                    norm_chars.append(
                        obs.model_copy(
                            update={
                                "page_idx": p_idx,
                                "crop_box_norm": _clamp_box_norm(obs.crop_box_norm),
                            }
                        )
                    )
                norm_beats: List[MangaBeat] = []
                for beat in raw.beats:
                    p_idx = beat.page_idx if beat.page_idx in valid_indices else default_page_idx
                    beat_id = f"ch{chapter_pages.chapter:02d}_b{beat_seq:03d}"
                    beat_seq += 1
                    norm_beats.append(
                        beat.model_copy(
                            update={
                                "beat_id": beat_id,
                                "chapter": chapter_pages.chapter,
                                "page_idx": p_idx,
                                "crop_box_norm": _clamp_box_norm(beat.crop_box_norm),
                            }
                        )
                    )
                extracted = MangaChunkExtraction(
                    chapter=chapter_pages.chapter,
                    start_page_idx=batch[0].page_idx,
                    end_page_idx=batch[-1].page_idx,
                    characters_observed=norm_chars,
                    beats=norm_beats,
                    chunk_tail_summary=raw.chunk_tail_summary.strip(),
                )
                cache_path.write_text(extracted.model_dump_json(indent=2), encoding="utf-8")
            else:
                for idx_in_chunk, beat in enumerate(extracted.beats):
                    beat_id = f"ch{chapter_pages.chapter:02d}_b{beat_seq:03d}"
                    beat_seq += 1
                    if beat.beat_id != beat_id:
                        extracted.beats[idx_in_chunk] = beat.model_copy(update={"beat_id": beat_id})

            prev_tail = extracted.chunk_tail_summary
            chunks.append(extracted)
        return chunks

    def build_character_roster_and_seeds(
        self,
        *,
        manifest: ComicManifest,
        chunks: Sequence[MangaChunkExtraction],
        working_dir: str | Path,
        character_refs: Optional[Dict[str, Any]] = None,
        upload_image_paths: Optional[Sequence[str | Path]] = None,
        series_characters: Optional[Sequence[CharacterInScene]] = None,
        series_seed_refs: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[List[CharacterInScene], Dict[str, List[str]], Dict[str, str]]:
        """Consolidate observations cleanly across distinct physical forms and collect character reference seeds.

        When ``series_characters`` / ``series_seed_refs`` are supplied (multi-episode series mode),
        recurring characters lock onto established canonical series names, appearance baselines, and seeds
        within the same physical form category so cross-episode portraits remain identical.

        Returns ``(characters, character_seed_refs, name_alias_map)``.
        """
        work_root = Path(working_dir)
        seeds_dir = work_root / "character_seeds"
        seeds_dir.mkdir(parents=True, exist_ok=True)
        page_lookup: Dict[Tuple[int, int], MangaPage] = {
            (ch.chapter, page.page_idx): page
            for ch in manifest.chapters
            for page in ch.pages
        }

        upload_paths = [Path(p) for p in (upload_image_paths or []) if Path(p).is_file()]
        grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}

        # Build lookup tables for cross-episode character continuity
        series_by_exact: Dict[Tuple[str, str], CharacterInScene] = {}
        series_by_form: Dict[str, List[Tuple[set[str], CharacterInScene]]] = {}
        series_forms_for_base: Dict[str, set[str]] = {}
        for sc in series_characters or ():
            s_name = " ".join((sc.identifier_in_scene or "").split()).strip()
            if not s_name or not getattr(sc, "is_visible", True):
                continue
            s_form = _form_category("", s_name, sc.static_features or "")
            s_base = _strip_form_parenthetical(s_name).lower() or s_name.lower()
            series_by_exact.setdefault((s_base, s_form), sc)
            series_forms_for_base.setdefault(s_base, set()).add(s_form)
            s_tokens = set(_name_tokens(s_base))
            if s_tokens:
                series_by_form.setdefault(s_form, []).append((s_tokens, sc))

        def _match_series_character(base_key: str, form_cat: str) -> Optional[CharacterInScene]:
            if (base_key, form_cat) in series_by_exact:
                return series_by_exact[(base_key, form_cat)]
            obs_tokens = set(_name_tokens(base_key))
            if not obs_tokens:
                return None
            candidates = [
                sc
                for sc_tokens, sc in series_by_form.get(form_cat, [])
                if obs_tokens <= sc_tokens or sc_tokens <= obs_tokens
            ]
            if len(candidates) == 1:
                return candidates[0]
            return None

        # Group observations by (normalized_base_name, physical_form_category)
        for chunk in chunks:
            for obs in chunk.characters_observed:
                raw_name = " ".join((obs.name or "").split()).strip()
                if not raw_name:
                    continue
                form_cat = _form_category(obs.species_or_form, raw_name, obs.appearance_summary)
                base_key = _strip_form_parenthetical(raw_name).lower() or raw_name.lower()
                group_key = (base_key, form_cat)
                entry = grouped.setdefault(
                    group_key,
                    {
                        "raw_names": set(),
                        "form_cat": form_cat,
                        "species_samples": [],
                        "appearance": [],
                        "outfit": [],
                        "observations": [],
                    },
                )
                entry["raw_names"].add(raw_name)
                if obs.species_or_form:
                    entry["species_samples"].append(obs.species_or_form.strip())
                if obs.appearance_summary:
                    entry["appearance"].append(obs.appearance_summary.strip())
                if obs.outfit_summary:
                    entry["outfit"].append(obs.outfit_summary.strip())
                entry["observations"].append((chunk.chapter, obs))

        # Check which base names span multiple distinct forms (e.g. a hero in human vs animal form)
        forms_per_base: Dict[str, set[str]] = {}
        for (base_key, form_cat) in grouped:
            forms_per_base.setdefault(base_key, set()).add(form_cat)

        characters: List[CharacterInScene] = []
        character_seed_refs: Dict[str, List[str]] = {}
        alias_map: Dict[str, str] = {}

        idx_counter = 0
        for (base_key, form_cat), info in grouped.items():
            raw_names = sorted(info["raw_names"], key=lambda n: (-len(n), n))
            multi_form = (
                len(forms_per_base.get(base_key, set())) > 1
                or len(series_forms_for_base.get(base_key, set()) | forms_per_base.get(base_key, set())) > 1
            )
            species_text = info["species_samples"][0] if info["species_samples"] else form_cat
            matched_series = _match_series_character(base_key, form_cat)
            if matched_series is not None:
                canonical_name = matched_series.identifier_in_scene
                alias_map[f"{canonical_name.lower()}::{form_cat}"] = canonical_name
                if not multi_form:
                    alias_map[canonical_name.lower()] = canonical_name
            elif multi_form:
                canonical_name = _canonical_form_label(raw_names[0], form_cat, species_text)
            else:
                canonical_name = raw_names[0]

            # Record aliases scoped when multi_form vs unscoped when single-form
            for alias in raw_names:
                alias_map[f"{alias.lower()}::{form_cat}"] = canonical_name
                if not multi_form:
                    alias_map[alias.lower()] = canonical_name
            if not multi_form:
                alias_map[base_key] = canonical_name
            alias_map[f"{base_key}::{form_cat}"] = canonical_name

            appearance_merged = "; ".join(dict.fromkeys(a for a in info["appearance"] if a))[:500]
            outfit_merged = "; ".join(dict.fromkeys(o for o in info["outfit"] if o))[:350]
            if matched_series is not None and matched_series.static_features:
                appearance_merged = matched_series.static_features
                if not outfit_merged and matched_series.dynamic_features:
                    outfit_merged = matched_series.dynamic_features
            if form_cat in {"feline", "canine", "animal"}:
                if "quadruped" not in appearance_merged.lower() and "four paws" not in appearance_merged.lower():
                    appearance_merged = f"{appearance_merged}; natural four-legged quadruped walking on all four paws, NEVER bipedal".strip("; ")
                if not outfit_merged or any(tok in outfit_merged.lower() for tok in ("coat", "shirt", "jacket", "pants")):
                    outfit_merged = "Unclothed natural fur on all four paws"

            is_narrator = base_key in {"narrator", "người dẫn chuyện", "caption"}
            char_obj = CharacterInScene(
                idx=idx_counter,
                identifier_in_scene=canonical_name,
                is_visible=not is_narrator,
                static_features=appearance_merged or species_text or canonical_name,
                dynamic_features=outfit_merged or None,
            )
            characters.append(char_obj)
            idx_counter += 1

            if char_obj.is_visible:
                user_seeds = _match_user_ref_to_character(
                    canonical_name,
                    form_cat,
                    character_refs or {},
                    upload_paths,
                )
                prior_series_seeds = [
                    str(p)
                    for p in ((series_seed_refs or {}).get(canonical_name) or [])
                    if str(p).strip() and Path(str(p)).is_file()
                ]
                if user_seeds:
                    character_seed_refs[canonical_name] = user_seeds[:4]
                elif prior_series_seeds:
                    character_seed_refs[canonical_name] = prior_series_seeds[:4]
                else:
                    cropped_seeds: List[str] = []
                    ranked_obs = sorted(
                        info["observations"],
                        key=lambda pair: float(getattr(pair[1], "confidence", 1.0)),
                        reverse=True,
                    )
                    for seed_i, (chap, obs) in enumerate(ranked_obs[:2]):
                        page = page_lookup.get((chap, obs.page_idx))
                        if page is None:
                            continue
                        seed_file = seeds_dir / f"{char_obj.idx}_{safe_path_component(canonical_name)}_seed_{seed_i}.jpg"
                        _crop_from_normalized_box(
                            page.path,
                            obs.crop_box_norm,
                            seed_file,
                            pad_norm=60,
                            min_box_norm=100,
                        )
                        cropped_seeds.append(str(seed_file))
                    if cropped_seeds:
                        character_seed_refs[canonical_name] = cropped_seeds

        # Ensure spoken-only speakers from dialogues also map cleanly (e.g. Narrator)
        existing_lower = {c.identifier_in_scene.lower() for c in characters}
        for chunk in chunks:
            for beat in chunk.beats:
                for dlg in beat.dialogues:
                    spk = " ".join((dlg.speaker or "").split()).strip()
                    if not spk:
                        continue
                    if spk.lower() in existing_lower or spk.lower() in alias_map:
                        continue
                    is_narr = spk.lower() in {"narrator", "người dẫn chuyện"}
                    if any(f"{spk.lower()}::{fc}" in alias_map for fc in ("human", "feline", "canine", "animal")):
                        continue
                    characters.append(
                        CharacterInScene(
                            idx=idx_counter,
                            identifier_in_scene=spk,
                            is_visible=not is_narr,
                            static_features="Narrator voice" if is_narr else spk,
                            dynamic_features=None,
                        )
                    )
                    existing_lower.add(spk.lower())
                    alias_map[spk.lower()] = spk
                    idx_counter += 1

        (work_root / "characters.json").write_text(
            json.dumps([c.model_dump() for c in characters], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_root / "character_seed_refs.json").write_text(
            json.dumps(character_seed_refs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return characters, character_seed_refs, alias_map

    def build_episode_scenes(
        self,
        *,
        manifest: ComicManifest,
        chunks: Sequence[MangaChunkExtraction],
        characters: List[CharacterInScene],
        character_seed_refs: Dict[str, List[str]],
        alias_map: Dict[str, str],
        working_dir: str | Path,
        style: str = "2D Japanese anime",
        episode_title: str = "",
        max_words_per_shot: int = 14,
        max_beats_per_scene: int = 14,
        min_beats_per_scene: int = 4,
    ) -> Tuple[MangaEpisodePlan, Dict[int, List[MangaBeat]]]:
        work_root = Path(working_dir)
        anchors_dir = work_root / "panel_anchors"
        anchors_dir.mkdir(parents=True, exist_ok=True)
        page_lookup: Dict[Tuple[int, int], MangaPage] = {
            (ch.chapter, page.page_idx): page
            for ch in manifest.chapters
            for page in ch.pages
        }

        _feline_cue_re = re.compile(
            r"\b(?:cat|cats|tabby|kitten|feline|meo|mèo|four\s+paws|quadruped|muzzle|whiskers|orange\s+(?:and\s+white\s+)?fur)\b"
        )
        _human_cue_re = re.compile(
            r"\b(?:human|man|woman|boy|girl|long\s+black\s+hair|pale\s+skin|trench\s+coat|hoodie|jacket|turtleneck|trousers|boots|hand\s+pressed)\b"
        )

        def _resolve_canon_name(raw_name: str, context_text: str) -> str:
            clean = " ".join((raw_name or "").split()).strip()
            if not clean:
                return "Narrator"
            base = _strip_form_parenthetical(clean).lower()
            # 1. If the raw name explicitly states a form parenthetical, obey it first.
            if "(" in clean and ")" in clean:
                explicit_form = _form_category(clean, clean, clean)
                if f"{clean.lower()}::{explicit_form}" in alias_map:
                    return alias_map[f"{clean.lower()}::{explicit_form}"]
                if f"{base}::{explicit_form}" in alias_map:
                    return alias_map[f"{base}::{explicit_form}"]
                if clean.lower() in alias_map:
                    return alias_map[clean.lower()]

            # 2. Match species cues using word boundaries outside quoted bubble text
            #    so words like "location"/"calculating" or thought text "'... A CAT?'"
            #    cannot flip a human beat into an animal form.
            unquoted_ctx = re.sub(r"(['\"]).*?\1", " ", str(context_text or ""))
            ctx_l = unquoted_ctx.lower()
            has_human_cue = bool(_human_cue_re.search(ctx_l))
            has_feline_cue = bool(_feline_cue_re.search(ctx_l))

            if has_human_cue and not has_feline_cue:
                if f"{clean.lower()}::human" in alias_map:
                    return alias_map[f"{clean.lower()}::human"]
                if f"{base}::human" in alias_map:
                    return alias_map[f"{base}::human"]
            if has_feline_cue and not has_human_cue:
                if f"{clean.lower()}::feline" in alias_map:
                    return alias_map[f"{clean.lower()}::feline"]
                if f"{base}::feline" in alias_map:
                    return alias_map[f"{base}::feline"]
            if has_human_cue and f"{base}::human" in alias_map:
                return alias_map[f"{base}::human"]
            if has_feline_cue and f"{base}::feline" in alias_map:
                return alias_map[f"{base}::feline"]
            return alias_map.get(clean.lower()) or alias_map.get(base) or clean

        # Flatten, normalize dialogues within word budget, and crop panel anchors once per unique beat
        normalized_beats: List[MangaBeat] = []
        sub_beat_idx = 0
        for chunk in chunks:
            for beat in chunk.beats:
                ctx = f"{beat.setting} {beat.action_desc} {beat.visual_details}"
                canon_chars = [_resolve_canon_name(c, ctx) for c in beat.characters_present]
                canon_chars = list(dict.fromkeys(canon_chars))

                page = page_lookup.get((beat.chapter, beat.page_idx))
                anchor_path = ""
                if page is not None:
                    raw_id = beat.beat_id or f"ch{beat.chapter:02d}_b{sub_beat_idx:03d}"
                    anchor_file = anchors_dir / f"{safe_path_component(raw_id)}.jpg"
                    anchor_path = _crop_from_normalized_box(
                        page.path,
                        beat.crop_box_norm,
                        anchor_file,
                        pad_norm=25,
                        min_box_norm=120,
                    )

                # Expand overloaded dialogue beats into atomic beats within <= max_words_per_shot
                expanded_dialogues: List[MangaDialogueLine] = []
                for dlg in beat.dialogues:
                    speaker = _resolve_canon_name(dlg.speaker, f"{ctx} {dlg.line}")
                    for chunk_line, chunk_viet in split_dialogue_to_word_budget(
                        dlg.line, dlg.vietsub, max_words=max_words_per_shot
                    ):
                        if chunk_line:
                            expanded_dialogues.append(
                                MangaDialogueLine(
                                    speaker=speaker,
                                    line=chunk_line,
                                    line_type=dlg.line_type,
                                    emotion=dlg.emotion,
                                    vietsub=chunk_viet,
                                )
                            )

                if len(expanded_dialogues) <= 1:
                    normalized_beats.append(
                        beat.model_copy(
                            update={
                                "beat_id": f"ch{beat.chapter:02d}_b{sub_beat_idx:03d}",
                                "characters_present": canon_chars,
                                "dialogues": expanded_dialogues,
                                "anchor_image_path": anchor_path,
                            }
                        )
                    )
                    sub_beat_idx += 1
                else:
                    for d_i, dlg_item in enumerate(expanded_dialogues):
                        action_text = beat.action_desc if d_i == 0 else f"Continuous reaction and delivery: {beat.action_desc}"
                        normalized_beats.append(
                            beat.model_copy(
                                update={
                                    "beat_id": f"ch{beat.chapter:02d}_b{sub_beat_idx:03d}",
                                    "is_scene_break": beat.is_scene_break if d_i == 0 else False,
                                    "action_desc": action_text,
                                    "characters_present": canon_chars,
                                    "dialogues": [dlg_item],
                                    "sfx": beat.sfx if d_i == 0 else [],
                                    "anchor_image_path": anchor_path,
                                }
                            )
                        )
                        sub_beat_idx += 1

        if not normalized_beats:
            raise ValueError("MangaReader extracted 0 story beats from the provided comic pages.")

        # Group contiguous beats into manageable scene units for Script2VideoPipeline
        scene_groups: List[List[MangaBeat]] = []
        current_group: List[MangaBeat] = []
        for beat in normalized_beats:
            if current_group:
                hit_break = beat.is_scene_break and len(current_group) >= min_beats_per_scene
                hit_chapter = beat.chapter != current_group[-1].chapter and len(current_group) >= 2
                hit_setting_change = bool(
                    beat.setting and current_group[-1].setting and
                    beat.setting.strip().lower() != current_group[-1].setting.strip().lower() and
                    len(current_group) >= min_beats_per_scene
                )
                hit_cap = len(current_group) >= max_beats_per_scene and (beat.is_scene_break or hit_setting_change or len(current_group) >= 16)
                if hit_break or hit_chapter or hit_setting_change or hit_cap:
                    scene_groups.append(current_group)
                    current_group = []
            current_group.append(beat)
        if current_group:
            # Avoid orphaned 1-beat tail if prior group can absorb cleanly
            if len(current_group) == 1 and scene_groups and len(scene_groups[-1]) < max_beats_per_scene:
                scene_groups[-1].extend(current_group)
            else:
                scene_groups.append(current_group)

        scenes: List[MangaEpisodeScene] = []
        beats_by_scene: Dict[int, List[MangaBeat]] = {}
        scene_scripts: List[str] = []

        for scene_idx, group in enumerate(scene_groups):
            beats_by_scene[scene_idx] = group
            scene_dir = work_root / f"scene_{scene_idx}"
            scene_dir.mkdir(parents=True, exist_ok=True)
            setting_summary = group[0].setting or "Scene location"
            chap_num = group[0].chapter
            scene_title = f"Chapter {chap_num} - Scene {scene_idx + 1}"

            lines: List[str] = [
                f"SCENE {scene_idx + 1}: {scene_title}",
                f"SETTING: {setting_summary}",
                "",
            ]
            if scene_idx > 0 and scene_groups:
                prev_grp = scene_groups[scene_idx - 1]
                prev_ctx = f"{prev_grp[-1].setting} {prev_grp[-1].action_desc}"
                curr_ctx = f"{group[0].setting} {group[0].action_desc}"
                if detect_context_jump(prev_ctx, curr_ctx):
                    lines.append(f"DIRECTOR NOTE (MA & ATMOSPHERIC PACING): Context jump detected from previous scene. Maintain natural breathing room with an atmospheric establishing shot grounded strictly in this scene's setting ({setting_summary}).")
                    lines.append("")
            anchors_map: Dict[str, str] = {}
            for b_i, beat in enumerate(group):
                if beat.anchor_image_path:
                    anchors_map[beat.beat_id] = beat.anchor_image_path
                char_tags = ", ".join(f"<{c}>" for c in beat.characters_present)
                lines.append(f"Shot {b_i + 1} [RefBeat: {beat.beat_id}] ({beat.mood or 'dramatic'}):")
                if char_tags:
                    lines.append(f"Visible Characters: {char_tags}")
                lines.append(f"Setting: {beat.setting}")
                lines.append(f"Visual: {beat.action_desc} {beat.visual_details}".strip())
                audio_parts: List[str] = []
                if beat.sfx:
                    audio_parts.append(f"[Sound Effect] {', '.join(beat.sfx)}")
                for dlg in beat.dialogues:
                    em = dlg.emotion or "Natural"
                    if dlg.line_type == "thought":
                        part = f'[Thought] {dlg.speaker} ({em}): "{dlg.line}"'
                    elif dlg.line_type == "narration" or dlg.speaker.lower() in {"narrator", "người dẫn chuyện"}:
                        part = f'[Speaker] Narrator ({em}): "{dlg.line}"'
                    else:
                        part = f'[Speaker] {dlg.speaker} ({em}): "{dlg.line}"'
                    if dlg.vietsub:
                        part += f"\n(Vietsub: {dlg.vietsub})"
                    audio_parts.append(part)
                if not audio_parts:
                    audio_parts.append("[Sound Effect] Ambient atmosphere.")
                lines.append("\n".join(audio_parts))
                lines.append("")

            script_text = "\n".join(lines).strip() + "\n"
            scene_scripts.append(script_text)
            (scene_dir / "script.txt").write_text(script_text, encoding="utf-8")
            (scene_dir / "beat_map.json").write_text(
                json.dumps([b.model_dump() for b in group], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            scenes.append(
                MangaEpisodeScene(
                    scene_idx=scene_idx,
                    chapter=chap_num,
                    title=scene_title,
                    setting=setting_summary,
                    beat_ids=[b.beat_id for b in group],
                    script_text=script_text,
                    panel_anchors_by_beat=anchors_map,
                )
            )

        plan = MangaEpisodePlan(
            episode_title=episode_title or (manifest.chapters[0].title if manifest.chapters else "Manga Episode"),
            style=style,
            chapters=[ch.chapter for ch in manifest.chapters],
            characters=characters,
            character_seed_refs=character_seed_refs,
            scenes=scenes,
            total_beats=len(normalized_beats),
        )
        (work_root / "episode_plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        (work_root / "script.json").write_text(
            json.dumps(scene_scripts, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return plan, beats_by_scene
