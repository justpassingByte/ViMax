from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import re
import socket
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import parse_qs, urljoin, urlparse

import aiohttp
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ARCHIVE_EXTENSIONS = {".cbz", ".zip"}
_SKIP_DIR_NAMES = {"_raw", "_stitched", "_normalized", "__pycache__", ".ipynb_checkpoints"}
_RETRYABLE_HTTP_STATUS = {408, 429, 500, 502, 503, 504}
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


class MangaPage(BaseModel):
    page_idx: int = Field(..., description="0-based index within the chapter.")
    chapter: int = Field(..., description="1-based chapter number.")
    path: str = Field(..., description="Normalized image path on disk.")
    width: int = Field(..., description="Normalized pixel width.")
    height: int = Field(..., description="Normalized pixel height.")
    source_name: str = Field(default="", description="Original filename before slice/normalization.")


class ChapterPages(BaseModel):
    chapter: int = Field(..., description="1-based chapter number.")
    title: str = Field(default="", description="Chapter display title.")
    source: str = Field(default="", description="Original input directory, archive, PDF, or URL.")
    pages: List[MangaPage] = Field(default_factory=list)


class ComicManifest(BaseModel):
    working_dir: str = Field(..., description="Root manga2video working directory.")
    chapters: List[ChapterPages] = Field(default_factory=list)
    total_pages: int = Field(default=0)


def natural_sort_key(text: str) -> List[object]:
    parts = re.split(r"(\d+)", str(text or ""))
    return [int(part) if part.isdigit() else part.lower() for part in parts if part != ""]


def parse_chapter_range(
    chapter_range: str | int | Sequence[int] | None,
) -> Optional[Set[int]]:
    if chapter_range is None:
        return None
    if isinstance(chapter_range, bool):
        raise ValueError("chapter_range cannot be a boolean")
    if isinstance(chapter_range, int):
        if chapter_range < 0:
            raise ValueError(f"chapter_range must be non-negative: {chapter_range}")
        return {chapter_range}
    if isinstance(chapter_range, (list, tuple, set)):
        result: Set[int] = set()
        for item in chapter_range:
            val = int(item)
            if val < 0:
                raise ValueError(f"chapter_range contains negative chapter: {val}")
            result.add(val)
        return result or None

    raw = str(chapter_range).strip()
    if not raw or raw.lower() in {"all", "*"}:
        return None

    selected: Set[int] = set()
    for token in re.split(r"[,;\s]+", raw):
        if not token:
            continue
        m_range = re.fullmatch(r"(?:ch(?:apter)?)?[_\-]?0*(\d+)\s*[-–~]\s*(?:ch(?:apter)?)?[_\-]?0*(\d+)", token, re.IGNORECASE)
        if m_range:
            start = int(m_range.group(1))
            end = int(m_range.group(2))
            if start > end:
                raise ValueError(f"Invalid descending chapter range: {token}")
            selected.update(range(start, end + 1))
            continue
        m_single = re.fullmatch(r"(?:ch(?:apter)?)?[_\-]?0*(\d+)", token, re.IGNORECASE)
        if m_single:
            selected.add(int(m_single.group(1)))
            continue
        raise ValueError(f"Malformed chapter_range token: {token!r}")
    return selected or None


def infer_chapter_number(label: str, default: int) -> int:
    text = str(label or "")
    for pat in (
        r"(?:ch(?:apter)?|chap|ep(?:isode)?|chuong)[_\-\s]*0*(\d+)",
        r"^0*(\d+)(?:[_\-\s]|$)",
    ):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    nums = re.findall(r"\d+", text)
    if nums:
        return int(nums[0])
    return default


def extract_cbz_archive(
    archive_path: Path | str,
    dest_dir: Path | str,
    *,
    max_entry_bytes: int = 100 * 1024 * 1024,
) -> List[Path]:
    src = Path(archive_path)
    target_root = Path(dest_dir).resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    extracted: List[Path] = []

    with zipfile.ZipFile(src, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            # Reject POSIX symlink entries inside zip archives
            if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError(f"Unsafe symlink entry in archive: {info.filename}")
            member_raw = str(info.filename or "").replace("\\", "/")
            if not member_raw or member_raw.startswith("/") or re.match(r"^[A-Za-z]:", member_raw):
                raise ValueError(f"Unsafe absolute archive path: {info.filename}")
            parts = [p for p in member_raw.split("/") if p and p != "."]
            if not parts or any(p == ".." for p in parts):
                raise ValueError(f"Unsafe Zip-Slip entry escapes destination: {info.filename}")
            candidate = target_root.joinpath(*parts).resolve()
            if candidate != target_root and target_root not in candidate.parents:
                raise ValueError(f"Unsafe archive entry escapes destination: {info.filename}")
            if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if info.file_size > max_entry_bytes:
                raise ValueError(f"Archive entry exceeds maximum size ({info.file_size} bytes): {info.filename}")

            candidate.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src_stream, open(candidate, "wb") as dst_stream:
                remaining = max_entry_bytes + 1
                while True:
                    chunk = src_stream.read(min(65536, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    if remaining <= 0:
                        raise ValueError(f"Archive entry exceeded decompressed limit: {info.filename}")
                    dst_stream.write(chunk)
            extracted.append(candidate)

    extracted.sort(key=lambda p: natural_sort_key(p.relative_to(target_root).as_posix()))
    return extracted


def extract_pdf_pages(
    pdf_path: Path | str,
    dest_dir: Path | str,
    *,
    dpi: int = 150,
    max_width: int = 1024,
) -> List[Path]:
    src = Path(pdf_path)
    out_dir = Path(dest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import fitz  # PyMuPDF
    except ImportError:
        fitz = None

    if fitz is not None:
        saved: List[Path] = []
        zoom = max(1.0, float(dpi) / 72.0)
        matrix = fitz.Matrix(zoom, zoom)
        with fitz.open(str(src)) as doc:
            for idx, page in enumerate(doc):
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                try:
                    if img.width > max_width:
                        new_h = max(1, int(round(img.height * max_width / img.width)))
                        img = img.resize((max_width, new_h), Image.Resampling.LANCZOS)
                    out_path = out_dir / f"pdf_p{idx + 1:04d}.jpg"
                    img.save(out_path, format="JPEG", quality=92, optimize=True)
                    saved.append(out_path)
                finally:
                    img.close()
        return saved

    try:
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise RuntimeError(
            "PDF manga ingestion requires 'pymupdf' (fitz) or 'pdf2image' (+ poppler), "
            "neither of which is installed in this environment. Install pymupdf (`pip install pymupdf`) "
            "or export pages to an image directory / .cbz archive."
        ) from exc

    images = convert_from_path(str(src), dpi=dpi)
    saved_pdf2img: List[Path] = []
    for idx, img in enumerate(images):
        try:
            rgb = img.convert("RGB")
            try:
                if rgb.width > max_width:
                    new_h = max(1, int(round(rgb.height * max_width / rgb.width)))
                    rgb = rgb.resize((max_width, new_h), Image.Resampling.LANCZOS)
                out_path = out_dir / f"pdf_p{idx + 1:04d}.jpg"
                rgb.save(out_path, format="JPEG", quality=92, optimize=True)
                saved_pdf2img.append(out_path)
            finally:
                rgb.close()
        finally:
            img.close()
    return saved_pdf2img


class _WebtoonHTMLParser(HTMLParser):
    def __init__(self, base_url: str = "") -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.image_urls: List[str] = []
        self._in_title = False
        self._in_h1 = False
        self.title_parts: List[str] = []
        self.h1_parts: List[str] = []
        self._image_list_depth = 0
        self._div_stack: List[bool] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attr_map = {k.lower(): (v or "") for k, v in attrs}
        tag_l = tag.lower()
        if tag_l == "title":
            self._in_title = True
        elif tag_l == "h1":
            self._in_h1 = True
        elif tag_l == "div":
            is_list = attr_map.get("id") == "_imageList" or "viewer_img" in attr_map.get("class", "")
            self._div_stack.append(is_list)
            if is_list:
                self._image_list_depth += 1
        elif tag_l == "img":
            cls_tokens = set(attr_map.get("class", "").split())
            raw_url = (attr_map.get("data-url") or attr_map.get("src") or "").strip()
            if not raw_url:
                return
            in_viewer = self._image_list_depth > 0 or "_images" in cls_tokens or "pstatic.net" in raw_url.lower()
            if not in_viewer:
                return
            lowered = raw_url.lower()
            if any(skip in lowered for skip in ("blank.gif", "loading", "/static/", "banner", "avatar", "icon")):
                return
            resolved = urljoin(self.base_url, raw_url) if self.base_url else raw_url
            if resolved.startswith(("http://", "https://")) and resolved not in self.image_urls:
                self.image_urls.append(resolved)

    def handle_endtag(self, tag: str) -> None:
        tag_l = tag.lower()
        if tag_l == "title":
            self._in_title = False
        elif tag_l == "h1":
            self._in_h1 = False
        elif tag_l == "div" and self._div_stack:
            was_list = self._div_stack.pop()
            if was_list and self._image_list_depth > 0:
                self._image_list_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._in_h1:
            self.h1_parts.append(data)
        elif self._in_title:
            self.title_parts.append(data)

    @property
    def extracted_title(self) -> str:
        h1 = " ".join("".join(self.h1_parts).split()).strip()
        if h1:
            return h1
        return " ".join("".join(self.title_parts).split()).strip()


def parse_webtoon_image_urls(html: str, base_url: str = "https://www.webtoons.com/") -> Tuple[str, List[str]]:
    parser = _WebtoonHTMLParser(base_url=base_url)
    parser.feed(html or "")
    parser.close()
    return parser.extracted_title, parser.image_urls


class _WebtoonSeriesListParser(HTMLParser):
    def __init__(self, base_url: str = "") -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self._in_title = False
        self._in_h1 = False
        self.title_parts: List[str] = []
        self.h1_parts: List[str] = []
        self._li_ep_stack: List[Optional[int]] = []
        self.episodes_by_no: dict[int, str] = {}

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attr_map = {k.lower(): (v or "") for k, v in attrs}
        tag_l = tag.lower()
        if tag_l == "title":
            self._in_title = True
        elif tag_l == "h1":
            self._in_h1 = True
        elif tag_l == "li":
            ep_raw = attr_map.get("data-episode-no", "").strip()
            self._li_ep_stack.append(int(ep_raw) if ep_raw.isdigit() else None)
        elif tag_l == "a":
            href = attr_map.get("href", "").strip()
            if not href:
                return
            resolved = urljoin(self.base_url, href) if self.base_url else href
            if not resolved.startswith(("http://", "https://")):
                return
            parsed = urlparse(resolved)
            ep_no: Optional[int] = None
            q_ep = parse_qs(parsed.query).get("episode_no", [])
            if q_ep and q_ep[0].isdigit():
                ep_no = int(q_ep[0])
            elif self._li_ep_stack and self._li_ep_stack[-1] is not None and "/viewer" in parsed.path.lower():
                ep_no = self._li_ep_stack[-1]
            if ep_no is not None and ep_no > 0:
                self.episodes_by_no.setdefault(ep_no, resolved)

    def handle_endtag(self, tag: str) -> None:
        tag_l = tag.lower()
        if tag_l == "title":
            self._in_title = False
        elif tag_l == "h1":
            self._in_h1 = False
        elif tag_l == "li" and self._li_ep_stack:
            self._li_ep_stack.pop()

    def handle_data(self, data: str) -> None:
        if self._in_h1:
            self.h1_parts.append(data)
        elif self._in_title:
            self.title_parts.append(data)

    @property
    def extracted_title(self) -> str:
        h1 = " ".join("".join(self.h1_parts).split()).strip()
        if h1:
            return h1
        return " ".join("".join(self.title_parts).split()).strip()


def parse_webtoon_episode_list_urls(
    html: str,
    base_url: str = "https://www.webtoons.com/",
) -> Tuple[str, List[Tuple[int, str]]]:
    parser = _WebtoonSeriesListParser(base_url=base_url)
    parser.feed(html or "")
    parser.close()
    items = sorted(parser.episodes_by_no.items(), key=lambda pair: pair[0])
    return parser.extracted_title, items


def _is_webtoon_series_list_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    path_l = parsed.path.lower()
    if "/viewer" in path_l or "episode_no=" in parsed.query.lower():
        return False
    if path_l.rstrip("/").endswith("/list") or ("title_no=" in parsed.query.lower() and "episode_no=" not in parsed.query.lower()):
        return True
    return False


def _validate_public_http_url(url: str, *, allow_private_urls: bool = False) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Unsupported or invalid comic URL: {url!r}")
    if allow_private_urls:
        return url.strip()
    host = (parsed.hostname or "").strip().lower()
    if not host or host == "localhost" or host.endswith(".local"):
        raise ValueError(f"Private/local host rejected for comic URL: {url!r}")
    try:
        ip_obj = ipaddress.ip_address(host)
        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_reserved or ip_obj.is_multicast:
            raise ValueError(f"Private IP rejected for comic URL: {url!r}")
    except ValueError as exc:
        if "Private IP rejected" in str(exc):
            raise
        try:
            infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
            for info in infos:
                addr = info[4][0]
                ip_resolved = ipaddress.ip_address(addr)
                if ip_resolved.is_private or ip_resolved.is_loopback or ip_resolved.is_link_local or ip_resolved.is_reserved:
                    raise ValueError(f"URL resolves to private network address: {url!r}")
        except socket.gaierror:
            pass
    return url.strip()


async def _default_fetch_url(
    url: str,
    *,
    referer: str = "https://www.webtoons.com/",
    max_attempts: int = 3,
    retry_delay_sec: float = 0.75,
) -> bytes:
    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Referer": referer,
        "Accept": "text/html,application/xhtml+xml,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    timeout = aiohttp.ClientTimeout(total=60)
    last_exc: Optional[BaseException] = None
    attempts = max(1, int(max_attempts))
    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        for attempt in range(attempts):
            try:
                async with session.get(url) as resp:
                    if resp.status >= 400:
                        err = RuntimeError(f"HTTP {resp.status} fetching comic resource: {url}")
                        if resp.status in _RETRYABLE_HTTP_STATUS and attempt + 1 < attempts:
                            last_exc = err
                        else:
                            raise err
                    else:
                        return await resp.read()
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_exc = exc
                if attempt + 1 >= attempts:
                    raise RuntimeError(f"Failed fetching comic resource after {attempts} attempts ({url}): {exc}") from exc
            if attempt + 1 < attempts:
                await asyncio.sleep(retry_delay_sec * (2 ** attempt))
    assert last_exc is not None
    raise RuntimeError(str(last_exc))


async def download_webtoon_chapter(
    url: str,
    dest_dir: Path | str,
    *,
    referer: str = "https://www.webtoons.com/",
    fetcher=None,
    allow_private_urls: bool = False,
) -> Tuple[str, List[Path]]:
    safe_url = _validate_public_http_url(url, allow_private_urls=allow_private_urls)
    raw_dir = Path(dest_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    fetch_fn = fetcher or _default_fetch_url

    html_bytes = await fetch_fn(safe_url, referer=referer)
    html_text = html_bytes.decode("utf-8", errors="replace")
    title, image_urls = parse_webtoon_image_urls(html_text, base_url=safe_url)
    if not image_urls:
        raise RuntimeError(f"No comic viewer images found at Webtoon URL: {safe_url}")

    sem = asyncio.Semaphore(4)

    async def _download_one(idx: int, img_url: str) -> Path:
        _validate_public_http_url(img_url, allow_private_urls=allow_private_urls)
        async with sem:
            data = await fetch_fn(img_url, referer=safe_url)
        ext = Path(urlparse(img_url).path).suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            ext = ".jpg"
        out = raw_dir / f"raw_{idx + 1:04d}{ext}"
        out.write_bytes(data)
        return out

    raw_paths = await asyncio.gather(*(_download_one(i, u) for i, u in enumerate(image_urls)))
    stitched_paths = _stitch_short_strips(list(raw_paths), raw_dir / "_stitched")
    return title, stitched_paths


def _try_open_rgb(src_path: Path) -> Optional[Image.Image]:
    try:
        if not src_path.is_file() or src_path.stat().st_size == 0:
            logging.warning("Skipping missing or empty comic image: %s", src_path)
            return None
        with Image.open(src_path) as im:
            rgb = ImageOps.exif_transpose(im).convert("RGB")
            rgb.load()
        if rgb.width < 4 or rgb.height < 4:
            logging.warning("Skipping tiny/corrupt comic fragment (%dx%d): %s", rgb.width, rgb.height, src_path)
            rgb.close()
            return None
        return rgb
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        logging.warning("Skipping unreadable comic image %s: %s", src_path, exc)
        return None


def _stitch_short_strips(
    raw_paths: List[Path],
    stitched_dir: Path,
    *,
    chunk_max_height: int = 9000,
) -> List[Path]:
    """Group stacked thin Webtoon image strips into tall vertical sheets so gutter-splitting never bisects bubbles."""
    if len(raw_paths) <= 1:
        return [p for p in raw_paths if p.is_file() and p.stat().st_size > 0]
    stitched_dir.mkdir(parents=True, exist_ok=True)
    batches: List[List[Tuple[Image.Image, str]]] = []
    cur_batch: List[Tuple[Image.Image, str]] = []
    cur_h = 0

    try:
        for p in raw_paths:
            rgb = _try_open_rgb(p)
            if rgb is None:
                continue
            if cur_batch and cur_h + rgb.height > chunk_max_height:
                batches.append(cur_batch)
                cur_batch = []
                cur_h = 0
            cur_batch.append((rgb, p.name))
            cur_h += rgb.height
        if cur_batch:
            batches.append(cur_batch)

        out_paths: List[Path] = []
        for b_idx, items in enumerate(batches):
            if len(items) == 1:
                out_p = stitched_dir / f"strip_{b_idx + 1:03d}.jpg"
                items[0][0].save(out_p, format="JPEG", quality=93, optimize=True)
                out_paths.append(out_p)
            else:
                target_w = max(im.width for im, _ in items)
                resized: List[Image.Image] = []
                total_h = 0
                for im, _ in items:
                    if im.width != target_w:
                        new_h = max(1, int(round(im.height * target_w / im.width)))
                        r = im.resize((target_w, new_h), Image.Resampling.LANCZOS)
                        resized.append(r)
                        total_h += new_h
                    else:
                        resized.append(im)
                        total_h += im.height
                canvas = Image.new("RGB", (target_w, total_h), (255, 255, 255))
                y = 0
                for im in resized:
                    canvas.paste(im, (0, y))
                    y += im.height
                out_p = stitched_dir / f"strip_{b_idx + 1:03d}.jpg"
                canvas.save(out_p, format="JPEG", quality=93, optimize=True)
                canvas.close()
                for im in resized:
                    if all(im is not orig for orig, _ in items):
                        im.close()
                out_paths.append(out_p)
        return out_paths
    finally:
        for group in batches:
            for im, _ in group:
                try:
                    im.close()
                except Exception:
                    pass


def _find_gutter_split_y(
    gray: np.ndarray,
    start_y: int,
    height: int,
    *,
    split_height_threshold: int,
    target_slice_height: int,
) -> int:
    remaining = height - start_y
    if remaining <= split_height_threshold:
        return height

    min_step = max(200, int(round(target_slice_height * 0.65)))
    max_step = min(remaining - 200, split_height_threshold)
    if max_step <= min_step:
        return start_y + min(remaining, target_slice_height)

    lo = start_y + min_step
    hi = start_y + max_step
    band = gray[lo:hi]
    row_std = band.std(axis=1)

    # Prefer the midpoint of the longest solid uniform horizontal gutter band closest to target_slice_height
    ideal_local = max(0, min(len(row_std) - 1, (start_y + target_slice_height) - lo))
    flat_mask = row_std <= 4.5
    best_center: Optional[int] = None
    best_dist = float("inf")
    run_start: Optional[int] = None

    for idx, is_flat in enumerate(flat_mask):
        if is_flat and run_start is None:
            run_start = idx
        elif not is_flat and run_start is not None:
            if idx - run_start >= 8:
                center = (run_start + idx - 1) // 2
                dist = abs(center - ideal_local)
                if dist < best_dist:
                    best_dist = dist
                    best_center = center
            run_start = None
    if run_start is not None and len(row_std) - run_start >= 8:
        center = (run_start + len(row_std) - 1) // 2
        if abs(center - ideal_local) < best_dist:
            best_center = center

    if best_center is not None:
        return lo + best_center

    # Fallback: pick row with smallest 5-row smoothed variance near ideal height
    if len(row_std) >= 5:
        kernel = np.ones(5, dtype=np.float32) / 5.0
        smoothed = np.convolve(row_std, kernel, mode="same")
        score = smoothed + np.abs(np.arange(len(row_std)) - ideal_local) * 0.015
        return lo + int(np.argmin(score))
    return lo + int(np.argmin(row_std))


def split_and_normalize_images(
    raw_paths: Sequence[Path],
    out_dir: Path,
    *,
    chapter: int,
    max_width: int = 1024,
    split_height_threshold: int = 2600,
    target_slice_height: int = 1900,
) -> List[MangaPage]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pages: List[MangaPage] = []
    page_counter = 0

    for src_path in raw_paths:
        im = _try_open_rgb(Path(src_path))
        if im is None:
            continue
        try:
            if im.width > max_width:
                new_h = max(1, int(round(im.height * max_width / im.width)))
                resized = im.resize((max_width, new_h), Image.Resampling.LANCZOS)
                im.close()
                im = resized

            if im.height <= split_height_threshold:
                out_file = out_dir / f"p{page_counter + 1:04d}.jpg"
                im.save(out_file, format="JPEG", quality=92, optimize=True)
                pages.append(
                    MangaPage(
                        page_idx=page_counter,
                        chapter=chapter,
                        path=str(out_file),
                        width=im.width,
                        height=im.height,
                        source_name=src_path.name,
                    )
                )
                page_counter += 1
            else:
                gray = np.asarray(im.convert("L"), dtype=np.float32)
                y = 0
                while y < im.height:
                    next_y = _find_gutter_split_y(
                        gray,
                        y,
                        im.height,
                        split_height_threshold=split_height_threshold,
                        target_slice_height=target_slice_height,
                    )
                    if next_y <= y:
                        next_y = min(im.height, y + target_slice_height)
                    slice_im = im.crop((0, y, im.width, next_y))
                    out_file = out_dir / f"p{page_counter + 1:04d}.jpg"
                    slice_im.save(out_file, format="JPEG", quality=92, optimize=True)
                    slice_w, slice_h = slice_im.size
                    slice_im.close()
                    pages.append(
                        MangaPage(
                            page_idx=page_counter,
                            chapter=chapter,
                            path=str(out_file),
                            width=slice_w,
                            height=slice_h,
                            source_name=f"{src_path.name}#y{y}-{next_y}",
                        )
                    )
                    page_counter += 1
                    y = next_y
        finally:
            im.close()
    return pages


def _normalize_source_list(sources: str | Path | Sequence[str | Path]) -> List[str]:
    if isinstance(sources, Path):
        return [str(sources)]
    if isinstance(sources, str):
        lines = [part.strip() for part in re.split(r"[\r\n]+", sources) if part.strip()]
        if len(lines) == 1 and "," in lines[0] and not Path(lines[0]).exists():
            return [p.strip() for p in lines[0].split(",") if p.strip()]
        return lines
    if isinstance(sources, (list, tuple, Iterable)):
        return [str(item).strip() for item in sources if str(item).strip()]
    raise TypeError(f"Unsupported sources type: {type(sources).__name__}")


def _is_http_url(text: str) -> bool:
    parsed = urlparse(text.strip())
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def _is_raw_chapter_dir_name(name: str) -> bool:
    return bool(re.search(r"(?:^|[_\-\s])raw$", str(name or "").strip(), re.IGNORECASE))


def _discover_local_directory_chapters(
    root_dir: Path,
    next_default_ch: int,
    *,
    max_depth: int = 3,
) -> Tuple[List[Tuple[int, str, str, object]], int]:
    """Discover chapter folders, archives, and PDFs (including nested Volume/Season dirs), deduplicating *_raw siblings."""
    direct_imgs = sorted(
        (p for p in root_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: natural_sort_key(p.name),
    )
    if direct_imgs:
        ch_num = infer_chapter_number(root_dir.name, next_default_ch)
        return (
            [(ch_num, root_dir.name, str(root_dir), ("images", direct_imgs))],
            max(next_default_ch + 1, ch_num + 1),
        )

    # Collect (has_explicit_num, inferred_ch, is_raw, label, src_label, payload)
    raw_entries: List[Tuple[bool, Optional[int], bool, str, str, object]] = []

    def _walk_dir(directory: Path, depth: int) -> None:
        children = sorted(
            (
                c
                for c in directory.iterdir()
                if not c.name.startswith(".")
                and c.name.lower() not in _SKIP_DIR_NAMES
                and (
                    c.is_dir()
                    or (c.is_file() and (c.suffix.lower() in ARCHIVE_EXTENSIONS or c.suffix.lower() == ".pdf"))
                )
            ),
            key=lambda p: natural_sort_key(p.name),
        )
        for child in children:
            if child.is_dir():
                imgs = sorted(
                    (p for p in child.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                    key=lambda p: natural_sort_key(p.name),
                )
                if imgs:
                    has_num = bool(re.search(r"\d+", child.name))
                    inferred = infer_chapter_number(child.name, 0) if has_num else None
                    raw_entries.append(
                        (has_num, inferred, _is_raw_chapter_dir_name(child.name), child.name, str(child), ("images", imgs))
                    )
                elif depth < max_depth:
                    _walk_dir(child, depth + 1)
            elif child.suffix.lower() in ARCHIVE_EXTENSIONS:
                has_num = bool(re.search(r"\d+", child.stem))
                inferred = infer_chapter_number(child.stem, 0) if has_num else None
                raw_entries.append((has_num, inferred, False, child.stem, str(child), ("cbz", child)))
            elif child.suffix.lower() == ".pdf":
                has_num = bool(re.search(r"\d+", child.stem))
                inferred = infer_chapter_number(child.stem, 0) if has_num else None
                raw_entries.append((has_num, inferred, False, child.stem, str(child), ("pdf", child)))

    _walk_dir(root_dir, 1)

    non_raw_explicit_chapters = {
        ch_no
        for has_num, ch_no, is_raw, *_ in raw_entries
        if has_num and ch_no is not None and not is_raw
    }
    filtered = [
        entry
        for entry in raw_entries
        if not (entry[0] and entry[2] and entry[1] in non_raw_explicit_chapters)
    ]

    discovered: List[Tuple[int, str, str, object]] = []
    cur_default = next_default_ch
    for has_num, inferred, _is_raw, label, src_label, payload in filtered:
        ch_num = inferred if has_num and inferred is not None else cur_default
        cur_default = max(cur_default + 1, ch_num + 1)
        discovered.append((ch_num, label, src_label, payload))
    return discovered, cur_default


async def ingest_manga_sources(
    sources: str | Path | Sequence[str | Path],
    working_dir: str | Path,
    *,
    chapter_range: str | int | Sequence[int] | None = None,
    max_width: int = 1024,
    split_height_threshold: int = 2600,
    target_slice_height: int = 1900,
    webtoon_fetcher=None,
    allow_private_urls: bool = False,
) -> ComicManifest:
    work_root = Path(working_dir).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    sources_root = work_root / "sources"
    sources_root.mkdir(parents=True, exist_ok=True)

    selected_chapters = parse_chapter_range(chapter_range)
    source_items = _normalize_source_list(sources)
    if not source_items:
        raise ValueError("No manga/manhwa sources provided.")

    chapter_specs: List[Tuple[int, str, str, object]] = []
    # Entry tuple: (chapter_num, title, source_str, payload_kind_and_value)
    next_default_ch = 1
    standalone_images: List[Path] = []
    fetch_fn = webtoon_fetcher or _default_fetch_url

    for item in source_items:
        if _is_http_url(item):
            if _is_webtoon_series_list_url(item):
                safe_list_url = _validate_public_http_url(item, allow_private_urls=allow_private_urls)
                list_html = (await fetch_fn(safe_list_url, referer=safe_list_url)).decode("utf-8", errors="replace")
                series_title, ep_pairs = parse_webtoon_episode_list_urls(list_html, base_url=safe_list_url)
                if not ep_pairs:
                    raise RuntimeError(f"No episode viewer links found at Webtoon series list URL: {safe_list_url}")
                for ep_no, ep_url in ep_pairs:
                    ep_title = f"{series_title} - Chapter {ep_no}" if series_title else f"Chapter {ep_no}"
                    next_default_ch = max(next_default_ch + 1, ep_no + 1)
                    chapter_specs.append((ep_no, ep_title, ep_url, ("url", ep_url)))
                continue
            ch_num = infer_chapter_number(item, next_default_ch)
            next_default_ch = max(next_default_ch + 1, ch_num + 1)
            chapter_specs.append((ch_num, f"Chapter {ch_num}", item, ("url", item)))
            continue

        path = Path(item).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Manga source does not exist: {item}")

        if path.is_dir():
            found_specs, next_default_ch = _discover_local_directory_chapters(path, next_default_ch)
            chapter_specs.extend(found_specs)
        elif path.is_file():
            ext = path.suffix.lower()
            if ext in ARCHIVE_EXTENSIONS:
                ch_num = infer_chapter_number(path.stem, next_default_ch)
                next_default_ch = max(next_default_ch + 1, ch_num + 1)
                chapter_specs.append((ch_num, path.stem, str(path), ("cbz", path)))
            elif ext == ".pdf":
                ch_num = infer_chapter_number(path.stem, next_default_ch)
                next_default_ch = max(next_default_ch + 1, ch_num + 1)
                chapter_specs.append((ch_num, path.stem, str(path), ("pdf", path)))
            elif ext in IMAGE_EXTENSIONS:
                standalone_images.append(path)
            else:
                raise ValueError(f"Unsupported manga source file type: {path.name}")

    if standalone_images:
        standalone_images.sort(key=lambda p: natural_sort_key(p.name))
        ch_num = infer_chapter_number(standalone_images[0].parent.name, next_default_ch)
        chapter_specs.append((ch_num, f"Chapter {ch_num}", str(standalone_images[0].parent), ("images", standalone_images)))

    if selected_chapters is not None:
        chapter_specs = [spec for spec in chapter_specs if spec[0] in selected_chapters]

    if not chapter_specs:
        raise ValueError(f"No readable comic chapters matched sources/chapter_range={chapter_range!r}.")

    # Ensure unique ascending chapter ordering without silent clobbering
    chapter_specs.sort(key=lambda item: (item[0], natural_sort_key(item[1])))
    chapters: List[ChapterPages] = []
    seen_chapters: Set[int] = set()

    for idx, (ch_num, title, src_label, kind_payload) in enumerate(chapter_specs):
        effective_ch = ch_num
        while effective_ch in seen_chapters:
            effective_ch += 1
        seen_chapters.add(effective_ch)

        ch_dir = sources_root / f"ch_{effective_ch:02d}"
        raw_stage = ch_dir / "_raw"
        kind, val = kind_payload  # type: ignore[misc]
        resolved_title = title

        if kind == "images":
            raw_paths = list(val)
        elif kind == "cbz":
            raw_paths = extract_cbz_archive(val, raw_stage)
        elif kind == "pdf":
            raw_paths = extract_pdf_pages(val, raw_stage, max_width=max_width)
        elif kind == "url":
            dl_title, raw_paths = await download_webtoon_chapter(
                val,
                raw_stage,
                fetcher=webtoon_fetcher,
                allow_private_urls=allow_private_urls,
            )
            if dl_title:
                resolved_title = dl_title
        else:
            raise ValueError(f"Unknown ingestion source kind: {kind}")

        if not raw_paths:
            raise ValueError(f"Chapter {effective_ch} ({src_label}) produced 0 images.")

        normalized_pages = split_and_normalize_images(
            raw_paths,
            ch_dir,
            chapter=effective_ch,
            max_width=max_width,
            split_height_threshold=split_height_threshold,
            target_slice_height=target_slice_height,
        )
        if not normalized_pages:
            raise ValueError(f"Chapter {effective_ch} ({src_label}) produced 0 readable images after decoding.")
        chapters.append(
            ChapterPages(
                chapter=effective_ch,
                title=resolved_title,
                source=src_label,
                pages=normalized_pages,
            )
        )

    manifest = ComicManifest(
        working_dir=str(work_root),
        chapters=chapters,
        total_pages=sum(len(ch.pages) for ch in chapters),
    )
    (work_root / "comic_manifest.json").write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return manifest


def group_chapters_for_episodes(
    manifest: ComicManifest,
    chapters_per_episode: int = 1,
) -> List[ComicManifest]:
    """Partition a full-series ComicManifest sequentially into per-episode ComicManifest slices."""
    per_ep = int(chapters_per_episode)
    if per_ep < 1:
        raise ValueError(f"chapters_per_episode must be >= 1, got {chapters_per_episode!r}")
    if not manifest.chapters:
        return []
    episodes: List[ComicManifest] = []
    for start in range(0, len(manifest.chapters), per_ep):
        chunk = list(manifest.chapters[start : start + per_ep])
        episodes.append(
            ComicManifest(
                working_dir=manifest.working_dir,
                chapters=chunk,
                total_pages=sum(len(ch.pages) for ch in chunk),
            )
        )
    return episodes
