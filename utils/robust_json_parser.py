# Some chat models (observed with gemini-flash-lite via the OpenAI-compatible
# endpoint and muse.ai web DOM rendering) frequently emit a trailing comma
# before a closing `}`/`]`, bare language headers like `json\n{...}` without
# markdown backticks, or prose around structured JSON responses.
# Wrap PydanticOutputParser so a parse failure retries locally with formatting
# artifacts cleaned before giving up.
import re
from typing import Iterator, List

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.outputs import Generation

_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_LEADING_FENCE_OR_LABEL_RE = re.compile(
    r"^\s*(?:(?:```|~~~)\s*(?:json[a-zA-Z0-9_-]*)?|(?:json5|jsonc|json)\s*(?:\r?\n+\s*(?:Copy(?:\s+code)?|复制代码))?)\s*(?:\r?\n|$)",
    re.IGNORECASE,
)
_TRAILING_FENCE_RE = re.compile(r"(?:\r?\n)?\s*(?:```|~~~)\s*$")


def strip_trailing_commas(text: str) -> str:
    return _TRAILING_COMMA_RE.sub(r"\1", text)


def strip_markdown_json_wrapper(text: str) -> str:
    cleaned = text.lstrip("\ufeff").strip()
    cleaned = _LEADING_FENCE_OR_LABEL_RE.sub("", cleaned, count=1)
    cleaned = _TRAILING_FENCE_RE.sub("", cleaned, count=1)
    return cleaned.strip()


def iter_json_blocks(text: str) -> Iterator[str]:
    """Yield balanced JSON object/array substrings in order, respecting JSON strings."""
    idx = 0
    length = len(text)
    while idx < length:
        ch = text[idx]
        if ch not in "{[":
            idx += 1
            continue
        stack = ["}" if ch == "{" else "]"]
        in_string = False
        escaped = False
        matched_end = -1
        for end in range(idx + 1, length):
            cur = text[end]
            if in_string:
                if escaped:
                    escaped = False
                elif cur == "\\":
                    escaped = True
                elif cur == '"':
                    in_string = False
                continue
            if cur == '"':
                in_string = True
            elif cur == "{":
                stack.append("}")
            elif cur == "[":
                stack.append("]")
            elif cur in "}]":
                if not stack or cur != stack[-1]:
                    break
                stack.pop()
                if not stack:
                    matched_end = end
                    break
        if matched_end != -1:
            yield text[idx : matched_end + 1]
            idx = matched_end + 1
        else:
            idx += 1


def repair_unescaped_quotes_in_json(text: str) -> str:
    """Escape raw unescaped double quotes & control newlines inside JSON string values.

    Chat LLMs and Markdown DOM renderers often emit prose quotes inside JSON
    values (e.g. `... says: "Hello""`), breaking standard `json.loads`.
    Only intended as a fallback candidate after normal parsing fails.
    """
    n = len(text)
    if not text or '"' not in text:
        return text

    _NEXT_KEY_RE = re.compile(r'^"(?:\\.|[^"\\\r\n]){1,120}"\s*:')
    _PRIM_OR_VAL_END_RE = re.compile(r'^(?:[{}\[\]"]|true\b|false\b|null\b|-?\d)', re.IGNORECASE)

    def _skip_ws(idx: int) -> int:
        while idx < n and text[idx] in " \t\r\n":
            idx += 1
        return idx

    out: List[str] = []
    # Container frames: ('obj', phase) or ('arr', phase)
    frames: List[List[str]] = []
    in_string = False
    str_role = "val"  # 'key' | 'val'
    i = 0

    while i < n:
        ch = text[i]
        if not in_string:
            out.append(ch)
            if ch == "{":
                frames.append(["obj", "key_or_end"])
            elif ch == "[":
                frames.append(["arr", "val_or_end"])
            elif ch == ":" and frames and frames[-1][0] == "obj":
                frames[-1][1] = "val"
            elif ch == "," and frames:
                frames[-1][1] = "key" if frames[-1][0] == "obj" else "val"
            elif ch in "}]" and frames:
                frames.pop()
                if frames:
                    frames[-1][1] = "after_val"
            elif ch == '"':
                in_string = True
                if frames and frames[-1][0] == "obj" and frames[-1][1] in ("key_or_end", "key"):
                    str_role = "key"
                else:
                    str_role = "val"
            i += 1
            continue

        # Inside string literal
        if ch == "\\":
            out.append(ch)
            if i + 1 < n:
                out.append(text[i + 1])
                i += 2
            else:
                i += 1
            continue
        if ch == "\n":
            out.append("\\n")
            i += 1
            continue
        if ch == "\r":
            out.append("\\r")
            i += 1
            continue
        if ch == "\t":
            out.append("\\t")
            i += 1
            continue
        if ch == '"':
            j = _skip_ws(i + 1)
            nxt = text[j] if j < n else ""
            is_closing = False
            if str_role == "key":
                is_closing = (nxt == ":")
            else:
                expected_close = "}" if (frames and frames[-1][0] == "obj") else "]" if (frames and frames[-1][0] == "arr") else ""
                if j >= n:
                    is_closing = True
                elif nxt == expected_close and nxt in ("}", "]"):
                    k = _skip_ws(j + 1)
                    after_close = text[k] if k < n else ""
                    if k >= n or after_close in (",", "}", "]"):
                        is_closing = True
                elif nxt == ",":
                    k = _skip_ws(j + 1)
                    if frames and frames[-1][0] == "obj":
                        tail_slice = text[k:]
                        if k < n and (text[k] == "}" or bool(_NEXT_KEY_RE.match(tail_slice))):
                            is_closing = True
                    elif frames and frames[-1][0] == "arr":
                        tail_slice = text[k:]
                        if k < n and bool(_PRIM_OR_VAL_END_RE.match(tail_slice)):
                            is_closing = True
                    else:
                        is_closing = True

            if is_closing:
                out.append('"')
                in_string = False
                if frames:
                    frames[-1][1] = "after_key" if str_role == "key" else "after_val"
            else:
                out.append('\\"')
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def _candidate_json_texts(raw_text: str) -> List[str]:
    candidates: List[str] = []
    seen = {raw_text}

    def _add(text: str | None) -> None:
        if text and text not in seen:
            seen.add(text)
            candidates.append(text)

    _add(strip_trailing_commas(raw_text))

    unwrapped = strip_markdown_json_wrapper(raw_text)
    _add(unwrapped)
    _add(strip_trailing_commas(unwrapped))

    repaired_raw = repair_unescaped_quotes_in_json(raw_text)
    repaired_unwrapped = repair_unescaped_quotes_in_json(unwrapped)
    _add(repaired_unwrapped)
    _add(strip_trailing_commas(repaired_unwrapped))
    _add(repaired_raw)
    _add(strip_trailing_commas(repaired_raw))

    for source in (unwrapped, raw_text, repaired_unwrapped, repaired_raw):
        for block in iter_json_blocks(source):
            _add(block)
            _add(strip_trailing_commas(block))
            repaired_block = repair_unescaped_quotes_in_json(block)
            _add(repaired_block)
            _add(strip_trailing_commas(repaired_block))

    return candidates


class TrailingCommaTolerantPydanticOutputParser(PydanticOutputParser):
    """PydanticOutputParser tolerant of trailing commas, bare language labels, and surrounding prose."""

    def parse_result(self, result: List[Generation], *, partial: bool = False):
        try:
            return super().parse_result(result, partial=partial)
        except OutputParserException as first_exc:
            if not result:
                raise
            raw_text = result[0].text
            for candidate in _candidate_json_texts(raw_text):
                try:
                    return super().parse_result([Generation(text=candidate)], partial=partial)
                except OutputParserException:
                    continue
            raise first_exc
