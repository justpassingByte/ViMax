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

    for source in (unwrapped, raw_text):
        for block in iter_json_blocks(source):
            _add(block)
            _add(strip_trailing_commas(block))

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
