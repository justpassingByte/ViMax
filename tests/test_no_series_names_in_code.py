"""Guard rail: no series-specific names/aliases may appear in generic code.

All series data must live under ``series/<slug>/``; the pipeline code must stay
series-agnostic. This test reads every name/alias/title from
``series/*/characters.yaml`` + ``series.yaml`` and fails if any appears (word-
boundary, case-insensitive) in agents/ pipelines/ utils/ tools/ interfaces/
agent_runtime/.
"""

import re
import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
CODE_DIRS = ["agents", "pipelines", "utils", "tools", "interfaces", "agent_runtime"]
ALLOWLIST_FILE = REPO / "tests" / "series_name_allowlist.txt"


def _collect_series_names():
    names = set()
    series_root = REPO / "series"
    if not series_root.is_dir():
        return names
    for slug_dir in series_root.iterdir():
        if not slug_dir.is_dir():
            continue
        sfile = slug_dir / "series.yaml"
        if sfile.is_file():
            data = yaml.safe_load(sfile.read_text(encoding="utf-8")) or {}
            title = str(data.get("title", "")).strip()
            if title:
                names.add(title)
        cfile = slug_dir / "characters.yaml"
        if cfile.is_file():
            data = yaml.safe_load(cfile.read_text(encoding="utf-8")) or {}
            for ch in (data.get("characters") or []):
                for alias in (ch.get("aliases") or []):
                    if isinstance(alias, str) and alias.strip():
                        names.add(alias.strip())
    return names


def _allowlisted_files():
    if not ALLOWLIST_FILE.is_file():
        return set()
    return {
        line.strip()
        for line in ALLOWLIST_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


class NoSeriesNamesInCodeTests(unittest.TestCase):
    def test_no_series_names_leak_into_generic_code(self):
        names = _collect_series_names()
        self.assertTrue(names, "expected at least one series name to check")
        patterns = [(n, re.compile(r"\b" + re.escape(n) + r"\b", re.IGNORECASE)) for n in names]
        allow = _allowlisted_files()

        violations = []
        for d in CODE_DIRS:
            base = REPO / d
            if not base.is_dir():
                continue
            for path in base.rglob("*.py"):
                if "__pycache__" in path.parts:
                    continue
                rel = path.relative_to(REPO).as_posix()
                if rel in allow:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                for name, pat in patterns:
                    if pat.search(text):
                        violations.append(f"{rel}: contains series name '{name}'")

        self.assertEqual(violations, [], "Series-specific names found in generic code:\n" + "\n".join(violations))


if __name__ == "__main__":
    unittest.main()
