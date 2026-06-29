#!/usr/bin/env python3
"""Check paper text for submission-facing wording hazards.

The check is intentionally conservative and only inspects the Typst source that
can affect paper-facing prose. Internal result directory names are ignored.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Finding:
    line_no: int
    rule: str
    text: str


def is_comment(line: str) -> bool:
    return line.lstrip().startswith("//")


def is_internal_path_line(line: str) -> bool:
    stripped = line.strip()
    if "image(" in stripped:
        return True
    return any(token in stripped for token in ("paper_official_", "official_4060_", "official_v100_", "official_3080_"))


def visible_lines(path: Path) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if is_comment(line) or is_internal_path_line(line):
            continue
        out.append((line_no, line))
    return out


def check_paper(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    rules = [
        ("visible-official", re.compile(r"\bofficial\b", re.IGNORECASE)),
        ("todo-marker", re.compile(r"\bTODO\b", re.IGNORECASE)),
        ("graph-positive-training", re.compile(r"Graph-positive short training", re.IGNORECASE)),
        ("deployable-controller-claim", re.compile(r"deployable (online )?controller", re.IGNORECASE)),
        ("production-throughput-claim", re.compile(r"production (serving )?throughput proof", re.IGNORECASE)),
        ("online-admission-overclaim", re.compile(r"worker-local online admission problem", re.IGNORECASE)),
        ("two-device-proof-overclaim", re.compile(r"two devices are enough to show", re.IGNORECASE)),
        ("production-trace-overclaim", re.compile(r"production traces support treating", re.IGNORECASE)),
        ("appendix-heading", re.compile(r"^=\s+Appendix\b", re.IGNORECASE)),
        ("moved-to-appendix", re.compile(r"moved to the appendix", re.IGNORECASE)),
    ]

    for line_no, line in visible_lines(path):
        for rule, pattern in rules:
            if pattern.search(line):
                findings.append(Finding(line_no=line_no, rule=rule, text=line.strip()))
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper",
        type=Path,
        required=True,
        help="Typst paper source to check.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    findings = check_paper(args.paper)
    if not findings:
        print(f"OK: no submission-hygiene findings in {args.paper}")
        return 0

    print(f"Found {len(findings)} submission-hygiene issue(s) in {args.paper}:")
    for finding in findings:
        print(f"{args.paper}:{finding.line_no}: {finding.rule}: {finding.text}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
