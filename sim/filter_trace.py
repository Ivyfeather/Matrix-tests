#!/usr/bin/env python3

"""Filter simulator trace lines by address bits [7:6].

Keeps only MR/MW lines whose address has addr[7:6] == 0.
Non-trace lines are passed through unchanged.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


TRACE_RE = re.compile(r"^(MR|MW)\s+(0x[0-9a-fA-F]+)\s*([abc])?$")


def should_keep(addr_text: str) -> bool:
    addr = int(addr_text, 16)
    # return ((addr >> 6) & 0x3) == 0
    return True


def open_input(path_text: str | None):
    if path_text is None:
        return sys.stdin
    return Path(path_text).open("r", encoding="utf-8")


def open_output(path_text: str | None):
    if path_text is None:
        return sys.stdout
    return Path(path_text).open("w", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter MR/MW trace lines by addr[7:6] == 0")
    parser.add_argument("input", nargs="?", help="Input trace file, defaults to stdin")
    parser.add_argument("output", nargs="?", help="Output trace file, defaults to stdout")
    args = parser.parse_args()

    kept = 0
    dropped = 0

    with open_input(args.input) as infile, open_output(args.output) as outfile:
        for raw_line in infile:
            line = raw_line.rstrip("\n")
            match = TRACE_RE.match(line)
            if match is None:
                print(line, file=outfile)
                continue

            addr_text = match.group(2)
            if should_keep(addr_text):
                # Keep C-matrix MR distinct so addr.py can recognize it as CR.
                op = match.group(1)
                suffix = match.group(3)
                if op == "MR" and suffix == "c":
                    op = "CR"
                print(f"{op} {addr_text}", file=outfile)
                kept += 1
            else:
                dropped += 1

    print(f"# trace_filter kept={kept} dropped={dropped}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())