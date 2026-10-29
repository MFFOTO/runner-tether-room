# -*- coding: utf-8 -*-
"""Compare two crop reports: did the same Premium crops come out?

Use it to check a speed experiment (a different detection size, the fast path on
or off) on your own photos. Run the same folder twice with "crop photos again"
ticked (or --recrop), then:

    python compare_reports.py runner_crop_report_A.csv runner_crop_report_B.csv

A crop is identified by its photo and the runner's number within it, so a small
change in a score does not make it look different. Photos and crops that appear
in only one report are listed.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


def premium(path: Path) -> Tuple[Set[Tuple[str, str]], Dict[str, int]]:
    crops: Set[Tuple[str, str]] = set()
    reasons: Dict[str, int] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            photo = Path(row.get("image", "")).name.lower()
            if row.get("decision") == "accept" and row.get("quality_class") == "premium":
                crops.add((photo, str(row.get("candidate_idx", ""))))
            else:
                key = row.get("reason") or row.get("decision") or "?"
                reasons[key] = reasons.get(key, 0) + 1
    return crops, reasons


def main(argv: List[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    a_path, b_path = Path(argv[1]), Path(argv[2])
    a, ra = premium(a_path)
    b, rb = premium(b_path)
    both, only_a, only_b = a & b, a - b, b - a
    print(f"A  {a_path.name}: {len(a)} Premium crops")
    print(f"B  {b_path.name}: {len(b)} Premium crops")
    print(f"in both: {len(both)}   only in A: {len(only_a)}   only in B: {len(only_b)}")
    for label, extra in (("only in A", only_a), ("only in B", only_b)):
        if extra:
            print(f"\n{label} (first 20):")
            for photo, idx in sorted(extra)[:20]:
                print(f"  {photo}  runner {idx}")
    print("\nwhy the other candidates were not kept (A | B):")
    for reason in sorted(set(ra) | set(rb), key=lambda r: -(ra.get(r, 0) + rb.get(r, 0)))[:12]:
        print(f"  {reason:34s} {ra.get(reason, 0):6d} | {rb.get(reason, 0):6d}")
    print("\nIDENTICAL: the same Premium crops came out." if not only_a and not only_b
          else "\nDIFFERENT: see the lists above.")
    return 0 if not only_a and not only_b else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
