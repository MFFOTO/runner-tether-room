# -*- coding: utf-8 -*-
"""Fetch the pieces this tool uses but does not own.

  runner_suite_core.py   the crop engine, from the public crop repo
  luts/*.cube            the colour grades, from the public LUT repo

Both are kept out of this repo deliberately: the crop engine keeps improving
upstream, and forking it here is how a copy quietly falls behind. Re-run this
any time to pick up newer versions.

    python fetch_deps.py            # fetch anything missing
    python fetch_deps.py --update   # overwrite with the latest
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

CORE_URL = ("https://raw.githubusercontent.com/MFFOTO/runner-suite-stage6/"
            "master/runner_suite_core.py")
EXTRA = {
    "check_cuda_environment.py": ("https://raw.githubusercontent.com/MFFOTO/"
                                  "runner-suite-stage6/master/check_cuda_environment.py"),
}
LUT_API = "https://api.github.com/repos/MFFOTO/runner-lut-suite/contents/luts"


def get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "tether-room-setup"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_file(name: str, url: str, update: bool) -> bool:
    dst = HERE / name
    if dst.exists() and not update:
        print(f"  {name}: already here (--update to refresh)")
        return True
    try:
        data = get(url)
    except urllib.error.URLError as exc:
        print(f"  {name}: FAILED ({exc})")
        return False
    dst.write_bytes(data)
    print(f"  {name}: {len(data) / 1024:.0f} KB")
    return True


def fetch_luts(update: bool) -> bool:
    """The LUT repo's .cube files. Optional -- the pipeline runs ungraded
    without them, and any folder of .cube files can be pointed at instead."""
    out = HERE / "luts"
    if out.exists() and any(out.glob("*.cube")) and not update:
        n = len(list(out.glob("*.cube")))
        print(f"  luts/: {n} .cube already here (--update to refresh)")
        return True
    out.mkdir(exist_ok=True)
    try:
        import json
        entries = json.loads(get(LUT_API).decode("utf-8"))
    except Exception as exc:
        print(f"  luts/: could not list them ({exc}) -- point lut.lut_folder at "
              f"your own .cube folder instead")
        return True                      # not fatal
    got = 0
    for e in entries:
        if not str(e.get("name", "")).lower().endswith(".cube"):
            continue
        try:
            (out / e["name"]).write_bytes(get(e["download_url"]))
            got += 1
        except Exception as exc:
            print(f"    {e['name']}: failed ({exc})")
    print(f"  luts/: {got} .cube file(s)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch the crop engine and LUTs")
    ap.add_argument("--update", action="store_true",
                    help="overwrite existing files with the latest")
    args = ap.parse_args()

    print("Fetching dependencies ...")
    ok = fetch_file("runner_suite_core.py", CORE_URL, args.update)
    for name, url in EXTRA.items():
        fetch_file(name, url, args.update)
    fetch_luts(args.update)

    if not ok:
        print("\nrunner_suite_core.py is required -- the tool cannot run without it.")
        return 1
    print("\nReady.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
