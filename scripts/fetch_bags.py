#!/usr/bin/env python3
"""Pulls new and changed runs from the kart into a local folder, resuming partial files, and confirms finished runs.

https://github.com/EVC-Purdue/A27/issues/20

Usage: scripts/fetch_bags.py http://<kart>:8000 [--out bags] [--since <unix s>]
"""

import argparse
import json
import os
import shutil
import urllib.parse
import urllib.request

CHUNK = 1 << 20


def get_json(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def download(url, path, size):
    have = os.path.getsize(path) if os.path.exists(path) else 0
    if have == size:
        return False
    if have > size:
        have = 0  # a different file under the same name, start over
    os.makedirs(os.path.dirname(path), exist_ok=True)
    request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(request, timeout=30) as response:
        resumed = response.status == 206
        with open(path, "ab" if resumed else "wb") as f:
            shutil.copyfileobj(response, f, CHUNK)
    if os.path.getsize(path) != size:
        raise OSError(f"{path} is {os.path.getsize(path)} bytes, expected {size}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kart", help="master API, e.g. http://100.93.187.32:8000")
    parser.add_argument("--out", default="bags", help="local folder for runs")
    parser.add_argument("--since", type=float, help="unix time; defaults to the cursor saved in --out")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cursor_path = os.path.join(args.out, ".cursor")
    since = args.since
    if since is None:
        since = float(open(cursor_path).read()) if os.path.exists(cursor_path) else 0.0

    listing = get_json(f"{args.kart}/runs?since={since}")
    for run in listing["runs"]:
        for file in run["files"]:
            url = f"{args.kart}/runs/{run['name']}/files/{urllib.parse.quote(file['path'])}"
            if download(url, os.path.join(args.out, run["name"], file["path"]), file["size"]):
                print(f"{run['name']}/{file['path']}")
        if not run["recording"] and not run["confirmed"]:
            request = urllib.request.Request(f"{args.kart}/runs/{run['name']}/confirm", method="POST")
            urllib.request.urlopen(request, timeout=10).close()
            print(f"confirmed {run['name']}")

    # Saved only after every file arrived, so a failed pull is retried from the same cursor
    with open(cursor_path, "w") as f:
        f.write(str(listing["now"]))


if __name__ == "__main__":
    main()
