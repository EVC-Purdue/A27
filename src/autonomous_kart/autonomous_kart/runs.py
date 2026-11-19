"""Run folders on disk: which files are closed, pull confirmation and disk cleanup.

https://github.com/EVC-Purdue/A27/issues/20
"""

import os
import re
import shutil

RECORDING = "recording"  # marker present while the recorder writes this run
CONFIRMED = "confirmed"  # marker a client sets once it has pulled the whole run


def run_names(bag_dir):
    if not os.path.isdir(bag_dir):
        return []
    return sorted(n for n in os.listdir(bag_dir) if n.startswith("run_"))


def is_recording(run_dir):
    return os.path.exists(os.path.join(run_dir, RECORDING))


def is_confirmed(run_dir):
    return os.path.exists(os.path.join(run_dir, CONFIRMED))


def split_index(name):
    match = re.search(r"_(\d+)\.mcap$", name)
    return int(match.group(1)) if match else -1


def closed_files(run_dir):
    # While recording, only the splits before the newest are final; the rest is rewritten until the run stops
    files = []
    for root, _, names in os.walk(run_dir):
        for name in names:
            if name in (RECORDING, CONFIRMED):
                continue
            files.append(os.path.relpath(os.path.join(root, name), run_dir))
    if not is_recording(run_dir):
        return sorted(files)
    splits = [f for f in files if f.endswith(".mcap")]
    newest = max(splits, key=split_index, default=None)
    return sorted(f for f in splits if f != newest)


def updated_at(run_dir):
    return max(
        (os.path.getmtime(os.path.join(root, n)) for root, _, names in os.walk(run_dir) for n in names), default=0.0
    )


def list_runs(bag_dir, since=0.0):
    runs = []
    for name in run_names(bag_dir):
        run_dir = os.path.join(bag_dir, name)
        if updated_at(run_dir) < since:
            continue
        files = [{"path": f, "size": os.path.getsize(os.path.join(run_dir, f))} for f in closed_files(run_dir)]
        runs.append(
            {
                "name": name,
                "recording": is_recording(run_dir),
                "confirmed": is_confirmed(run_dir),
                "files": files,
            }
        )
    return runs


def cleanup(bag_dir, min_free_bytes, free_bytes=None):
    # Deletes the oldest confirmed runs until there is room; unconfirmed runs are never deleted
    free_bytes = free_bytes or (lambda: shutil.disk_usage(bag_dir).free)
    removed = []
    for name in run_names(bag_dir):
        if free_bytes() >= min_free_bytes:
            break
        run_dir = os.path.join(bag_dir, name)
        if is_confirmed(run_dir) and not is_recording(run_dir):
            shutil.rmtree(run_dir)
            removed.append(name)
    return removed
