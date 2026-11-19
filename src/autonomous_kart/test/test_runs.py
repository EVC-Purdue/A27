"""Checks which run files are offered while recording and that cleanup only deletes confirmed runs.

https://github.com/EVC-Purdue/A27/issues/20
"""

import os
import time

from autonomous_kart import runs


def make_run(bag_dir, name, splits=3, recording=False, confirmed=False):
    run_dir = os.path.join(bag_dir, name)
    os.makedirs(os.path.join(run_dir, "bag"))
    for i in range(splits):
        with open(os.path.join(run_dir, "bag", f"bag_{i}.mcap"), "w") as f:
            f.write("x" * (i + 1))
    for marker, flag in ((runs.RECORDING, recording), (runs.CONFIRMED, confirmed), ("manifest.json", True)):
        if flag:
            open(os.path.join(run_dir, marker), "w").close()
    return run_dir


def test_recording_run_offers_only_closed_splits(tmp_path):
    run_dir = make_run(tmp_path, "run_a", splits=12, recording=True)
    assert runs.closed_files(run_dir) == sorted(f"bag/bag_{i}.mcap" for i in range(11))


def test_stopped_run_offers_everything(tmp_path):
    run_dir = make_run(tmp_path, "run_a", confirmed=True)
    assert runs.closed_files(run_dir) == ["bag/bag_0.mcap", "bag/bag_1.mcap", "bag/bag_2.mcap", "manifest.json"]


def test_list_runs_since(tmp_path):
    make_run(tmp_path, "run_a")
    old = time.time() - 100
    for root, _, names in os.walk(tmp_path / "run_a"):
        for n in names:
            os.utime(os.path.join(root, n), (old, old))
    make_run(tmp_path, "run_b", recording=True)
    assert [r["name"] for r in runs.list_runs(tmp_path)] == ["run_a", "run_b"]
    listed = runs.list_runs(tmp_path, since=old + 1)
    assert [r["name"] for r in listed] == ["run_b"]
    assert listed[0]["recording"] and [f["size"] for f in listed[0]["files"]] == [1, 2]


def test_cleanup_deletes_only_confirmed(tmp_path):
    make_run(tmp_path, "run_a")
    make_run(tmp_path, "run_b", confirmed=True)
    make_run(tmp_path, "run_c", confirmed=True, recording=True)
    make_run(tmp_path, "run_d", confirmed=True)
    assert runs.cleanup(tmp_path, min_free_bytes=1, free_bytes=lambda: 0) == ["run_b", "run_d"]
    assert runs.run_names(tmp_path) == ["run_a", "run_c"]


def test_cleanup_stops_once_there_is_room(tmp_path):
    make_run(tmp_path, "run_a", confirmed=True)
    make_run(tmp_path, "run_b", confirmed=True)
    free = iter([0, 10])
    assert runs.cleanup(tmp_path, min_free_bytes=5, free_bytes=lambda: next(free)) == ["run_a"]
