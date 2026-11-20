"""Puts the package source on sys.path so ROS-free tests run with plain pytest, and serves pathfinder.yaml params.

https://github.com/EVC-Purdue/A27/issues/11
"""

import os
import sys

import pytest
import yaml

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)


@pytest.fixture
def pathfinder_params():
    with open(os.path.join(_PKG_ROOT, "autonomous_kart", "params", "pathfinder.yaml")) as f:
        return yaml.safe_load(f)["pathfinder_node"]["ros__parameters"]
