"""Puts the package source on sys.path so ROS-free tests run with plain pytest.

https://github.com/EVC-Purdue/A27/issues/11
"""
import os
import sys

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
