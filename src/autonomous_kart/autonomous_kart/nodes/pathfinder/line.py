"""Loads a racing line CSV (s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps2) into controller line points.

https://github.com/EVC-Purdue/A27/issues/13
"""

import csv

from autonomous_kart.nodes.pathfinder.controllers.base import LinePoint


def load_line(path):
    with open(path, newline="") as f:
        return tuple(
            LinePoint(float(row["x_m"]), float(row["y_m"]), float(row["kappa_radpm"]), float(row["vx_mps"]))
            for row in csv.DictReader(f)
        )
