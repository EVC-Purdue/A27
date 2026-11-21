"""Geometry on a line of LinePoints, shared by the controllers.

https://github.com/EVC-Purdue/A27/issues/13
"""

import math


def distance(a, b):
    return math.hypot(b.x_m - a.x_m, b.y_m - a.y_m)


def is_closed(line, gap_m, min_length_m):
    # A loop has its ends close together and is long, so a short or hairpin open line is not one
    length_m = sum(distance(a, b) for a, b in zip(line, line[1:], strict=False))
    return distance(line[0], line[-1]) < gap_m and length_m > min_length_m


def next_index(i, n, closed):
    # The point after i; a closed line wraps to the start, an open one ends with None
    if i + 1 < n:
        return i + 1
    return 0 if closed else None


def previous_index(i, n, closed):
    if i > 0:
        return i - 1
    return n - 1 if closed else None


def heading_at(line, i):
    # Direction of travel at point i, from the segment that leaves it, or the one that enters it at the end
    if i + 1 < len(line):
        a, b = line[i], line[i + 1]
    else:
        a, b = line[i - 1], line[i]
    return math.atan2(b.y_m - a.y_m, b.x_m - a.x_m)


def index_ahead(line, start, distance_m, closed):
    # The first point at least distance_m along the line from start, or the last point an open line reaches
    travelled_m = 0.0
    i = start
    while travelled_m < distance_m:
        j = next_index(i, len(line), closed)
        if j is None or j == start:
            break
        travelled_m += distance(line[i], line[j])
        i = j
    return i


def distance_to_segment(x, y, a, b):
    ux, uy = b.x_m - a.x_m, b.y_m - a.y_m
    length_sq = ux * ux + uy * uy
    t = 0.0 if length_sq == 0.0 else clamp(((x - a.x_m) * ux + (y - a.y_m) * uy) / length_sq, 0.0, 1.0)
    return math.hypot(a.x_m + t * ux - x, a.y_m + t * uy - y)


def wrap_angle(angle_rad):
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def clamp(value, low, high):
    return max(low, min(value, high))
