"""A path of points for the kart to follow: the racing line, or a rejoin curve back to it.

https://github.com/EVC-Purdue/A27/issues/13
"""

import math


class Path:
    def __init__(self, points, is_loop=False):
        self.points = points
        self.is_loop = is_loop
        self.speeds_mps = [p.target_speed_mps for p in points]  # speed to drive at each point

    def next(self, i):
        # A loop wraps from its last point to its first
        if i + 1 < len(self.points):
            return i + 1
        return 0 if self.is_loop else None

    def previous(self, i):
        if i > 0:
            return i - 1
        return len(self.points) - 1 if self.is_loop else None

    def heading(self, i):
        # Direction of travel at point i
        j = self.next(i)
        if j is None:
            i, j = i - 1, i
        a, b = self.points[i], self.points[j]
        return math.atan2(b.y_m - a.y_m, b.x_m - a.x_m)

    def distance_to(self, i, x, y):
        return math.hypot(self.points[i].x_m - x, self.points[i].y_m - y)

    def indices_from(self, start, count):
        # Up to count indices, following the path forward from start
        indices = []
        i = start
        while i is not None and len(indices) < count:
            indices.append(i)
            i = self.next(i)
        return indices

    def nearest(self, indices, x, y):
        return min(indices, key=lambda i: self.distance_to(i, x, y))

    def index_ahead(self, start, distance_m):
        # Index of the point distance_m along the path from start; an open path stops at its last point
        travelled_m = 0.0
        i = start
        while travelled_m < distance_m:
            j = self.next(i)
            if j is None or j == start:
                break
            travelled_m += distance(self.points[i], self.points[j])
            i = j
        return i

    def cross_track_error(self, i, x, y):
        # Distance from (x, y) to the path, given that point i is the nearest point
        neighbours = [j for j in (self.previous(i), self.next(i)) if j is not None]
        return min(distance_to_segment(x, y, self.points[i], self.points[j]) for j in neighbours)


def is_loop(points, gap_m, min_length_m):
    # The ends meet, and the path is long enough that a short out-and-back line does not count
    length_m = sum(distance(a, b) for a, b in zip(points, points[1:], strict=False))
    return distance(points[0], points[-1]) < gap_m and length_m > min_length_m


def distance(a, b):
    return math.hypot(b.x_m - a.x_m, b.y_m - a.y_m)


def distance_to_segment(x, y, a, b):
    ux, uy = b.x_m - a.x_m, b.y_m - a.y_m
    length_sq = ux * ux + uy * uy
    if length_sq == 0.0:
        return math.hypot(a.x_m - x, a.y_m - y)
    # t is how far along the segment the closest point sits, 0 at a and 1 at b
    t = clamp(((x - a.x_m) * ux + (y - a.y_m) * uy) / length_sq, 0.0, 1.0)
    return math.hypot(a.x_m + t * ux - x, a.y_m + t * uy - y)


def angle_between(a_rad, b_rad):
    difference = a_rad - b_rad
    return abs(math.atan2(math.sin(difference), math.cos(difference)))


def clamp(value, low, high):
    return max(low, min(value, high))
