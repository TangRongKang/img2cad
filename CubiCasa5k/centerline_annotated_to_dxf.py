#!/usr/bin/env python3
"""
centerline_annotated_to_dxf.py

Convert a color-annotated floor plan image to DXF using a centerline-first wall
model.

Compared with color_annotated_to_dxf.py, this script does not emit the blue wall
mask contour directly. It:
  1. extracts the blue wall mask,
  2. thins it to a one-pixel skeleton,
  3. detects/merges dominant wall centerlines,
  4. snaps near-horizontal/vertical walls to exact axes and keeps real diagonals,
  5. rebuilds wall rectangles from centerlines and measured wall thickness.

The output is intentionally simpler but cleaner: fewer tiny contour notches and
less micro-tilt in walls that should be straight.

Usage:
    conda run -n agent python centerline_annotated_to_dxf.py test/annotated/4_annotated.png --width-mm 13700
    conda run -n agent python centerline_annotated_to_dxf.py input.png --scale 5.84 -o output.dxf
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import ezdxf
import numpy as np


BLUE_LO = np.array([95, 100, 80])
BLUE_HI = np.array([145, 255, 255])
RED_LO1 = np.array([0, 100, 80])
RED_HI1 = np.array([12, 255, 255])
RED_LO2 = np.array([165, 100, 80])
RED_HI2 = np.array([180, 255, 255])
GREEN_LO = np.array([40, 80, 80])
GREEN_HI = np.array([85, 255, 255])


def morph_clean(mask, close_k=5, open_k=3, close_iter=2, open_iter=1):
    kc = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
    ko = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc, iterations=close_iter)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, ko, iterations=open_iter)


def px_to_mm(pts, scale, x0, y0):
    return [((float(x) - x0) * scale, (y0 - float(y)) * scale) for x, y in pts]


def simplify(cnt, eps_frac):
    eps = eps_frac * cv2.arcLength(cnt, True)
    return cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)


def median_wall_thickness(blue_mask):
    h, w = blue_mask.shape[:2]
    runs = []

    def collect(line):
        n = 0
        for v in line:
            if v > 0:
                n += 1
            elif n:
                if 4 < n < 180:
                    runs.append(n)
                n = 0
        if 4 < n < 180:
            runs.append(n)

    for y in range(h // 8, 7 * h // 8, max(h // 24, 1)):
        collect(blue_mask[y])
    for x in range(w // 8, 7 * w // 8, max(w // 24, 1)):
        collect(blue_mask[:, x])
    return int(np.median(runs)) if runs else 20


def zhang_suen_thinning(mask):
    """Pure OpenCV/Numpy Zhang-Suen thinning, returns uint8 0/255 skeleton."""
    img = (mask > 0).astype(np.uint8)
    img[[0, -1], :] = 0
    img[:, [0, -1]] = 0

    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p2 = np.roll(img, -1, axis=0)
            p3 = np.roll(np.roll(img, -1, axis=0), 1, axis=1)
            p4 = np.roll(img, 1, axis=1)
            p5 = np.roll(np.roll(img, 1, axis=0), 1, axis=1)
            p6 = np.roll(img, 1, axis=0)
            p7 = np.roll(np.roll(img, 1, axis=0), -1, axis=1)
            p8 = np.roll(img, -1, axis=1)
            p9 = np.roll(np.roll(img, -1, axis=0), -1, axis=1)

            neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )
            if step == 0:
                cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (img == 1) & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1) & cond
            remove[[0, -1], :] = False
            remove[:, [0, -1]] = False
            if np.any(remove):
                img[remove] = 0
                changed = True
    return (img * 255).astype(np.uint8)


def angle_dist(a, b):
    return abs(((a - b + 90.0) % 180.0) - 90.0)


def line_angle(x1, y1, x2, y2):
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0


def snap_direction(theta, axis_tol=10.0, diag_tol=5.0):
    for target in (0.0, 90.0):
        if angle_dist(theta, target) <= axis_tol:
            return target
    for target in (45.0, 135.0):
        if angle_dist(theta, target) <= diag_tol:
            return target
    return theta


def cluster_directions(raw_lines, min_weight):
    clusters = []
    for x1, y1, x2, y2, length in raw_lines:
        theta = line_angle(x1, y1, x2, y2)
        best_i = None
        best_d = 12.0
        for i, cl in enumerate(clusters):
            d = angle_dist(theta, cl["theta"])
            if d < best_d:
                best_i = i
                best_d = d
        if best_i is None:
            clusters.append({"angles": [theta], "weights": [length], "theta": theta})
        else:
            cl = clusters[best_i]
            cl["angles"].append(theta)
            cl["weights"].append(length)
            cl["theta"] = mean_axial_angle(cl["angles"], cl["weights"])

    out = []
    total = sum(length for *_, length in raw_lines)
    keep_weight = max(min_weight, total * 0.025)
    for cl in clusters:
        weight = sum(cl["weights"])
        if weight >= keep_weight:
            out.append(snap_direction(mean_axial_angle(cl["angles"], cl["weights"])))

    for target in (0.0, 90.0):
        if not any(angle_dist(target, t) < 1e-3 for t in out):
            near = [l for l in raw_lines if angle_dist(line_angle(*l[:4]), target) <= 12.0]
            if sum(l[4] for l in near) >= min_weight:
                out.append(target)
    return sorted(set(round(t, 3) for t in out), key=lambda a: (angle_dist(a, 0), a))


def mean_axial_angle(angles, weights):
    sx = 0.0
    sy = 0.0
    for a, w in zip(angles, weights):
        r = math.radians(2.0 * a)
        sx += math.cos(r) * w
        sy += math.sin(r) * w
    return (math.degrees(math.atan2(sy, sx)) / 2.0) % 180.0


def direction_vectors(theta):
    r = math.radians(theta)
    d = np.array([math.cos(r), math.sin(r)], dtype=float)
    n = np.array([-d[1], d[0]], dtype=float)
    return d, n


def cluster_scalar(values, tol):
    if not values:
        return []
    values = sorted(values, key=lambda x: x[0])
    groups = [[values[0]]]
    for item in values[1:]:
        if abs(item[0] - groups[-1][-1][0]) <= tol:
            groups[-1].append(item)
        else:
            groups.append([item])
    return groups


def merge_intervals(intervals, max_gap):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for a, b in intervals[1:]:
        if a <= merged[-1][1] + max_gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def extract_hough_lines(skeleton, wall_t):
    min_len = max(18, int(round(wall_t * 1.1)))
    max_gap = max(8, int(round(wall_t * 1.7)))
    threshold = max(12, int(round(wall_t * 0.65)))
    lines = cv2.HoughLinesP(
        skeleton,
        rho=1,
        theta=np.pi / 180.0,
        threshold=threshold,
        minLineLength=min_len,
        maxLineGap=max_gap,
    )
    raw = []
    if lines is None:
        return raw
    for l in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, l)
        length = math.hypot(x2 - x1, y2 - y1)
        if length >= min_len:
            raw.append((x1, y1, x2, y2, length))
    return raw


def build_centerlines_from_hough(raw_lines, wall_t):
    if not raw_lines:
        return []
    dirs = cluster_directions(raw_lines, min_weight=max(80.0, wall_t * 5.0))
    if not dirs:
        return []

    assigned = {theta: [] for theta in dirs}
    for line in raw_lines:
        theta = line_angle(*line[:4])
        best = min(dirs, key=lambda t: angle_dist(theta, t))
        if angle_dist(theta, best) <= 16.0:
            assigned[best].append(line)

    c_tol = max(5.0, wall_t * 0.55)
    gap_tol = max(18.0, wall_t * 2.4)
    min_seg = max(18.0, wall_t * 1.1)
    segments = []

    for theta, lines in assigned.items():
        d, n = direction_vectors(theta)
        projected = []
        for x1, y1, x2, y2, length in lines:
            p1 = np.array([x1, y1], dtype=float)
            p2 = np.array([x2, y2], dtype=float)
            mid = (p1 + p2) / 2.0
            c = float(np.dot(n, mid))
            t1 = float(np.dot(d, p1))
            t2 = float(np.dot(d, p2))
            if t2 < t1:
                t1, t2 = t2, t1
            projected.append((c, t1, t2, length))

        for group in cluster_scalar(projected, c_tol):
            weight = sum(g[3] for g in group)
            c = sum(g[0] * g[3] for g in group) / max(weight, 1e-9)
            intervals = merge_intervals([(g[1], g[2]) for g in group], gap_tol)
            for t1, t2 in intervals:
                if t2 - t1 < min_seg:
                    continue
                p1 = d * t1 + n * c
                p2 = d * t2 + n * c
                segments.append({"theta": theta, "p1": p1, "p2": p2})

    snap_segment_junctions(segments, wall_t)
    return merge_near_duplicate_segments(segments, wall_t)


def segment_params(seg):
    theta = seg["theta"]
    d, n = direction_vectors(theta)
    t1 = float(np.dot(d, seg["p1"]))
    t2 = float(np.dot(d, seg["p2"]))
    if t2 < t1:
        t1, t2 = t2, t1
    c = float(np.dot(n, (seg["p1"] + seg["p2"]) / 2.0))
    return d, n, c, t1, t2


def line_intersection(theta1, p1, theta2, p2):
    d1, _ = direction_vectors(theta1)
    d2, _ = direction_vectors(theta2)
    det = d2[0] * d1[1] - d1[0] * d2[1]
    if abs(det) < 1e-6:
        return None
    r = p2 - p1
    t = (d2[0] * r[1] - r[0] * d2[1]) / det
    return p1 + d1 * t


def snap_segment_junctions(segments, wall_t):
    tol = max(12.0, wall_t * 2.0)
    if len(segments) < 2:
        return
    for i, a in enumerate(segments):
        for end_name in ("p1", "p2"):
            endpoint = a[end_name]
            best = None
            best_d = tol
            for j, b in enumerate(segments):
                if i == j or angle_dist(a["theta"], b["theta"]) < 8.0:
                    continue
                p = line_intersection(a["theta"], endpoint, b["theta"], b["p1"])
                if p is None:
                    continue
                d_b, _, _, bt1, bt2 = segment_params(b)
                tp = float(np.dot(d_b, p))
                if not (bt1 - tol <= tp <= bt2 + tol):
                    continue
                dist = float(np.linalg.norm(p - endpoint))
                if dist < best_d:
                    best = p
                    best_d = dist
            if best is not None:
                a[end_name] = best


def merge_near_duplicate_segments(segments, wall_t):
    by_theta = {}
    for seg in segments:
        by_theta.setdefault(seg["theta"], []).append(seg)

    out = []
    c_tol = max(4.0, wall_t * 0.45)
    gap_tol = max(10.0, wall_t * 1.5)
    for theta, group in by_theta.items():
        d, n = direction_vectors(theta)
        vals = []
        for seg in group:
            c = float(np.dot(n, (seg["p1"] + seg["p2"]) / 2.0))
            t1 = float(np.dot(d, seg["p1"]))
            t2 = float(np.dot(d, seg["p2"]))
            if t2 < t1:
                t1, t2 = t2, t1
            vals.append((c, t1, t2, seg))
        for cgroup in cluster_scalar(vals, c_tol):
            weight = sum(max(g[2] - g[1], 1.0) for g in cgroup)
            c = sum(g[0] * max(g[2] - g[1], 1.0) for g in cgroup) / max(weight, 1e-9)
            for t1, t2 in merge_intervals([(g[1], g[2]) for g in cgroup], gap_tol):
                out.append({"theta": theta, "p1": d * t1 + n * c, "p2": d * t2 + n * c})
    return out


def measure_width_at(mask, p, normal, wall_t):
    h, w = mask.shape[:2]
    radius = int(max(8, wall_t * 2.5))

    def inside_at(offset):
        q = p + normal * offset
        x = int(round(q[0]))
        y = int(round(q[1]))
        return 0 <= x < w and 0 <= y < h and mask[y, x] > 0

    center = 0
    if not inside_at(0):
        found = None
        for k in range(1, radius + 1):
            if inside_at(k):
                found = k
                break
            if inside_at(-k):
                found = -k
                break
        if found is None:
            return None
        center = found

    lo = center
    while lo - 1 >= -radius and inside_at(lo - 1):
        lo -= 1
    hi = center
    while hi + 1 <= radius and inside_at(hi + 1):
        hi += 1
    width = hi - lo + 1
    if 0.35 * wall_t <= width <= 3.5 * wall_t:
        return width
    return None


def segment_wall_width(mask, seg, wall_t):
    d, n, _, t1, t2 = segment_params(seg)
    samples = []
    count = max(5, min(15, int((t2 - t1) / max(wall_t * 3.0, 1.0))))
    for t in np.linspace(t1 + wall_t * 0.3, t2 - wall_t * 0.3, count):
        p = d * t + n * float(np.dot(n, (seg["p1"] + seg["p2"]) / 2.0))
        width = measure_width_at(mask, p, n, wall_t)
        if width is not None:
            samples.append(width)
    if not samples:
        return float(wall_t)
    return float(np.median(samples))


def centerlines_to_wall_polys(blue_mask, segments, wall_t):
    polys = []
    for seg in segments:
        d, n, _, _, _ = segment_params(seg)
        width = segment_wall_width(blue_mask, seg, wall_t)
        half = max(width / 2.0, 2.0)
        p1 = seg["p1"]
        p2 = seg["p2"]
        poly = [p1 + n * half, p2 + n * half, p2 - n * half, p1 - n * half]
        polys.append([(float(p[0]), float(p[1])) for p in poly])
    return polys


def skeleton_degrees(skeleton):
    sk = (skeleton > 0).astype(np.uint8)
    deg = np.zeros_like(sk, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            deg += np.roll(np.roll(sk, dy, axis=0), dx, axis=1)
    deg[sk == 0] = 0
    return deg


def neighbor_pixels(sk, y, x):
    h, w = sk.shape
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            yy = y + dy
            xx = x + dx
            if 0 <= yy < h and 0 <= xx < w and sk[yy, xx]:
                out.append((yy, xx))
    return out


def trace_skeleton_paths(skeleton, min_len=8):
    """Return centerline paths as [(x, y), ...] traced between graph nodes."""
    sk = skeleton > 0
    deg = skeleton_degrees(skeleton)
    node = sk & (deg != 2)
    visited = set()
    paths = []

    def edge_key(a, b):
        return tuple(sorted((a, b)))

    def walk(start, nxt):
        path = [start, nxt]
        visited.add(edge_key(start, nxt))
        prev = start
        cur = nxt
        while not node[cur]:
            nbs = [p for p in neighbor_pixels(sk, cur[0], cur[1]) if p != prev]
            if not nbs:
                break
            best = None
            for nb in nbs:
                if edge_key(cur, nb) not in visited:
                    best = nb
                    break
            if best is None:
                break
            visited.add(edge_key(cur, best))
            prev, cur = cur, best
            path.append(cur)
        return [(float(x), float(y)) for y, x in path]

    for start in map(tuple, np.argwhere(node)):
        for nxt in neighbor_pixels(sk, start[0], start[1]):
            if edge_key(start, nxt) not in visited:
                p = walk(start, nxt)
                if path_length(p) >= min_len:
                    paths.append(p)

    # Closed loops have no degree!=2 nodes; trace any remaining unvisited cycle.
    for start in map(tuple, np.argwhere(sk)):
        nbs = neighbor_pixels(sk, start[0], start[1])
        unused = [nb for nb in nbs if edge_key(start, nb) not in visited]
        if not unused:
            continue
        path = [(float(start[1]), float(start[0]))]
        prev = start
        cur = unused[0]
        visited.add(edge_key(prev, cur))
        while cur != start:
            path.append((float(cur[1]), float(cur[0])))
            nbs = [p for p in neighbor_pixels(sk, cur[0], cur[1]) if p != prev]
            nxt = None
            for nb in nbs:
                if edge_key(cur, nb) not in visited:
                    nxt = nb
                    break
            if nxt is None:
                break
            visited.add(edge_key(cur, nxt))
            prev, cur = cur, nxt
        if path_length(path) >= min_len:
            paths.append(path)
    return paths


def path_length(path):
    if len(path) < 2:
        return 0.0
    pts = np.asarray(path, dtype=float)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def max_line_deviation(path):
    if len(path) < 3:
        return 0.0
    pts = np.asarray(path, dtype=float)
    a = pts[0]
    b = pts[-1]
    ab = b - a
    norm = np.linalg.norm(ab)
    if norm < 1e-6:
        return float("inf")
    rel = pts - a
    cross = ab[0] * rel[:, 1] - ab[1] * rel[:, 0]
    return float(np.abs(cross / norm).max())


def simplify_open_path(path, eps):
    pts = np.asarray(path, dtype=np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(pts, eps, False).reshape(-1, 2)
    if len(approx) == 1:
        approx = np.asarray([path[0], path[-1]], dtype=np.float32)
    return [(float(x), float(y)) for x, y in approx]


def regularize_centerline_path(path, wall_t):
    """Straighten line-like paths; keep real curved paths as simplified polylines."""
    if len(path) < 2:
        return path
    pts = np.asarray(path, dtype=float)
    chord = float(np.linalg.norm(pts[-1] - pts[0]))
    length = path_length(path)
    if chord < max(6.0, wall_t * 0.6):
        return []
    deviation = max_line_deviation(path)
    is_straight = deviation <= max(2.2, wall_t * 0.28) and length <= chord * 1.08
    if is_straight:
        a = pts[0]
        b = pts[-1]
        theta = snap_direction(line_angle(a[0], a[1], b[0], b[1]), axis_tol=12.0, diag_tol=7.0)
        d, n = direction_vectors(theta)
        c = float(np.dot(n, pts.mean(axis=0)))
        t1 = float(np.dot(d, a))
        t2 = float(np.dot(d, b))
        if t2 < t1:
            t1, t2 = t2, t1
        p1 = d * t1 + n * c
        p2 = d * t2 + n * c
        return [(float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))]
    return simplify_open_path(path, max(1.6, wall_t * 0.22))


def snap_path_endpoints(paths, wall_t):
    endpoints = []
    for pi, p in enumerate(paths):
        if len(p) < 2:
            continue
        endpoints.append((np.asarray(p[0], dtype=float), pi, 0))
        endpoints.append((np.asarray(p[-1], dtype=float), pi, -1))
    used = [False] * len(endpoints)
    tol = max(4.0, wall_t * 0.65)
    for i, (pt, _, _) in enumerate(endpoints):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        for j in range(i + 1, len(endpoints)):
            if not used[j] and np.linalg.norm(endpoints[j][0] - pt) <= tol:
                used[j] = True
                group.append(j)
        if len(group) < 2:
            continue
        rep = sum((endpoints[g][0] for g in group), np.zeros(2, dtype=float)) / len(group)
        for g in group:
            _, pi, ei = endpoints[g]
            paths[pi][ei] = (float(rep[0]), float(rep[1]))


def bridge_near_path_endpoints(paths, wall_t):
    endpoints = []
    for pi, p in enumerate(paths):
        if len(p) < 2:
            continue
        endpoints.append((np.asarray(p[0], dtype=float), pi, 0))
        endpoints.append((np.asarray(p[-1], dtype=float), pi, -1))

    tol = max(6.0, wall_t * 1.15)
    for i, (pt_i, pi, ei) in enumerate(endpoints):
        best = None
        best_d = tol
        for j, (pt_j, pj, ej) in enumerate(endpoints):
            if i == j or pi == pj:
                continue
            dist = float(np.linalg.norm(pt_i - pt_j))
            if dist >= best_d:
                continue
            # Only bridge small raster breaks. Door/window gaps are generally
            # larger than one wall thickness, so this avoids closing openings.
            best = (pj, ej, (pt_i + pt_j) / 2.0)
            best_d = dist
        if best is not None:
            pj, ej, rep = best
            paths[pi][ei] = (float(rep[0]), float(rep[1]))
            paths[pj][ej] = (float(rep[0]), float(rep[1]))


def path_axis_kind(path, tol=15.0):
    if len(path) < 2:
        return None
    a = np.asarray(path[0], dtype=float)
    b = np.asarray(path[-1], dtype=float)
    theta = line_angle(a[0], a[1], b[0], b[1])
    if angle_dist(theta, 0.0) <= tol:
        return "h"
    if angle_dist(theta, 90.0) <= tol:
        return "v"
    return None


def weighted_value_clusters(items, tol):
    """Cluster [(value, weight), ...] and return value->representative mapping."""
    if not items:
        return {}
    ordered = sorted(items, key=lambda item: item[0])
    groups = [[ordered[0]]]
    for item in ordered[1:]:
        if abs(item[0] - groups[-1][-1][0]) <= tol:
            groups[-1].append(item)
        else:
            groups.append([item])
    snap = {}
    for group in groups:
        weight = sum(max(w, 1e-6) for _, w in group)
        rep = sum(v * max(w, 1e-6) for v, w in group) / weight
        for v, _ in group:
            snap[v] = rep
    return snap


def axis_path_info(path):
    kind = path_axis_kind(path)
    if kind is None:
        return None
    pts = np.asarray(path, dtype=float)
    if kind == "h":
        c = float(np.mean(pts[:, 1]))
        t1 = float(np.min(pts[:, 0]))
        t2 = float(np.max(pts[:, 0]))
    else:
        c = float(np.mean(pts[:, 0]))
        t1 = float(np.min(pts[:, 1]))
        t2 = float(np.max(pts[:, 1]))
    return kind, c, t1, t2


def snap_axis_centerline_grid(paths, wall_t):
    """Force all near-horizontal/vertical centerlines onto a shared H/V grid."""
    infos = []
    h_vals = []
    v_vals = []
    for i, path in enumerate(paths):
        info = axis_path_info(path)
        if info is None:
            continue
        kind, c, _, _ = info
        length = max(path_length(path), 1.0)
        infos.append((i, kind, c, length))
        if kind == "h":
            h_vals.append((c, length))
        else:
            v_vals.append((c, length))

    tol = max(2.0, wall_t * 0.45)
    h_snap = weighted_value_clusters(h_vals, tol)
    v_snap = weighted_value_clusters(v_vals, tol)

    for i, kind, c, _ in infos:
        if kind == "h":
            y = h_snap.get(c, c)
            paths[i] = [(float(x), float(y)) for x, _ in paths[i]]
        else:
            x = v_snap.get(c, c)
            paths[i] = [(float(x), float(y)) for _, y in paths[i]]

    # Rebuild ranges after coordinate snapping, then make T/L endpoints exact.
    h_lines = []
    v_lines = []
    for i, path in enumerate(paths):
        info = axis_path_info(path)
        if info is None:
            continue
        kind, c, t1, t2 = info
        if kind == "h":
            h_lines.append((i, c, t1, t2))
        else:
            v_lines.append((i, c, t1, t2))

    jtol = max(3.0, wall_t * 0.75)
    for i, path in enumerate(paths):
        kind = path_axis_kind(path)
        if kind not in ("h", "v") or len(path) < 2:
            continue
        for ei in (0, -1):
            x, y = path[ei]
            if kind == "h":
                candidates = [
                    (abs(x - vx), vx)
                    for _, vx, vy1, vy2 in v_lines
                    if abs(x - vx) <= jtol and vy1 - jtol <= y <= vy2 + jtol
                ]
                if candidates:
                    _, vx = min(candidates, key=lambda item: item[0])
                    path[ei] = (float(vx), float(y))
            else:
                candidates = [
                    (abs(y - hy), hy)
                    for _, hy, hx1, hx2 in h_lines
                    if abs(y - hy) <= jtol and hx1 - jtol <= x <= hx2 + jtol
                ]
                if candidates:
                    _, hy = min(candidates, key=lambda item: item[0])
                    path[ei] = (float(x), float(hy))


def paths_to_wall_mask(shape, paths, wall_t):
    mask = np.zeros(shape, dtype=np.uint8)
    thickness = max(3, int(round(wall_t)))
    radius = max(2, thickness // 2)
    for path in paths:
        if len(path) < 2:
            continue
        arr = np.round(np.asarray(path, dtype=float)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(mask, [arr], False, 255, thickness=thickness, lineType=cv2.LINE_8)
        for x, y in arr.reshape(-1, 2):
            cv2.circle(mask, (int(x), int(y)), radius, 255, thickness=-1, lineType=cv2.LINE_8)
    k = max(3, int(round(wall_t * 0.35)) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return mask


def contour_polys_from_wall_mask(mask, wall_t, min_area=80):
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    eps = max(1.2, wall_t * 0.10)
    for cnt in cnts:
        if cv2.contourArea(cnt) < max(min_area, wall_t * wall_t * 0.5):
            continue
        approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
        if len(approx) >= 3:
            polys.append([(float(x), float(y)) for x, y in approx])
    return polys


def centerline_dicts(paths):
    out = []
    for p in paths:
        if len(p) >= 2:
            out.append({"points": [tuple(pt) for pt in p], "p1": np.asarray(p[0]), "p2": np.asarray(p[-1])})
    return out


def trace_walls_centerline(blue_mask, wall_t):
    close = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, wall_t // 3 * 2 + 1),) * 2)
    clean = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, close, iterations=1)
    clean = morph_clean(clean, close_k=3, open_k=3, close_iter=1)
    skeleton = zhang_suen_thinning(clean)
    raw_lines = extract_hough_lines(skeleton, wall_t)
    raw_paths = trace_skeleton_paths(skeleton, min_len=max(8, int(round(wall_t * 0.8))))
    paths = [p for p in (regularize_centerline_path(path, wall_t) for path in raw_paths) if len(p) >= 2]
    snap_path_endpoints(paths, wall_t)
    bridge_near_path_endpoints(paths, wall_t)
    snap_axis_centerline_grid(paths, wall_t)
    rebuilt = paths_to_wall_mask(clean.shape, paths, wall_t)
    # Keep the reconstructed walls close to the annotation while preserving clean
    # centerline joins. A light union retains curved/odd walls that skeleton strokes
    # may under-fill, then a close removes small raster seams.
    keep = cv2.bitwise_or(rebuilt, cv2.bitwise_and(clean, cv2.dilate(rebuilt, np.ones((max(3, wall_t | 1),) * 2, np.uint8))))
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    wall_polys = contour_polys_from_wall_mask(keep, wall_t)
    return wall_polys, centerline_dicts(paths), skeleton, raw_lines


def contour_touches_structure(cnt, struct_mask, wall_t):
    if struct_mask is None:
        return True
    h, w = struct_mask.shape[:2]
    cmask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(cmask, [cnt], -1, 255, thickness=cv2.FILLED)
    k = max(3, int(round(wall_t * 1.2)) | 1)
    cmask = cv2.dilate(cmask, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    return bool(cv2.bitwise_and(cmask, struct_mask).any())


def trace_windows(green_mask, struct_mask=None, wall_t=20, min_area=120):
    windows = []
    cnts, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        if not contour_touches_structure(cnt, struct_mask, wall_t):
            continue
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        (cx, cy), (rw, rh), angle = rect
        if rw <= 1 or rh <= 1:
            continue
        rect_fill = area / (rw * rh + 1e-9)
        if rect_fill < 0.38:
            continue
        e0 = box[1] - box[0]
        e1 = box[2] - box[1]
        if np.linalg.norm(e0) >= np.linalg.norm(e1):
            d = e0 / (np.linalg.norm(e0) + 1e-9)
            length = np.linalg.norm(e0)
        else:
            d = e1 / (np.linalg.norm(e1) + 1e-9)
            length = np.linalg.norm(e1)
        center = np.array([cx, cy], dtype=float)
        windows.append(
            {
                "poly": [(float(x), float(y)) for x, y in box],
                "lines": [[tuple(center - d * length / 2.0), tuple(center + d * length / 2.0)]],
            }
        )
    return windows


def farthest_pair(pts):
    hull = cv2.convexHull(pts.astype(np.int32)).reshape(-1, 2).astype(float)
    best = (0.0, hull[0], hull[0])
    for i in range(len(hull)):
        for j in range(i + 1, len(hull)):
            dist = float(np.linalg.norm(hull[i] - hull[j]))
            if dist > best[0]:
                best = (dist, hull[i], hull[j])
    return best


def door_axes(cnt):
    pts = cnt.reshape(-1, 2).astype(float)
    dist, a, b = farthest_pair(pts)
    if dist < 6:
        return None
    mid = (a + b) / 2.0
    ab = b - a
    n = np.array([-ab[1], ab[0]], dtype=float)
    n /= np.linalg.norm(n) + 1e-9
    center = pts.mean(axis=0)
    h1 = mid + n * dist / 2.0
    h2 = mid - n * dist / 2.0
    hinge = h1 if np.linalg.norm(h1 - center) > np.linalg.norm(h2 - center) else h2
    radius = dist / math.sqrt(2.0)
    arm_a = (a - hinge) / (np.linalg.norm(a - hinge) + 1e-9)
    arm_b = (b - hinge) / (np.linalg.norm(b - hinge) + 1e-9)
    return hinge, radius, arm_a, arm_b


def struct_count(mask, p, half):
    h, w = mask.shape[:2]
    x0 = max(0, int(round(p[0] - half)))
    x1 = min(w, int(round(p[0] + half)) + 1)
    y0 = max(0, int(round(p[1] - half)))
    y1 = min(h, int(round(p[1] + half)) + 1)
    if x1 <= x0 or y1 <= y0:
        return 0
    return int((mask[y0:y1, x0:x1] > 0).sum())


def snap_dir_45(v):
    theta = round(math.atan2(v[1], v[0]) / (math.pi / 4.0)) * (math.pi / 4.0)
    return np.array([math.cos(theta), math.sin(theta)], dtype=float)


def build_door_symbol(hinge, wall_dir, swing_dir, width, wall_t, arc_n=22):
    h = np.asarray(hinge, dtype=float)
    w = np.asarray(wall_dir, dtype=float)
    s = np.asarray(swing_dir, dtype=float)
    t = float(wall_t)
    opening = [h, h + w * width, h + w * width - s * t, h - s * t]
    panel_t = max(width * 0.08, 3.0)
    leaf = [h - s * t, h - s * t + w * panel_t, h + s * width + w * panel_t, h + s * width]
    arc = [h + width * (w * math.cos(a) + s * math.sin(a)) for a in np.linspace(0, math.pi / 2, arc_n)]
    return {
        "opening": [(float(p[0]), float(p[1])) for p in opening],
        "leaf": [(float(p[0]), float(p[1])) for p in leaf],
        "arc": [(float(p[0]), float(p[1])) for p in arc],
    }


def trace_doors(red_mask, struct_mask, wall_t, min_area=120):
    out = []
    cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    half = max(6, int(wall_t))
    for cnt in cnts:
        if cv2.contourArea(cnt) < min_area:
            continue
        axes = door_axes(cnt)
        if axes is None:
            continue
        hinge, radius, arm_a, arm_b = axes
        end_a = hinge + arm_a * radius
        end_b = hinge + arm_b * radius
        if struct_count(struct_mask, end_a, half) >= struct_count(struct_mask, end_b, half):
            wall_dir, swing_dir = arm_a, arm_b
        else:
            wall_dir, swing_dir = arm_b, arm_a
        wall_dir = snap_dir_45(wall_dir)
        perp = np.array([-wall_dir[1], wall_dir[0]], dtype=float)
        swing_dir = perp if np.dot(perp, swing_dir) >= 0 else -perp
        out.append(build_door_symbol(hinge, wall_dir, swing_dir, radius, wall_t))
    return out


def make_detail_mask(img_bgr, blue_raw, red_raw, green_raw):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, fg = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    color_union = cv2.bitwise_or(cv2.bitwise_or(blue_raw, red_raw), green_raw)
    color_union = cv2.dilate(color_union, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    detail = cv2.bitwise_and(fg, cv2.bitwise_not(color_union))
    return cv2.morphologyEx(detail, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))


def trace_details(detail_mask, min_area=30):
    cnts, hier = cv2.findContours(detail_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    return [simplify(cnt, 0.005) for cnt in cnts if cv2.contourArea(cnt) >= min_area]


def emit_dxf(out_path, wall_polys, centerlines, windows, doors, details, scale, x0, y0, include_centerlines):
    doc = ezdxf.new("R2010")
    doc.units = 4
    doc.header["$LTSCALE"] = 1
    msp = doc.modelspace()
    doc.layers.add("WALLS", color=7, lineweight=35)
    doc.layers.add("WALL_CENTER", color=9, lineweight=13)
    doc.layers.add("DOORS", color=1, lineweight=25)
    doc.layers.add("WINDOWS", color=3, lineweight=25)
    doc.layers.add("DETAILS", color=8, lineweight=13)
    if "DASHED" not in doc.linetypes:
        doc.linetypes.add("DASHED", [200.0, 100.0, -50.0], description="dashed")

    for poly in wall_polys:
        if len(poly) >= 3:
            msp.add_lwpolyline(px_to_mm(poly, scale, x0, y0), close=True, dxfattribs={"layer": "WALLS"})

    if include_centerlines:
        for seg in centerlines:
            pts = seg.get("points", [tuple(seg["p1"]), tuple(seg["p2"])])
            msp.add_lwpolyline(
                px_to_mm(pts, scale, x0, y0),
                close=False,
                dxfattribs={"layer": "WALL_CENTER", "linetype": "DASHED"},
            )

    for win in windows:
        msp.add_lwpolyline(px_to_mm(win["poly"], scale, x0, y0), close=True, dxfattribs={"layer": "WINDOWS"})
        for line in win["lines"]:
            msp.add_lwpolyline(px_to_mm(line, scale, x0, y0), close=False, dxfattribs={"layer": "WINDOWS"})

    for door in doors:
        msp.add_lwpolyline(px_to_mm(door["opening"], scale, x0, y0), close=True, dxfattribs={"layer": "DOORS"})
        msp.add_lwpolyline(px_to_mm(door["leaf"], scale, x0, y0), close=True, dxfattribs={"layer": "DOORS"})
        msp.add_lwpolyline(
            px_to_mm(door["arc"], scale, x0, y0),
            close=False,
            dxfattribs={"layer": "DOORS", "linetype": "DASHED"},
        )

    for poly in details:
        if len(poly) >= 2:
            msp.add_lwpolyline(px_to_mm(poly, scale, x0, y0), close=True, dxfattribs={"layer": "DETAILS"})

    doc.saveas(out_path)


def render_preview(shape, out_path, wall_polys, centerlines, windows, doors, details, skeleton=None):
    canvas = np.ones((shape[0], shape[1], 3), dtype=np.uint8) * 255
    if skeleton is not None:
        canvas[skeleton > 0] = (230, 230, 230)
    for poly in details:
        cv2.polylines(canvas, [np.array(poly, np.int32).reshape(-1, 1, 2)], True, (170, 170, 170), 1)
    for poly in wall_polys:
        cv2.polylines(canvas, [np.array(poly, np.int32).reshape(-1, 1, 2)], True, (0, 0, 0), 2)
    for seg in centerlines:
        pts = np.array(seg.get("points", [tuple(seg["p1"]), tuple(seg["p2"])]), np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], False, (230, 180, 0), 1)
    for win in windows:
        cv2.polylines(canvas, [np.array(win["poly"], np.int32).reshape(-1, 1, 2)], True, (0, 160, 0), 2)
        for line in win["lines"]:
            cv2.polylines(canvas, [np.array(line, np.int32).reshape(-1, 1, 2)], False, (0, 160, 0), 2)
    for door in doors:
        cv2.polylines(canvas, [np.array(door["opening"], np.int32).reshape(-1, 1, 2)], True, (0, 0, 200), 2)
        cv2.polylines(canvas, [np.array(door["leaf"], np.int32).reshape(-1, 1, 2)], True, (0, 0, 200), 2)
        cv2.polylines(canvas, [np.array(door["arc"], np.int32).reshape(-1, 1, 2)], False, (0, 0, 200), 2)
    cv2.imwrite(str(out_path), canvas)


def parse_masks(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, BLUE_LO, BLUE_HI)
    red = cv2.bitwise_or(cv2.inRange(hsv, RED_LO1, RED_HI1), cv2.inRange(hsv, RED_LO2, RED_HI2))
    green = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    return blue, red, green


def main():
    ap = argparse.ArgumentParser(description="Color-annotated floor plan PNG to centerline-cleaned DXF")
    ap.add_argument("input", help="Input PNG, blue=walls, red=doors, green=windows")
    ap.add_argument("-o", "--output", default=None, help="Output DXF path")
    ap.add_argument("--width-mm", type=float, default=None, help="Real floorplan width in millimeters")
    ap.add_argument("--scale", type=float, default=None, help="Millimeters per pixel")
    ap.add_argument("--no-details", action="store_true", help="Skip gray detail layer")
    ap.add_argument("--walls-only", action="store_true", help="Only emit/review the WALLS layer")
    ap.add_argument("--centerlines", action="store_true", help="Also emit detected wall centerlines")
    ap.add_argument("--debug", action="store_true", help="Write skeleton debug PNG")
    args = ap.parse_args()

    input_path = Path(args.input)
    out_dxf = Path(args.output) if args.output else input_path.with_suffix(".dxf")
    img = cv2.imread(str(input_path))
    if img is None:
        sys.exit(f"Cannot read: {input_path}")
    h, w = img.shape[:2]
    print(f"Image: {w}x{h} px")

    blue_raw, red_raw, green_raw = parse_masks(img)
    cols = np.where(blue_raw.any(axis=0))[0]
    rows = np.where(blue_raw.any(axis=1))[0]
    if not len(cols):
        sys.exit("No blue pixels found. Check color thresholds.")
    x0 = int(cols[0])
    y0 = int(rows[-1])
    plan_w = int(cols[-1]) - x0
    if args.scale is not None:
        scale = args.scale
    elif args.width_mm is not None:
        scale = args.width_mm / max(plan_w, 1)
    else:
        print("Warning: no scale supplied; using 1 px = 1 mm")
        scale = 1.0

    blue = morph_clean(blue_raw, close_k=3, open_k=3, close_iter=1)
    green = morph_clean(green_raw, close_k=5, open_k=3, close_iter=2)
    red_open = cv2.morphologyEx(red_raw, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
    red = cv2.morphologyEx(red_open, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=2)

    wall_t = median_wall_thickness(blue)
    print(f"Scale {scale:.4f} mm/px   wall thickness ~{wall_t}px")
    wall_polys, centerlines, skeleton, raw_lines = trace_walls_centerline(blue, wall_t)
    if args.walls_only:
        windows = []
        doors = []
        details = []
    else:
        windows = trace_windows(green, blue, wall_t)
        doors = trace_doors(red, cv2.bitwise_or(blue, green), wall_t)
        details = [] if args.no_details else trace_details(make_detail_mask(img, blue_raw, red_raw, green_raw))

    print(
        f"Extracted: raw_hough={len(raw_lines)}, centerlines={len(centerlines)}, "
        f"wall_polys={len(wall_polys)}, doors={len(doors)}, windows={len(windows)}, details={len(details)}"
    )

    emit_dxf(out_dxf, wall_polys, centerlines, windows, doors, details, scale, x0, y0, args.centerlines)
    print(f"Saved: {out_dxf}")
    preview = out_dxf.with_name(out_dxf.stem + "_preview.png")
    render_preview((h, w), preview, wall_polys, centerlines, windows, doors, details, skeleton if args.debug else None)
    print(f"Preview: {preview}")
    if args.debug:
        debug_skel = out_dxf.with_name(out_dxf.stem + "_skeleton.png")
        cv2.imwrite(str(debug_skel), skeleton)
        print(f"Skeleton: {debug_skel}")


if __name__ == "__main__":
    main()
