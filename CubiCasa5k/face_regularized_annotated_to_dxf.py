#!/usr/bin/env python3
"""
face_regularized_annotated_to_dxf.py

Color-annotated floor plan PNG -> DXF, using wall-face regularization.

This is deliberately not a centerline pipeline. Walls are traced as connected
face contours, so real notches, jambs, pilasters, diagonal faces, and curved
segments remain part of the wall outline. Only near-horizontal/vertical face
lines are snapped to a global coordinate lattice, with a small tolerance intended
to remove 1-2px annotation/raster drift without deleting real small corners.

Usage:
    conda run -n agent python face_regularized_annotated_to_dxf.py test/annotated/1_annotated.png --width-mm 13700
    conda run -n agent python face_regularized_annotated_to_dxf.py input.png --scale 5.84 -o output.dxf
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

    for y in range(h // 6, 5 * h // 6, max(h // 22, 1)):
        collect(blue_mask[y])
    for x in range(w // 6, 5 * w // 6, max(w // 22, 1)):
        collect(blue_mask[:, x])
    return int(np.median(runs)) if runs else 20


def angle_delta(a, b):
    return abs(((a - b + 90.0) % 180.0) - 90.0)


def edge_angle(a, b):
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0


def line_intersection(p1, d1, p2, d2):
    det = d2[0] * d1[1] - d1[0] * d2[1]
    if abs(det) < 1e-6:
        return None
    rx = p2[0] - p1[0]
    ry = p2[1] - p1[1]
    t = (d2[0] * ry - rx * d2[1]) / det
    return np.array([p1[0] + t * d1[0], p1[1] + t * d1[1]], dtype=float)


def weighted_clusters(items, tol):
    """Cluster [(value, weight), ...], preserving close real dimensions."""
    if not items:
        return {}
    items = sorted(items, key=lambda x: x[0])
    groups = [[items[0]]]
    for item in items[1:]:
        if item[0] - groups[-1][-1][0] <= tol:
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


def classify_edge(a, b, wall_t, ang_tol=15.0, allow_diagonal=True):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length = float(math.hypot(dx, dy))
    if length < 1e-6:
        return {"kind": "point", "length": 0.0}
    ang = edge_angle(a, b)
    if angle_delta(ang, 0.0) <= ang_tol:
        return {"kind": "h", "length": length, "coord": (a[1] + b[1]) / 2.0}
    if angle_delta(ang, 90.0) <= ang_tol:
        return {"kind": "v", "length": length, "coord": (a[0] + b[0]) / 2.0}
    # Real diagonal walls are long. Short diagonal/chamfer edges near corners are
    # usually raster/simplification artifacts and are handled later as 90-degree
    # small jogs, not as diagonal walls.
    if not allow_diagonal or length < max(12.0, wall_t * 1.2):
        return {"kind": "raw", "length": length}
    for target in (45.0, 135.0):
        if angle_delta(ang, target) <= 5.0:
            rad = math.radians(target)
            d = np.array([math.cos(rad), math.sin(rad)], dtype=float)
            return {"kind": "diag", "length": length, "dir": d, "anchor": (a + b) / 2.0}
    return {"kind": "raw", "length": length}


def build_edge_line(a, b, info, h_snap, v_snap):
    kind = info["kind"]
    if kind == "h":
        y = h_snap.get(info["coord"], info["coord"])
        return {"snapped": True, "p": np.array([0.0, y]), "d": np.array([1.0, 0.0])}
    if kind == "v":
        x = v_snap.get(info["coord"], info["coord"])
        return {"snapped": True, "p": np.array([x, 0.0]), "d": np.array([0.0, 1.0])}
    if kind == "diag":
        return {"snapped": True, "p": info["anchor"], "d": info["dir"]}
    return {"snapped": False, "p": (a + b) / 2.0, "d": b - a}


def regularize_polygon_faces(points, wall_t, h_snap, v_snap, allow_diagonal=True):
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 3:
        return [(float(x), float(y)) for x, y in pts]

    infos = [classify_edge(pts[i], pts[(i + 1) % n], wall_t, allow_diagonal=allow_diagonal) for i in range(n)]
    lines = [build_edge_line(pts[i], pts[(i + 1) % n], infos[i], h_snap, v_snap) for i in range(n)]
    max_pull = max(4.0, wall_t * 0.75)
    out = []
    for i in range(n):
        prev_i = (i - 1) % n
        raw = pts[i]
        prev_line = lines[prev_i]
        cur_line = lines[i]
        if prev_line["snapped"] and cur_line["snapped"]:
            p = line_intersection(prev_line["p"], prev_line["d"], cur_line["p"], cur_line["d"])
            if p is not None and np.linalg.norm(p - raw) <= max_pull:
                out.append((float(p[0]), float(p[1])))
                continue
        # If only one adjacent face is confidently axis-aligned, project just that
        # coordinate. This fixes one-sided raster drift but keeps the other degree
        # of freedom for real small steps.
        x, y = raw
        if cur_line["snapped"] and infos[i]["kind"] == "h":
            y = cur_line["p"][1]
        elif prev_line["snapped"] and infos[prev_i]["kind"] == "h":
            y = prev_line["p"][1]
        if cur_line["snapped"] and infos[i]["kind"] == "v":
            x = cur_line["p"][0]
        elif prev_line["snapped"] and infos[prev_i]["kind"] == "v":
            x = prev_line["p"][0]
        if math.hypot(x - raw[0], y - raw[1]) <= max_pull:
            out.append((float(x), float(y)))
        else:
            out.append((float(raw[0]), float(raw[1])))
    out = remove_duplicate_neighbors(out)
    out = restore_short_orthogonal_jogs(out, wall_t, allow_diagonal=allow_diagonal)
    if not allow_diagonal:
        return force_manhattan_polygon(out)
    return snap_remaining_near_axis_edges(out)


def restore_short_orthogonal_jogs(poly, wall_t, allow_diagonal=True):
    """Replace tiny diagonal chamfers with explicit 90-degree jogs.

    This preserves real small wall offsets as H/V geometry. True diagonal walls
    are long enough to survive classify_edge() as diagonal lines before this pass.
    """
    if len(poly) < 3:
        return poly
    pts = [np.array(p, dtype=float) for p in poly]
    out = []
    short_diag = max(8.0, wall_t * 1.15)
    for i, p in enumerate(pts):
        q = pts[(i + 1) % len(pts)]
        dx = q[0] - p[0]
        dy = q[1] - p[1]
        length = float(math.hypot(dx, dy))
        ang = edge_angle(p, q) if length > 1e-6 else 0.0
        is_axis = angle_delta(ang, 0.0) <= 8.0 or angle_delta(ang, 90.0) <= 8.0
        out.append((float(p[0]), float(p[1])))
        should_orthogonalize = (
            not is_axis
            and abs(dx) > 1.0
            and abs(dy) > 1.0
            and (not allow_diagonal or length <= short_diag)
        )
        if should_orthogonalize:
            prev = pts[(i - 1) % len(pts)]
            nxt = pts[(i + 2) % len(pts)]
            prev_ang = edge_angle(prev, p)
            next_ang = edge_angle(q, nxt)
            prev_h = angle_delta(prev_ang, 0.0) <= 15.0
            prev_v = angle_delta(prev_ang, 90.0) <= 15.0
            next_h = angle_delta(next_ang, 0.0) <= 15.0
            next_v = angle_delta(next_ang, 90.0) <= 15.0
            prev_axis = prev_h or prev_v
            next_axis = next_h or next_v
            prev_len = float(np.linalg.norm(p - prev))
            next_len = float(np.linalg.norm(nxt - q))
            # When diagonal walls are globally disabled, every non-axis segment is
            # a contour artifact and must become a 90-degree jog. Otherwise keep
            # long diagonal/curved runs: if a short segment is surrounded by other
            # non-axis segments, it is probably one piece of a longer sloped wall.
            isolated = (
                not allow_diagonal
                or prev_axis
                or next_axis
                or prev_len > short_diag * 1.4
                or next_len > short_diag * 1.4
            )
            if isolated:
                corner1 = (float(q[0]), float(p[1]))
                corner2 = (float(p[0]), float(q[1]))
                if prev_h or next_v:
                    corner = corner1
                elif prev_v or next_h:
                    corner = corner2
                else:
                    # With no reliable neighboring axis, choose the shorter
                    # visual jog. Both alternatives are 90-degree.
                    corner = corner1 if abs(dx) <= abs(dy) else corner2
                if math.hypot(corner[0] - out[-1][0], corner[1] - out[-1][1]) > 0.75:
                    out.append(corner)
    return remove_duplicate_neighbors(out)


def remove_duplicate_neighbors(poly, eps=0.75):
    out = []
    for p in poly:
        if not out or math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > eps:
            out.append(p)
    if len(out) > 1 and math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) <= eps:
        out.pop()
    return out


def snap_remaining_near_axis_edges(poly, ang_tol=15.0):
    """Make any remaining near-axis contour edges exactly H/V."""
    if len(poly) < 2:
        return poly
    pts = [np.array(p, dtype=float) for p in poly]
    for _ in range(6):
        for i in range(len(pts)):
            j = (i + 1) % len(pts)
            p = pts[i]
            q = pts[j]
            length = float(np.linalg.norm(q - p))
            if length < 1e-6:
                continue
            ang = edge_angle(p, q)
            if angle_delta(ang, 0.0) <= ang_tol:
                y = (p[1] + q[1]) / 2.0
                pts[i][1] = y
                pts[j][1] = y
            elif angle_delta(ang, 90.0) <= ang_tol:
                x = (p[0] + q[0]) / 2.0
                pts[i][0] = x
                pts[j][0] = x
        cleaned = remove_duplicate_neighbors([(float(p[0]), float(p[1])) for p in pts])
        pts = [np.array(p, dtype=float) for p in cleaned]
        if len(pts) < 2:
            break
    for i in range(len(pts)):
        j = (i + 1) % len(pts)
        p = pts[i]
        q = pts[j]
        length = float(np.linalg.norm(q - p))
        if length < 1e-6:
            continue
        ang = edge_angle(p, q)
        if angle_delta(ang, 0.0) <= ang_tol:
            y = round((p[1] + q[1]) / 2.0, 6)
            pts[i][1] = y
            pts[j][1] = y
        elif angle_delta(ang, 90.0) <= ang_tol:
            x = round((p[0] + q[0]) / 2.0, 6)
            pts[i][0] = x
            pts[j][0] = x
    return [(round(float(p[0]), 6), round(float(p[1]), 6)) for p in pts]


def force_manhattan_polygon(poly):
    """Rebuild a polygon so every edge is exactly horizontal or vertical."""
    if len(poly) < 3:
        return poly
    pts = [np.array(p, dtype=float) for p in poly]
    edges = []
    for i in range(len(pts)):
        a = pts[i]
        b = pts[(i + 1) % len(pts)]
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        if abs(dx) >= abs(dy):
            edges.append({"kind": "h", "coord": round(float((a[1] + b[1]) / 2.0), 6)})
        else:
            edges.append({"kind": "v", "coord": round(float((a[0] + b[0]) / 2.0), 6)})

    out = []
    for i, raw in enumerate(pts):
        prev = edges[(i - 1) % len(edges)]
        cur = edges[i]
        x = round(float(raw[0]), 6)
        y = round(float(raw[1]), 6)
        if prev["kind"] == "h" and cur["kind"] == "v":
            x = cur["coord"]
            y = prev["coord"]
        elif prev["kind"] == "v" and cur["kind"] == "h":
            x = prev["coord"]
            y = cur["coord"]
        elif prev["kind"] == "h" and cur["kind"] == "h":
            y = round(float((prev["coord"] + cur["coord"]) / 2.0), 6)
        elif prev["kind"] == "v" and cur["kind"] == "v":
            x = round(float((prev["coord"] + cur["coord"]) / 2.0), 6)
        out.append((x, y))
    return remove_duplicate_neighbors(out)


def collect_face_snap_lines(raw_polys, wall_t, allow_diagonal=True):
    h_vals = []
    v_vals = []
    min_face = max(5.0, wall_t * 0.25)
    for poly in raw_polys:
        pts = np.asarray(poly, dtype=float)
        for i in range(len(pts)):
            a = pts[i]
            b = pts[(i + 1) % len(pts)]
            info = classify_edge(a, b, wall_t, allow_diagonal=allow_diagonal)
            if info["length"] < min_face:
                continue
            if info["kind"] == "h":
                h_vals.append((info["coord"], info["length"]))
            elif info["kind"] == "v":
                v_vals.append((info["coord"], info["length"]))
    # This is intentionally small. It removes raster drift while preserving real
    # small notches/jamb offsets that are more than a few pixels.
    tol = max(1.5, min(4.0, wall_t * 0.18))
    return weighted_clusters(h_vals, tol), weighted_clusters(v_vals, tol)


def has_real_diagonal_wall(raw_polys, wall_t):
    """Detect whether the plan has meaningful long diagonal wall faces."""
    min_len = max(70.0, wall_t * 3.0)
    total = 0.0
    longest = 0.0
    for poly in raw_polys:
        pts = np.asarray(poly, dtype=float)
        for i in range(len(pts)):
            a = pts[i]
            b = pts[(i + 1) % len(pts)]
            length = float(np.linalg.norm(b - a))
            if length < min_len:
                continue
            ang = edge_angle(a, b)
            if angle_delta(ang, 0.0) > 15.0 and angle_delta(ang, 90.0) > 15.0:
                total += length
                longest = max(longest, length)
    return longest >= min_len and total >= min_len * 1.5


def trace_walls_face_regularized(blue_mask, wall_t, min_area=400, force_orthogonal=False):
    cnts, hier = cv2.findContours(blue_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    raw_polys = []
    eps = max(1.2, wall_t * 0.045)
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
        if len(approx) >= 3:
            raw_polys.append([(float(x), float(y)) for x, y in approx])
    if not raw_polys:
        return [], [], []

    allow_diagonal = (not force_orthogonal) and has_real_diagonal_wall(raw_polys, wall_t)
    h_snap, v_snap = collect_face_snap_lines(raw_polys, wall_t, allow_diagonal=allow_diagonal)
    wall_polys = [
        regularize_polygon_faces(poly, wall_t, h_snap, v_snap, allow_diagonal=allow_diagonal)
        for poly in raw_polys
    ]
    x_lines = sorted(set(v_snap.values()))
    y_lines = sorted(set(h_snap.values()))
    return [p for p in wall_polys if len(p) >= 3], x_lines, y_lines


def snap1(v, lines, tol):
    best, bd = v, tol
    for line in lines:
        d = abs(line - v)
        if d < bd:
            best, bd = line, d
    return best


def snap_pair(a, b, lines, tol):
    na = snap1(a, lines, tol)
    nb = snap1(b, lines, tol)
    if abs(na - nb) < 1:
        return a, b
    return na, nb


def trace_windows(green_mask, x_lines, y_lines, wall_t, min_area=180):
    out = []
    cnts, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        rw, rh = rect[1]
        if rw <= 1 or rh <= 1:
            continue
        rect_fill = area / (rw * rh + 1e-9)
        if rect_fill < 0.45:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if abs(rw - w) < wall_t * 0.5 and abs(rh - h) < wall_t * 0.5:
            if w >= h:
                y1, y2 = snap_pair(y, y + h, y_lines, wall_t * 0.9)
                x1 = snap1(x, x_lines, wall_t * 0.7)
                x2 = snap1(x + w, x_lines, wall_t * 0.7)
                cy = (y1 + y2) / 2.0
                poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                lines = [[(x1, cy), (x2, cy)]]
            else:
                x1, x2 = snap_pair(x, x + w, x_lines, wall_t * 0.9)
                y1 = snap1(y, y_lines, wall_t * 0.7)
                y2 = snap1(y + h, y_lines, wall_t * 0.7)
                cx = (x1 + x2) / 2.0
                poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                lines = [[(cx, y1), (cx, y2)]]
            out.append({"poly": poly, "lines": lines})
        else:
            e0 = box[1] - box[0]
            e1 = box[2] - box[1]
            d, length = (e0, np.linalg.norm(e0)) if np.linalg.norm(e0) >= np.linalg.norm(e1) else (e1, np.linalg.norm(e1))
            d = d / (np.linalg.norm(d) + 1e-9)
            c = np.array(rect[0])
            out.append({"poly": [tuple(p) for p in box], "lines": [[tuple(c - d * length / 2), tuple(c + d * length / 2)]]})
    return out


def farthest_pair(pts):
    hull = cv2.convexHull(pts.astype(np.int32)).reshape(-1, 2).astype(float)
    best = (0.0, hull[0], hull[0])
    for i in range(len(hull)):
        for j in range(i + 1, len(hull)):
            d = float(np.linalg.norm(hull[i] - hull[j]))
            if d > best[0]:
                best = (d, hull[i], hull[j])
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


def snap_dir45(v):
    theta = round(math.atan2(v[1], v[0]) / (math.pi / 4.0)) * (math.pi / 4.0)
    return np.array([math.cos(theta), math.sin(theta)], dtype=float)


def build_door(hinge, wall_dir, swing_dir, width, wall_t, arc_n=22):
    h = np.asarray(hinge, dtype=float)
    w = np.asarray(wall_dir, dtype=float)
    s = np.asarray(swing_dir, dtype=float)
    t = float(wall_t)
    opening = [h, h + w * width, h + w * width - s * t, h - s * t]
    leaf_t = max(width * 0.08, 3.0)
    leaf = [h - s * t, h - s * t + w * leaf_t, h + s * width + w * leaf_t, h + s * width]
    arc = [h + width * (w * math.cos(a) + s * math.sin(a)) for a in np.linspace(0, math.pi / 2, arc_n)]
    return {
        "opening": [(float(p[0]), float(p[1])) for p in opening],
        "leaf": [(float(p[0]), float(p[1])) for p in leaf],
        "arc": [(float(p[0]), float(p[1])) for p in arc],
    }


def trace_doors(red_mask, struct_mask, wall_t, min_area=180):
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
        wall_dir = snap_dir45(wall_dir)
        perp = np.array([-wall_dir[1], wall_dir[0]], dtype=float)
        swing_dir = perp if np.dot(perp, swing_dir) >= 0 else -perp
        out.append(build_door(hinge, wall_dir, swing_dir, radius, wall_t))
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


def emit_dxf(out_path, wall_polys, windows, doors, details, scale, x0, y0):
    doc = ezdxf.new("R2010")
    doc.units = 4
    doc.header["$LTSCALE"] = 1
    msp = doc.modelspace()
    doc.layers.add("WALLS", color=7, lineweight=35)
    doc.layers.add("DOORS", color=1, lineweight=25)
    doc.layers.add("WINDOWS", color=3, lineweight=25)
    doc.layers.add("DETAILS", color=8, lineweight=13)
    doc.linetypes.add("DASHED", [200.0, 100.0, -50.0], description="dashed")

    for poly in wall_polys:
        if len(poly) >= 3:
            msp.add_lwpolyline(px_to_mm(poly, scale, x0, y0), close=True, dxfattribs={"layer": "WALLS"})
    for win in windows:
        msp.add_lwpolyline(px_to_mm(win["poly"], scale, x0, y0), close=True, dxfattribs={"layer": "WINDOWS"})
        for line in win["lines"]:
            msp.add_lwpolyline(px_to_mm(line, scale, x0, y0), close=False, dxfattribs={"layer": "WINDOWS"})
    for door in doors:
        msp.add_lwpolyline(px_to_mm(door["opening"], scale, x0, y0), close=True, dxfattribs={"layer": "DOORS"})
        msp.add_lwpolyline(px_to_mm(door["leaf"], scale, x0, y0), close=True, dxfattribs={"layer": "DOORS"})
        msp.add_lwpolyline(px_to_mm(door["arc"], scale, x0, y0), close=False, dxfattribs={"layer": "DOORS", "linetype": "DASHED"})
    for poly in details:
        if len(poly) >= 2:
            msp.add_lwpolyline(px_to_mm(poly, scale, x0, y0), close=True, dxfattribs={"layer": "DETAILS"})
    doc.saveas(out_path)


def render_preview(shape, out_path, wall_polys, windows, doors, details):
    canvas = np.ones((shape[0], shape[1], 3), dtype=np.uint8) * 255
    for poly in details:
        cv2.polylines(canvas, [np.array(poly, np.int32).reshape(-1, 1, 2)], True, (170, 170, 170), 1)
    for poly in wall_polys:
        cv2.polylines(canvas, [np.array(poly, np.int32).reshape(-1, 1, 2)], True, (0, 0, 0), 2)
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
    ap = argparse.ArgumentParser(description="Color-annotated floor plan PNG to face-regularized DXF")
    ap.add_argument("input", help="Input PNG, blue=walls, red=doors, green=windows")
    ap.add_argument("-o", "--output", default=None, help="Output DXF path")
    ap.add_argument("--width-mm", type=float, default=None, help="Real floorplan width in millimeters")
    ap.add_argument("--scale", type=float, default=None, help="Millimeters per pixel")
    ap.add_argument("--walls-only", action="store_true", help="Only emit/review the WALLS layer")
    ap.add_argument("--orthogonal-walls", action="store_true", help="Force every wall face to horizontal/vertical")
    ap.add_argument("--no-details", action="store_true", help="Skip gray detail layer")
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
    walls, x_lines, y_lines = trace_walls_face_regularized(blue, wall_t, force_orthogonal=args.orthogonal_walls)
    if args.walls_only:
        windows, doors, details = [], [], []
    else:
        windows = trace_windows(green, x_lines, y_lines, wall_t)
        doors = trace_doors(red, cv2.bitwise_or(blue, green), wall_t)
        details = [] if args.no_details else trace_details(make_detail_mask(img, blue_raw, red_raw, green_raw))

    print(f"Extracted: walls={len(walls)}, doors={len(doors)}, windows={len(windows)}, details={len(details)}")
    emit_dxf(out_dxf, walls, windows, doors, details, scale, x0, y0)
    print(f"Saved: {out_dxf}")
    preview = out_dxf.with_name(out_dxf.stem + "_preview.png")
    render_preview((h, w), preview, walls, windows, doors, details)
    print(f"Preview: {preview}")


if __name__ == "__main__":
    main()
