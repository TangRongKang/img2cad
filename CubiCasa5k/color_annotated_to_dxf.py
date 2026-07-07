#!/usr/bin/env python3
"""
color_annotated_to_dxf.py

Convert a color-annotated floor plan image to DXF.
  Blue  → WALLS
  Red   → DOORS
  Green → WINDOWS
  other dark content → DETAILS

Method — trace each colour in place, then regularise:
  1. trace walls/windows/doors from their colour masks (no relocation).
  2. self-regularise walls: orthogonalise + cluster near-coincident face lines
     with a SUB-wall-width tolerance → straight, equal-width walls.  The tolerance
     is far smaller than any real wall, so a wall's two faces are never merged
     (thin walls stay thin, thick stay thick).
  3. snap window/door faces & jambs to the wall lines → flush, equal-width,
     connected boundaries — by snapping to the wall lattice, not by rebuilding.
  4. emit once (DXF + matching preview).

Generalises across plans: the only scale is the auto-detected median wall
thickness; every tolerance is a fraction of it.

Usage:
    python color_annotated_to_dxf.py input.png --width-mm 13700 [-o output.dxf]
    python color_annotated_to_dxf.py input.png --scale 5.84
"""

import argparse
import math
import sys
import numpy as np
import cv2
import ezdxf
from shapely.geometry import Polygon

# ── HSV colour thresholds ──────────────────────────────────────────────────────
BLUE_LO  = np.array([ 95, 100,  80]); BLUE_HI  = np.array([145, 255, 255])
RED_LO1  = np.array([  0, 100,  80]); RED_HI1  = np.array([ 12, 255, 255])
RED_LO2  = np.array([165, 100,  80]); RED_HI2  = np.array([180, 255, 255])
GREEN_LO = np.array([ 40,  80,  80]); GREEN_HI = np.array([ 85, 255, 255])


# ── mask helpers ────────────────────────────────────────────────────────────────

def morph_clean(mask, close_k=5, open_k=3, close_iter=2, open_iter=1):
    kc = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
    ko = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k,  open_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc, iterations=close_iter)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN,  ko, iterations=open_iter)


def smooth_edges(mask, sigma=2.0):
    """Gaussian-blur mask boundary for smoother contour vertices."""
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    _, out = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    return out.astype(np.uint8)


def simplify(cnt, eps_frac):
    eps = eps_frac * cv2.arcLength(cnt, True)
    return cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)


def _line_intersect(p1, d1, p2, d2):
    """Intersection of lines p1+t·d1 and p2+s·d2, or None if (near) parallel."""
    det = d2[0] * d1[1] - d1[0] * d2[1]
    if abs(det) < 1e-6:
        return None
    rx, ry = p2[0] - p1[0], p2[1] - p1[1]
    t = (d2[0] * ry - rx * d2[1]) / det
    return (p1[0] + t * d1[0], p1[1] + t * d1[1])


def regularize_dirs(pts, ang_tol=18.0, max_pull=None):
    """Snap each edge to the nearest 0/45/90/135° direction, then place every
    vertex at the intersection of its two snapped edge-lines.

    Axis-aligned walls land on 0°/90°; a 45° wall keeps its slope; an edge too far
    from any 45° multiple keeps its own direction.  This faithfully hugs the mask,
    but at a corner where one edge is a short raster chamfer or a thin off-grid
    needle, the two snapped lines graze and their intersection flies far outside
    the wall — the spike the user saw.  `max_pull` caps that: if the intersection
    strays more than max_pull from the raw contour vertex it replaces, the raw
    vertex is kept instead (no spike), without dropping or moving any real feature.

    Returns (new_pts, edge_is_axis), where edge_is_axis[i] marks edge i as
    horizontal/vertical (used for face clustering).
    """
    pts = [(float(x), float(y)) for x, y in pts]
    n = len(pts)
    if n < 3:
        return pts, [False] * n
    if max_pull is None:
        max_pull = float('inf')
    dirs, anchors, axis = [], [], []
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        ang = math.atan2(y2 - y1, x2 - x1)
        snapped = round(ang / (math.pi / 4)) * (math.pi / 4)
        if abs(((ang - snapped + math.pi) % (2 * math.pi)) - math.pi) > math.radians(ang_tol):
            snapped = ang                      # too far from the 45° grid → keep raw
        d = (math.cos(snapped), math.sin(snapped))
        dirs.append(d)
        anchors.append(((x1 + x2) / 2, (y1 + y2) / 2))
        axis.append(abs(d[0]) < 1e-6 or abs(d[1]) < 1e-6)
    res = []
    for i in range(n):
        j = (i - 1) % n
        p = _line_intersect(anchors[j], dirs[j], anchors[i], dirs[i])
        if p is None or math.hypot(p[0] - pts[i][0], p[1] - pts[i][1]) > max_pull:
            p = pts[i]                         # runaway / parallel → keep raw vertex
        res.append(p)
    return res, axis


def px_to_mm(pts, scale, x0, y0):
    """Pixel coords → mm.  Origin at bottom-left of floor plan. Y-flipped."""
    return [((x - x0) * scale, (y0 - y) * scale) for x, y in pts]


def _cluster_snap(vals, tol):
    """Map each value to the mean of its cluster (values within tol of each other)."""
    if not vals:
        return {}
    sv = sorted(set(vals))
    groups = [[sv[0]]]
    for v in sv[1:]:
        if v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    snap = {}
    for g in groups:
        rep = sum(g) / len(g)
        for v in g:
            snap[v] = rep
    return snap


def _median_wall_thickness(blue_mask):
    """Estimate median wall thickness (px) via scan-line run-length analysis."""
    H, W = blue_mask.shape[:2]
    runs = []

    def _collect(line):
        in_run, n = False, 0
        for v in line:
            if v > 0:
                n += 1; in_run = True
            elif in_run:
                if 5 < n < 150:
                    runs.append(n)
                n = 0; in_run = False
        if in_run and 5 < n < 150:
            runs.append(n)

    for y in range(H // 6, 5 * H // 6, max(H // 20, 1)):
        _collect(blue_mask[y])
    for x in range(W // 6, 5 * W // 6, max(W // 20, 1)):
        _collect(blue_mask[:, x])
    return int(np.median(runs)) if runs else 20


def _l_arm_thicknesses(roi):
    """Return (t_h, t_v): height of the horizontal arm, width of the vertical arm."""
    row_sum = (roi > 0).astype(np.int32).sum(axis=1)
    col_sum = (roi > 0).astype(np.int32).sum(axis=0)
    mr, mc = max(int(row_sum.max()), 1), max(int(col_sum.max()), 1)
    t_h = int(np.sum(row_sum > mr * 0.5))
    t_v = int(np.sum(col_sum > mc * 0.5))
    return max(t_h, 1), max(t_v, 1)


# ── lattice snapping ─────────────────────────────────────────────────────────────

def _snap1(v, lines, tol):
    """Snap a single coordinate to the nearest reference line within tol."""
    best, bd = v, tol
    for L in lines:
        d = abs(L - v)
        if d < bd:
            best, bd = L, d
    return best


def _snap_pair(a, b, lines, tol):
    """Snap two opposite faces to lines, but never collapse them together."""
    na, nb = _snap1(a, lines, tol), _snap1(b, lines, tol)
    if abs(na - nb) < 1:                  # would collapse → keep originals
        return a, b
    return na, nb


# ── stage 1+2: walls (trace → self-regularise → lattice) ────────────────────────

def trace_walls(blue_c, wall_t, min_area=400):
    """Trace walls as closed polygons regularised to the 0/45/90/135° grid.

    Returns (wall_polys, x_lines, y_lines): the polygons (pixel coords) and the
    canonical vertical/horizontal wall-face coordinates that windows snap onto.
    Diagonal walls keep their slope; only axis-aligned faces are clustered.
    """
    cnts, hier = cv2.findContours(blue_c, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    polys, axes = [], []
    # Absolute epsilon between mask-noise (~2-3px) and real features (~10px), so
    # pixel jaggies are removed but a door-jamb step where a frame meets the wall
    # is preserved instead of being flattened into a straight face.  (A
    # perimeter-relative eps would be tens of px on a long wall and erase it.)
    eps = max(wall_t * 0.12, 3.0)
    # Cap how far a snapped corner may stray from the raw contour: at a short raster
    # chamfer or a thin off-grid needle the two snapped lines graze and their
    # intersection flies out into a spike, so beyond ~one wall thickness we keep the
    # raw vertex instead.  Real corners (intersection ≈ raw vertex) are untouched.
    max_pull = max(wall_t * 0.6, 4.0)
    for cnt in cnts:
        if cv2.contourArea(cnt) < min_area:
            continue
        p, ax = regularize_dirs(cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2),
                                max_pull=max_pull)
        polys.append(p); axes.append(ax)
    if not polys:
        return [], [], []

    # A vertex is "axis" only if BOTH its edges are horizontal/vertical.  Cluster
    # (straighten + equalise widths) only those — snapping a diagonal vertex's x/y
    # independently would bend the slope, so diagonal vertices are left untouched.
    vaxis = [[ax[(i - 1) % len(ax)] and ax[i] for i in range(len(ax))] for ax in axes]
    tol = max(2, min(int(round(wall_t * 0.15)), 6))
    ax_x = [round(p[i][0]) for p, va in zip(polys, vaxis) for i in range(len(p)) if va[i]]
    ax_y = [round(p[i][1]) for p, va in zip(polys, vaxis) for i in range(len(p)) if va[i]]
    x_snap, y_snap = _cluster_snap(ax_x, tol), _cluster_snap(ax_y, tol)
    polys = [[(x_snap.get(round(x), x), y_snap.get(round(y), y)) if va[i] else (x, y)
              for i, (x, y) in enumerate(p)]
             for p, va in zip(polys, vaxis)]

    x_lines = sorted({p[i][0] for p, va in zip(polys, vaxis)
                      for i in range(len(p)) if va[i]})
    y_lines = sorted({p[i][1] for p, va in zip(polys, vaxis)
                      for i in range(len(p)) if va[i]})
    return polys, x_lines, y_lines


# ── stage 3: windows (snap faces & ends to the wall lattice) ─────────────────────

def _straight_window(bx, by, bw, bh, x_lines, y_lines, wall_t):
    """One straight window snapped flush into its wall. Returns (poly, lines)."""
    face_tol, end_tol = wall_t * 0.9, wall_t * 0.7
    if bw >= bh:                          # horizontal: thickness in y, length in x
        y1, y2 = _snap_pair(by, by + bh, y_lines, face_tol)
        x1 = _snap1(bx, x_lines, end_tol)
        x2 = _snap1(bx + bw, x_lines, end_tol)
        cy = (y1 + y2) / 2
        lines = [[(x1, cy), (x2, cy)]]
    else:                                 # vertical: thickness in x, length in y
        x1, x2 = _snap_pair(bx, bx + bw, x_lines, face_tol)
        y1 = _snap1(by, y_lines, end_tol)
        y2 = _snap1(by + bh, y_lines, end_tol)
        cx = (x1 + x2) / 2
        lines = [[(cx, y1), (cx, y2)]]
    poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return poly, lines


def _rotated_window(cnt):
    """A straight window at any angle, from its min-area (rotated) rectangle."""
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    e0, e1 = box[1] - box[0], box[2] - box[1]
    d, L = (e0, np.linalg.norm(e0)) if np.linalg.norm(e0) >= np.linalg.norm(e1) \
        else (e1, np.linalg.norm(e1))
    d = d / (np.linalg.norm(d) + 1e-9)
    c = np.array(rect[0])
    line = [tuple(c - d * L / 2), tuple(c + d * L / 2)]
    return {'poly': [tuple(p) for p in box], 'lines': [line]}


def _window_rect_stats(cnt):
    """(tilt°, rect_fill) of a contour: tilt of its long axis from the 0/90 grid,
    and how solidly it fills its rotated rectangle (a thin diagonal pane fills its
    *rotated* rect well even though its axis-aligned bbox fill is tiny)."""
    (_, _), (rw, rh), ang = cv2.minAreaRect(cnt)
    long_ang = ang if rw >= rh else ang + 90.0
    tilt = abs((long_ang % 90.0 + 45.0) % 90.0 - 45.0)
    rect_fill = cv2.contourArea(cnt) / (rw * rh + 1e-9)
    return tilt, rect_fill


def trace_windows(green_c, x_lines, y_lines, wall_t, min_area=500, min_fill=0.20):
    """Windows: axis-aligned → lattice-snapped pane; diagonal → rotated pane;
    L-corner (axis) → two arm panes."""
    out = []
    cnts, _ = cv2.findContours(green_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        fill = area / (bw * bh)
        if fill < min_fill:
            continue
        tilt, rect_fill = _window_rect_stats(cnt)
        if tilt > 15.0 and rect_fill >= 0.55:
            out.append(_rotated_window(cnt))   # tilted straight window
            continue
        if fill >= 0.65:
            poly, lines = _straight_window(bx, by, bw, bh, x_lines, y_lines, wall_t)
            out.append({'poly': poly, 'lines': lines})
            continue
        # L-shape → split into a horizontal arm and a vertical arm, each on a wall
        roi = green_c[by:by + bh, bx:bx + bw]
        mh, mw = max(bh // 2, 1), max(bw // 2, 1)
        q = [int((roi[:mh, :mw] > 0).sum()), int((roi[:mh, mw:] > 0).sum()),
             int((roi[mh:, :mw] > 0).sum()), int((roi[mh:, mw:] > 0).sum())]
        eq = int(np.argmin(q))
        t_h, t_v = _l_arm_thicknesses(roi)
        hy = by if eq in (2, 3) else by + bh - t_h        # horizontal arm top edge
        vx = bx if eq in (1, 3) else bx + bw - t_v        # vertical   arm left edge
        # Snap each arm to its wall, then fuse into ONE L-outline (not 2 panes).
        poly_h, lines_h = _straight_window(bx, hy, bw, t_h, x_lines, y_lines, wall_t)
        poly_v, lines_v = _straight_window(vx, by, t_v, bh, x_lines, y_lines, wall_t)
        merged = Polygon(poly_h).union(Polygon(poly_v))
        if merged.geom_type == 'Polygon':
            lpoly = [(float(x), float(y)) for x, y in merged.exterior.coords[:-1]]
            # One continuous L centre line: far end of one arm → corner → far end
            (hx1, cy), (hx2, _) = lines_h[0]
            (cx, vy1), (_, vy2) = lines_v[0]
            far_x = hx1 if abs(hx1 - cx) > abs(hx2 - cx) else hx2
            far_y = vy1 if abs(vy1 - cy) > abs(vy2 - cy) else vy2
            l_centre = [(far_x, cy), (cx, cy), (cx, far_y)]
            out.append({'poly': lpoly, 'lines': [l_centre]})
        else:                                             # disjoint after snap → keep 2
            out.append({'poly': poly_h, 'lines': lines_h})
            out.append({'poly': poly_v, 'lines': lines_v})
    return out


# ── stage 3: doors (trace red regions as precise closed contours) ────────────────

def _fit_straight_door(cnt, x_lines, y_lines, wall_t):
    """Fit an axis-aligned red region as a rectangular door/sliding panel and
    snap its ends flush to the nearest wall lines.

    Returns the 4-corner polygon or None if the contour is not rectangular
    enough (e.g. a swing-door sector with an arc).
    """
    area = cv2.contourArea(cnt)
    bx, by, bw, bh = cv2.boundingRect(cnt)
    if bw * bh < 1:
        return None
    fill = area / (bw * bh)
    aspect = max(bw, bh) / (min(bw, bh) + 1e-9)
    # Sliding doors / door leaves are compact rectangles; swing-door sectors
    # that include a quarter-circle fill their bbox poorly OR have a square-ish
    # bbox.  Allow low fill only for very elongated panels (sliding doors).
    if aspect >= 2.5:
        if fill < 0.22:
            return None
    elif aspect >= 1.8:
        if fill < 0.55:
            return None
    else:
        # Near-square regions are usually swing-door sectors with an arc; only
        # accept them as rectangles if they are almost completely solid.
        if fill < 0.88:
            return None

    face_tol, end_tol = wall_t * 0.9, wall_t * 0.8
    if bw >= bh:                          # horizontal panel
        y1, y2 = _snap_pair(by, by + bh, y_lines, face_tol)
        x1 = _snap1(bx, x_lines, end_tol)
        x2 = _snap1(bx + bw, x_lines, end_tol)
    else:                                 # vertical panel
        x1, x2 = _snap_pair(bx, bx + bw, x_lines, face_tol)
        y1 = _snap1(by, y_lines, end_tol)
        y2 = _snap1(by + bh, y_lines, end_tol)
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def trace_doors(red_c, wall_t, x_lines=None, y_lines=None, min_area=80):
    """Trace red door symbols.

    - Sliding / swing door leaves are axis-aligned rectangles: snap their
      ends to the wall lattice so they sit flush in the opening.
    - Swing-door sectors with arcs keep their original contour but use a much
      finer polyline so the arc looks smooth instead of jagged.
    """
    cnts, _ = cv2.findContours(red_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # Try rectangular panel first (sliding doors / plain door leaves)
        if x_lines is not None and y_lines is not None:
            rect = _fit_straight_door(cnt, x_lines, y_lines, wall_t)
            if rect is not None:
                polys.append(rect)
                continue

        # Fallback: high-resolution contour for arcs / arbitrary symbols
        peri = cv2.arcLength(cnt, True)
        eps = max(0.35, peri * 0.0008)
        approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        polys.append([(float(x), float(y)) for x, y in approx])
    return polys



# ── details (everything else, traced as-is) ─────────────────────────────────────

def make_detail_mask(img_bgr, blue_raw, red_raw, green_raw):
    """Binary mask of non-wall/door/window dark content."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, fg = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    color_union = cv2.dilate(
        cv2.bitwise_or(cv2.bitwise_or(blue_raw, red_raw), green_raw), k)
    detail = cv2.bitwise_and(fg, cv2.bitwise_not(color_union))
    # Small CLOSE (not OPEN) to bridge 1-2px gaps in thin lines without
    # thickening them significantly. This reduces broken detail segments
    # after skeletonization.
    kc = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(detail, cv2.MORPH_CLOSE, kc, iterations=1)


def _poly_is_closed(poly, tol=5.0):
    """True if a polyline's first and last points meet (within tol pixels)."""
    if len(poly) < 3:
        return False
    return math.hypot(poly[0][0] - poly[-1][0], poly[0][1] - poly[-1][1]) < tol


def _reconnect_polylines(polys, reconnect_tol=5.0, angle_tol=60.0):
    """Merge nearby open polylines whose endpoints face each other.

    Skeletonization sometimes leaves 1-3px gaps in thin lines.  This step
    reconnects such gaps while avoiding accidental merges of unrelated objects.
    """
    if len(polys) < 2:
        return polys

    def tangent(poly, end):
        if len(poly) < 2:
            return np.array([0.0, 0.0])
        if end == 'last':
            v = np.array(poly[-1], float) - np.array(poly[-2], float)
        else:
            v = np.array(poly[0], float) - np.array(poly[1], float)
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-6 else np.array([0.0, 0.0])

    merged = [list(p) for p in polys]
    changed = True
    max_iter = 10
    iteration = 0
    while changed and iteration < max_iter:
        changed = False
        iteration += 1
        n = len(merged)
        for i in range(n):
            if not merged[i] or len(merged[i]) < 2:
                continue
            best = None
            best_score = float('inf')
            pi = merged[i]
            for j in range(i + 1, n):
                if not merged[j] or len(merged[j]) < 2:
                    continue
                pj = merged[j]
                for end_i in ('last', 'first'):
                    pt_i = pi[-1] if end_i == 'last' else pi[0]
                    tan_i = tangent(pi, end_i)
                    for end_j in ('first', 'last'):
                        pt_j = pj[0] if end_j == 'first' else pj[-1]
                        tan_j = tangent(pj, end_j)
                        d = math.hypot(pt_i[0] - pt_j[0], pt_i[1] - pt_j[1])
                        if d > reconnect_tol:
                            continue
                        # Endpoints must roughly point toward each other
                        dir_i = tan_i if end_i == 'last' else -tan_i
                        dir_j = -tan_j if end_j == 'first' else tan_j
                        dot = float(np.dot(dir_i, dir_j))
                        if dot < math.cos(math.radians(angle_tol)):
                            continue
                        score = d + (1.0 - dot) * 5.0
                        if score < best_score:
                            best_score = score
                            best = (j, end_i, end_j)
            if best:
                j, end_i, end_j = best
                pj = merged[j]
                if end_i == 'last' and end_j == 'first':
                    pi.extend(pj)
                elif end_i == 'last' and end_j == 'last':
                    pi.extend(reversed(pj))
                elif end_i == 'first' and end_j == 'first':
                    pi[:] = list(reversed(pj)) + pi
                else:  # first of pi connects to last of pj
                    pi[:] = pj + pi
                merged[j] = []
                changed = True

    return [p for p in merged if len(p) >= 2]


def _skeleton_to_polylines(skel_bool, min_length=15):
    """Vectorise a 1-pixel wide skeleton into clean open/closed polylines.

    Steps:
      1. Find endpoints (degree 1) and junctions (degree > 2).
      2. Walk from every endpoint to the next endpoint/junction.
      3. Walk every unvisited branch between junctions.
      4. Walk remaining isolated loops.
    Returns a list of [(x, y), ...] polylines in pixel coordinates.
    """
    h, w = skel_bool.shape
    kernel = np.ones((3, 3), dtype=np.uint8)
    degree = cv2.filter2D(skel_bool.astype(np.uint8), -1, kernel) - skel_bool.astype(np.uint8)

    def neighbors(rc):
        r, c = rc
        res = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and skel_bool[nr, nc]:
                    res.append((nr, nc))
        return res

    endpoints, junctions, pixels = [], [], set()
    for r in range(h):
        for c in range(w):
            if skel_bool[r, c]:
                pixels.add((r, c))
                d = degree[r, c]
                if d == 1:
                    endpoints.append((r, c))
                elif d > 2:
                    junctions.append((r, c))

    endpoints_set = set(endpoints)
    junctions_set = set(junctions)
    visited = set()
    polys = []

    def trace(start, prev=None):
        path = [start]
        visited.add(start)
        current = start
        prev_pt = prev
        while True:
            nbs = [n for n in neighbors(current) if n != prev_pt and n not in visited]
            if not nbs:
                break
            next_pt = nbs[0]
            path.append(next_pt)
            visited.add(next_pt)
            prev_pt = current
            current = next_pt
            # Stop at another endpoint or junction
            if current in endpoints_set or current in junctions_set:
                break
        return path

    # 1. Paths starting at endpoints
    for ep in endpoints:
        if ep in visited:
            continue
        path = trace(ep)
        if len(path) >= 2:
            polys.append(path)

    # 2. Branches between junctions
    for j in junctions:
        for nb in neighbors(j):
            if nb not in visited and nb not in junctions_set:
                path = trace(nb, prev=j)
                if len(path) >= 1:
                    polys.append([j] + path)

    # 3. Isolated loops (no endpoints, no junctions)
    remaining = [p for p in pixels if p not in visited]
    while remaining:
        start = remaining[0]
        path = [start]
        visited.add(start)
        current = start
        prev_pt = None
        while True:
            nbs = [n for n in neighbors(current) if n != prev_pt and n not in visited]
            if not nbs:
                break
            next_pt = nbs[0]
            path.append(next_pt)
            visited.add(next_pt)
            prev_pt = current
            current = next_pt
            if current == start:
                break
        if len(path) >= 3:
            polys.append(path)
        remaining = [p for p in pixels if p not in visited]

    # Convert (row, col) -> (x, y) and filter short paths
    result = []
    for path in polys:
        length = sum(
            math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
            for i in range(len(path) - 1)
        )
        if length >= min_length:
            result.append([(float(c), float(r)) for r, c in path])
    return result


def trace_details(detail_mask, min_length=10, eps_frac=0.003, close_gap=2):
    """Trace detail lines as single centreline polylines.

    The old approach traced the boundary of thick strokes (RETR_CCOMP), which
    produced double lines and forcibly closed every contour. Instead we:
      1. Bridge tiny gaps in the detail mask so thin strokes stay connected.
      2. Skeletonize the detail mask to a 1-pixel-wide centreline.
      3. Walk the skeleton into individual edge paths.
      4. Reconnect small gaps between path endpoints (larger tolerance so
         corners and T-junctions do not break apart).
      5. Slightly smooth each path with approxPolyDP while keeping curves.
      6. Emit closed only if the endpoints actually meet.

    This keeps curved furniture (sofas, round tables, arcs) as single clean
    polylines instead of jagged double outlines.
    """
    from skimage.morphology import skeletonize

    # Very mild close to reconnect 1px breaks before thinning.
    if close_gap > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_gap * 2 + 1, close_gap * 2 + 1))
        detail_mask = cv2.morphologyEx(detail_mask, cv2.MORPH_CLOSE, k, iterations=1)

    skel = skeletonize(detail_mask.astype(bool))
    polys = _skeleton_to_polylines(skel, min_length=min_length)
    polys = _reconnect_polylines(polys, reconnect_tol=8.0, angle_tol=90.0)

    smoothed = []
    for poly in polys:
        if len(poly) < 3:
            smoothed.append(poly)
            continue
        pts = np.array(poly, dtype=np.float32).reshape(-1, 1, 2)
        peri = cv2.arcLength(pts, False)
        eps = max(1.0, peri * eps_frac)
        approx = cv2.approxPolyDP(pts, eps, closed=False)
        smoothed.append(approx.reshape(-1, 2).astype(float).tolist())
    return smoothed


# ── stage 4: emit ────────────────────────────────────────────────────────────────

def emit_dxf(wall_polys, windows, doors, details, scale, x0, y0):
    doc = ezdxf.new('R2010')
    doc.units = 4
    doc.header['$LTSCALE'] = 1
    msp = doc.modelspace()
    doc.layers.add('WALLS',   color=7, lineweight=35)
    doc.layers.add('DOORS',   color=1, lineweight=25)
    doc.layers.add('WINDOWS', color=4, lineweight=25)
    doc.layers.add('DETAILS', color=8, lineweight=13)

    for poly in wall_polys:
        if len(poly) >= 3:
            msp.add_lwpolyline(px_to_mm(poly, scale, x0, y0), close=True,
                               dxfattribs={'layer': 'WALLS'})

    for win in windows:
        attr = {'layer': 'WINDOWS'}
        msp.add_lwpolyline(px_to_mm(win['poly'], scale, x0, y0), close=True, dxfattribs=attr)
        for pl in win['lines']:
            mm = px_to_mm(pl, scale, x0, y0)
            if len(mm) >= 2:
                msp.add_lwpolyline(mm, close=False, dxfattribs=attr)

    for poly in doors:
        if len(poly) >= 3:
            msp.add_lwpolyline(px_to_mm(poly, scale, x0, y0), close=True,
                               dxfattribs={'layer': 'DOORS'})

    for poly in details:
        if len(poly) >= 2:
            closed = _poly_is_closed(poly)
            msp.add_lwpolyline(px_to_mm(poly, scale, x0, y0), close=closed,
                               dxfattribs={'layer': 'DETAILS'})
    return doc


def render_preview(shape, wall_polys, windows, doors, details, out_path):
    canvas = np.ones((shape[0], shape[1], 3), np.uint8) * 255
    for poly in details:
        if len(poly) < 2:
            continue
        closed = _poly_is_closed(poly)
        cv2.polylines(canvas, [np.array(poly, np.int32).reshape(-1, 1, 2)],
                      closed, (160, 160, 160), 1)
    for poly in wall_polys:
        cv2.polylines(canvas, [np.array(poly, np.int32).reshape(-1, 1, 2)],
                      True, (0, 0, 0), 2)
    for win in windows:
        cv2.polylines(canvas, [np.array(win['poly'], np.int32)], True, (0, 160, 0), 2)
        for pl in win['lines']:
            cv2.polylines(canvas, [np.array(pl, np.int32)], False, (0, 160, 0), 2)
    for poly in doors:
        if len(poly) >= 3:
            cv2.polylines(canvas, [np.array(poly, np.int32).reshape(-1, 1, 2)],
                          True, (0, 0, 200), 2)
    cv2.imwrite(out_path, canvas)
    print(f'Preview → {out_path}')


# ── driver ──────────────────────────────────────────────────────────────────────

def convert_annotated_image(img_bgr, output_dxf, width_mm=None, scale=None,
                            no_details=False, save_preview=True):
    """Convert a color-annotated floor plan BGR image to DXF.

    Args:
        img_bgr: numpy array in BGR format (as returned by cv2.imread)
        output_dxf: path to write DXF file
        width_mm: real-world floor plan width in mm (mutually exclusive with scale)
        scale: mm per pixel (mutually exclusive with width_mm)
        no_details: if True, skip the DETAILS layer
        save_preview: if True, also save a preview PNG next to the DXF
    """
    if img_bgr is None:
        raise ValueError('Input image is None')
    H, W = img_bgr.shape[:2]
    print(f'Image: {W}×{H} px')

    hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    blue  = cv2.inRange(hsv, BLUE_LO, BLUE_HI)
    red   = cv2.bitwise_or(cv2.inRange(hsv, RED_LO1, RED_HI1),
                           cv2.inRange(hsv, RED_LO2, RED_HI2))
    green = cv2.inRange(hsv, GREEN_LO, GREEN_HI)

    blue_w  = morph_clean(blue, close_k=3, open_k=3, close_iter=1)   # sharp — wall geometry
    blue_c  = smooth_edges(blue_w, sigma=2.0)                        # smoothed — thickness/struct
    # Doors are thin outline arcs or rails: open(2) to drop speckle, then a close
    # to consolidate each symbol — keeping the contour intact so it can be traced
    # as-is (smoothing/large opening would distort the shape).
    red_o   = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    red_c   = cv2.morphologyEx(red_o, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=2)
    green_c = morph_clean(green, close_k=5, open_k=3, close_iter=2)

    cols = np.where(blue.any(axis=0))[0]
    rows = np.where(blue.any(axis=1))[0]
    if not len(cols):
        raise ValueError('No blue pixels found — check colour thresholds.')
    x0, y0 = int(cols[0]), int(rows[-1])
    plan_w = int(cols[-1]) - x0
    if scale:
        used_scale = scale
    elif width_mm:
        used_scale = width_mm / plan_w
    else:
        print('Warning: no scale given — 1 px = 1 mm')
        used_scale = 1.0

    wall_t = _median_wall_thickness(blue_c)
    print(f'Scale {used_scale:.4f} mm/px   wall thickness ~{wall_t}px')

    # trace → regularise → emit
    wall_polys, x_lines, y_lines = trace_walls(blue_w, wall_t)
    windows = trace_windows(green_c, x_lines, y_lines, wall_t)
    # Doors: pass wall lattice so straight leaves snap flush to walls; arcs keep
    # high-resolution contours for a smooth curve.
    doors   = trace_doors(red_c, wall_t, x_lines, y_lines)
    details = [] if no_details else trace_details(make_detail_mask(img_bgr, blue, red, green))
    print(f'Extracted: walls={len(wall_polys)}, doors={len(doors)}, '
          f'windows={len(windows)}, details={len(details)}')

    doc = emit_dxf(wall_polys, windows, doors, details, used_scale, x0, y0)
    doc.saveas(output_dxf)
    print(f'Saved: {output_dxf}')

    if save_preview:
        render_preview((H, W), wall_polys, windows, doors, details,
                       output_dxf.rsplit('.', 1)[0] + '_preview.png')

    return output_dxf


def main():
    ap = argparse.ArgumentParser(description='Color-annotated floor plan PNG → DXF')
    ap.add_argument('input', help='Input PNG (blue=walls, red=doors, green=windows)')
    ap.add_argument('-o', '--output', default=None)
    ap.add_argument('--width-mm', type=float, default=None,
                    help='Real-world floor plan width in mm')
    ap.add_argument('--scale', type=float, default=None, help='mm per pixel')
    ap.add_argument('--no-details', action='store_true', help='Skip the DETAILS layer')
    args = ap.parse_args()

    out_dxf = args.output or args.input.rsplit('.', 1)[0] + '.dxf'
    img = cv2.imread(args.input)
    if img is None:
        sys.exit(f'Cannot read: {args.input}')

    convert_annotated_image(
        img, out_dxf,
        width_mm=args.width_mm,
        scale=args.scale,
        no_details=args.no_details,
        save_preview=True
    )


if __name__ == '__main__':
    main()
