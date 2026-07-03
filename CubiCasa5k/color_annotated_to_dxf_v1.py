#!/usr/bin/env python3
"""
color_annotated_to_dxf.py

Convert a color-annotated floor plan image to DXF.
  Blue  → WALLS
  Red   → DOORS
  Green → WINDOWS (3 parallel lines)

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

# ── HSV colour thresholds ──────────────────────────────────────────────────────
BLUE_LO  = np.array([ 95, 100,  80]); BLUE_HI  = np.array([145, 255, 255])
RED_LO1  = np.array([  0, 100,  80]); RED_HI1  = np.array([ 12, 255, 255])
RED_LO2  = np.array([165, 100,  80]); RED_HI2  = np.array([180, 255, 255])
GREEN_LO = np.array([ 40,  80,  80]); GREEN_HI = np.array([ 85, 255, 255])


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


def orthogonalize(pts):
    """Snap all polygon edges to be exactly horizontal or vertical.

    For each edge, choose H or V by which axis has the larger span.
    Compute the snap value (avg y for H, avg x for V).
    Each vertex is then the intersection of its two adjacent snapped edges.
    """
    pts = [(float(x), float(y)) for x, y in pts]
    n = len(pts)
    if n < 3:
        return pts

    # Classify each edge and compute its representative coordinate
    types = []   # 'H' or 'V'
    vals  = []   # avg-y for H edges, avg-x for V edges
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if abs(x2 - x1) >= abs(y2 - y1):
            types.append('H');  vals.append((y1 + y2) / 2)
        else:
            types.append('V');  vals.append((x1 + x2) / 2)

    # Each vertex is at the intersection of the previous and current edge lines
    result = []
    for i in range(n):
        prev = (i - 1) % n
        tp, vp = types[prev], vals[prev]
        tc, vc = types[i],    vals[i]
        if   tp == 'H' and tc == 'V':
            result.append((vc, vp))            # x from V-edge, y from H-edge
        elif tp == 'V' and tc == 'H':
            result.append((vp, vc))            # x from V-edge, y from H-edge
        elif tp == 'H' and tc == 'H':
            result.append((pts[i][0], (vp + vc) / 2))
        else:  # V–V
            result.append(((vp + vc) / 2, pts[i][1]))
    return result


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


def draw_contours(msp, mask, layer, scale, x0, y0,
                  min_area, eps_frac, retrieve=cv2.RETR_CCOMP, ortho=False,
                  snap_tol=4):
    cnts, hier = cv2.findContours(mask, retrieve, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return 0

    # First pass: collect simplified (+ orthogonalized) point lists
    all_pts = []
    for cnt in cnts:
        if cv2.contourArea(cnt) < min_area:
            continue
        pts = simplify(cnt, eps_frac)
        if ortho:
            pts = orthogonalize(pts)
        all_pts.append(pts)

    if not all_pts:
        return 0

    # Snap close H/V coordinates so adjacent wall segments share exact values
    if ortho and snap_tol > 0:
        flat = [(x, y) for pts in all_pts for x, y in pts]
        x_snap = _cluster_snap([round(x) for x, _ in flat], snap_tol)
        y_snap = _cluster_snap([round(y) for _, y in flat], snap_tol)
        all_pts = [
            [(x_snap.get(round(x), x), y_snap.get(round(y), y)) for x, y in pts]
            for pts in all_pts
        ]

    for pts in all_pts:
        pts_mm = px_to_mm(pts, scale, x0, y0)
        if len(pts_mm) >= 2:
            msp.add_lwpolyline(pts_mm, close=True, dxfattribs={'layer': layer})
    return len(all_pts)


def _l_arm_thicknesses(roi):
    """Return (t_h, t_v): height of horizontal arm and width of vertical arm."""
    row_sum = (roi > 0).astype(np.int32).sum(axis=1)
    col_sum = (roi > 0).astype(np.int32).sum(axis=0)
    mr = max(int(row_sum.max()), 1)
    mc = max(int(col_sum.max()), 1)
    t_h = int(np.sum(row_sum > mr * 0.5))
    t_v = int(np.sum(col_sum > mc * 0.5))
    return max(t_h, 1), max(t_v, 1)


def _window_rect(msp, bx, by, bw, bh, scale, x0, y0, attr):
    """Draw one straight window: closed rectangle + centre line."""
    pts = px_to_mm([(bx, by), (bx+bw, by), (bx+bw, by+bh), (bx, by+bh)],
                   scale, x0, y0)
    msp.add_lwpolyline(pts, close=True, dxfattribs=attr)
    if bw >= bh:
        cy = (y0 - by - bh / 2) * scale
        msp.add_line(((bx - x0)*scale, cy), ((bx+bw - x0)*scale, cy), dxfattribs=attr)
    else:
        cx = (bx + bw / 2 - x0) * scale
        msp.add_line((cx, (y0 - by)*scale), (cx, (y0 - by - bh)*scale), dxfattribs=attr)


def _window_l_shape(msp, bx, by, bw, bh, eq, t_h, t_v, scale, x0, y0, attr):
    """Draw one L-shaped window: single closed 6-vertex polygon + L-shaped centre line."""
    if eq == 3:   # empty BR: horiz arm top, vert arm left
        outer  = [(bx,     by),      (bx+bw,    by),
                  (bx+bw,  by+t_h),  (bx+t_v,   by+t_h),
                  (bx+t_v, by+bh),   (bx,       by+bh)]
        centre = [(bx+bw,  by+t_h/2),(bx+t_v/2, by+t_h/2),(bx+t_v/2, by+bh)]
    elif eq == 2: # empty BL: horiz arm top, vert arm right
        outer  = [(bx,        by),      (bx+bw,      by),
                  (bx+bw,     by+bh),   (bx+bw-t_v,  by+bh),
                  (bx+bw-t_v, by+t_h),  (bx,         by+t_h)]
        centre = [(bx,        by+t_h/2),(bx+bw-t_v/2,by+t_h/2),(bx+bw-t_v/2,by+bh)]
    elif eq == 1: # empty TR: horiz arm bottom, vert arm left
        outer  = [(bx,    by),         (bx+t_v,  by),
                  (bx+t_v,by+bh-t_h),  (bx+bw,   by+bh-t_h),
                  (bx+bw, by+bh),      (bx,      by+bh)]
        centre = [(bx+t_v/2, by),(bx+t_v/2, by+bh-t_h/2),(bx+bw, by+bh-t_h/2)]
    else:         # empty TL: horiz arm bottom, vert arm right
        outer  = [(bx+bw-t_v, by),       (bx+bw,      by),
                  (bx+bw,     by+bh),     (bx,         by+bh),
                  (bx,        by+bh-t_h), (bx+bw-t_v,  by+bh-t_h)]
        centre = [(bx+bw-t_v/2,by),(bx+bw-t_v/2,by+bh-t_h/2),(bx,by+bh-t_h/2)]
    msp.add_lwpolyline(px_to_mm(outer,  scale, x0, y0), close=True,  dxfattribs=attr)
    msp.add_lwpolyline(px_to_mm(centre, scale, x0, y0), close=False, dxfattribs=attr)


def add_windows_rect(msp, mask, layer, scale, x0, y0,
                     min_area=500, min_fill=0.20):
    """Draw each window as a closed outline + centre line.

    Straight windows → closed rectangle + centre line.
    L-shaped windows → single closed 6-vertex L-polygon + L-shaped centre line.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        fill = area / (bw * bh)
        if fill < min_fill:
            continue
        attr = {'layer': layer}
        if fill >= 0.65:
            _window_rect(msp, bx, by, bw, bh, scale, x0, y0, attr)
        else:
            roi = mask[by:by+bh, bx:bx+bw]
            mh, mw = max(bh//2, 1), max(bw//2, 1)
            q = [int((roi[:mh, :mw] > 0).sum()), int((roi[:mh, mw:] > 0).sum()),
                 int((roi[mh:, :mw] > 0).sum()), int((roi[mh:, mw:] > 0).sum())]
            eq = int(np.argmin(q))
            t_h, t_v = _l_arm_thicknesses(roi)
            _window_l_shape(msp, bx, by, bw, bh, eq, t_h, t_v, scale, x0, y0, attr)
        count += 1
    return count


def add_doors_arc(msp, mask, layer, scale, x0, y0, min_area=2000):
    """Draw each door as: 1 straight panel line + 1 dashed quarter-circle arc.

    Pivot detection: find which bbox sub-quadrant has fewest pixels — the arc's
    convex edge cuts across that quadrant, so pivot is the diagonally opposite corner.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0

    for cnt in cnts:
        if cv2.contourArea(cnt) < min_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)

        # Quadrant pixel counts to find empty corner
        roi = mask[by:by+bh, bx:bx+bw]
        mh, mw = max(bh // 2, 1), max(bw // 2, 1)
        q = [int((roi[:mh, :mw] > 0).sum()),   # q0: TL
             int((roi[:mh, mw:] > 0).sum()),   # q1: TR
             int((roi[mh:, :mw] > 0).sum()),   # q2: BL
             int((roi[mh:, mw:] > 0).sum())]   # q3: BR
        eq = int(np.argmin(q))

        # Pivot = corner diagonally opposite the least-filled sub-quadrant
        pivot_px = {0: (bx+bw, by+bh), 1: (bx, by+bh),
                    2: (bx+bw, by),    3: (bx, by)}[eq]

        # Arc angles in DXF y-up coords (CCW, 90° sweep)
        sa, ea = {0: (90, 180), 1: (0, 90),
                  2: (180, 270), 3: (270, 360)}[eq]

        r_mm = ((bw + bh) / 2) * scale
        piv_mm = ((pivot_px[0] - x0) * scale, (y0 - pivot_px[1]) * scale)

        # Panel line: from pivot to the start-angle endpoint
        sa_rad = math.radians(sa)
        panel_end = (piv_mm[0] + r_mm * math.cos(sa_rad),
                     piv_mm[1] + r_mm * math.sin(sa_rad))

        msp.add_line(piv_mm, panel_end, dxfattribs={'layer': layer})
        msp.add_arc(center=piv_mm, radius=r_mm, start_angle=sa, end_angle=ea,
                    dxfattribs={'layer': layer, 'linetype': 'DOOR_DASH'})
        count += 1
    return count


def _draw_arc_cv(canvas, pivot_px, r_px, sa_dxf, ea_dxf, color, n=40):
    """Approximate a DXF arc as a polyline on an OpenCV canvas."""
    pts = []
    for i in range(n + 1):
        a = math.radians(sa_dxf + (ea_dxf - sa_dxf) * i / n)
        px = int(pivot_px[0] + r_px * math.cos(a))
        py = int(pivot_px[1] - r_px * math.sin(a))
        pts.append([px, py])
    cv2.polylines(canvas, [np.array(pts, np.int32)], False, color, 2)


def _draw_arc_cv_dashed(canvas, pivot_px, r_px, sa_dxf, ea_dxf, color,
                        n_dashes=10, dash_frac=0.55):
    """Draw a dashed arc on an OpenCV canvas."""
    total = ea_dxf - sa_dxf
    if total <= 0:
        total += 360
    for i in range(n_dashes):
        t0 = i / n_dashes
        t1 = t0 + dash_frac / n_dashes
        a0 = math.radians(sa_dxf + total * t0)
        a1 = math.radians(sa_dxf + total * t1)
        steps = max(3, int(total * dash_frac / n_dashes / 2))
        pts = []
        for j in range(steps + 1):
            a = a0 + (a1 - a0) * j / steps
            pts.append([int(pivot_px[0] + r_px * math.cos(a)),
                        int(pivot_px[1] - r_px * math.sin(a))])
        if len(pts) >= 2:
            cv2.polylines(canvas, [np.array(pts, np.int32)], False, color, 2)


def make_detail_mask(img_bgr, blue_raw, red_raw, green_raw):
    """Return a binary mask of non-wall/door/window dark content."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, fg = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

    # Dilate colour masks to cleanly erase their edges
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    color_union = cv2.dilate(
        cv2.bitwise_or(cv2.bitwise_or(blue_raw, red_raw), green_raw), k)

    detail = cv2.bitwise_and(fg, cv2.bitwise_not(color_union))

    # Remove single-pixel noise
    ko = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(detail, cv2.MORPH_OPEN, ko)


def add_detail_layer(msp, detail_mask, layer, scale, x0, y0,
                     min_area=30, eps_frac=0.005):
    """Vectorize the detail mask: trace ALL contours (outer + inner holes).

    RETR_CCOMP captures every individual stroke as its own contour, so
    nearby shapes are not merged into a single connected region.
    """
    cnts, hier = cv2.findContours(detail_mask, cv2.RETR_CCOMP,
                                  cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return 0
    attr = {'layer': layer}
    count = 0
    for cnt in cnts:
        if cv2.contourArea(cnt) < min_area:
            continue
        pts = simplify(cnt, eps_frac)
        pts_mm = px_to_mm(pts, scale, x0, y0)
        if len(pts_mm) >= 2:
            msp.add_lwpolyline(pts_mm, close=True, dxfattribs=attr)
            count += 1
    return count


def render_preview(img, blue_s, red_s, green_c, detail_mask, out_path):
    H, W = img.shape[:2]
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255

    # ── Details: all contours (outer + inner), drawn first (background) ─────
    cnts, _ = cv2.findContours(detail_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        if cv2.contourArea(cnt) < 30:
            continue
        pts = simplify(cnt, 0.005)
        cv2.polylines(canvas, [np.array(pts, np.int32).reshape(-1, 1, 2)],
                      True, (160, 160, 160), 1)

    # ── Walls: simplify + orthogonalize ───────────────────────────────────────
    cnts, _ = cv2.findContours(blue_s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        if cv2.contourArea(cnt) < 400:
            continue
        pts = orthogonalize(simplify(cnt, 0.001))
        cv2.polylines(canvas, [np.array(pts, np.int32).reshape(-1, 1, 2)],
                      True, (0, 0, 0), 2)

    # ── Doors: geometric arc + panel line ─────────────────────────────────────
    cnts, _ = cv2.findContours(red_s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        if cv2.contourArea(cnt) < 2000:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        roi = red_s[by:by+bh, bx:bx+bw]
        mh, mw = max(bh//2, 1), max(bw//2, 1)
        q = [int((roi[:mh, :mw] > 0).sum()), int((roi[:mh, mw:] > 0).sum()),
             int((roi[mh:, :mw] > 0).sum()), int((roi[mh:, mw:] > 0).sum())]
        eq = int(np.argmin(q))
        pivot_px = {0: (bx+bw, by+bh), 1: (bx, by+bh),
                    2: (bx+bw, by),    3: (bx, by)}[eq]
        sa, ea = {0: (90, 180), 1: (0, 90), 2: (180, 270), 3: (270, 360)}[eq]
        r_px = (bw + bh) / 2
        # Panel line (solid)
        panel_end = (int(pivot_px[0] + r_px * math.cos(math.radians(sa))),
                     int(pivot_px[1] - r_px * math.sin(math.radians(sa))))
        cv2.line(canvas, pivot_px, panel_end, (0, 0, 200), 2)
        # Dashed arc
        _draw_arc_cv_dashed(canvas, pivot_px, r_px, sa, ea, (0, 0, 200))

    # ── Windows: closed rect + centre line ───────────────────────────────────
    def _win_rect_cv(rx, ry, rw, rh):
        cv2.rectangle(canvas, (rx, ry), (rx+rw, ry+rh), (0, 160, 0), 2)
        if rw >= rh:
            cy = ry + rh // 2
            cv2.line(canvas, (rx, cy), (rx+rw, cy), (0, 160, 0), 2)
        else:
            cx = rx + rw // 2
            cv2.line(canvas, (cx, ry), (cx, ry+rh), (0, 160, 0), 2)

    cnts, _ = cv2.findContours(green_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        fill = area / (bw * bh)
        if fill < 0.20:
            continue
        if fill >= 0.65:
            _win_rect_cv(bx, by, bw, bh)
        else:
            roi = green_c[by:by+bh, bx:bx+bw]
            mh, mw = max(bh//2, 1), max(bw//2, 1)
            q = [int((roi[:mh,:mw]>0).sum()), int((roi[:mh,mw:]>0).sum()),
                 int((roi[mh:,:mw]>0).sum()), int((roi[mh:,mw:]>0).sum())]
            eq = int(np.argmin(q))
            t_h, t_v = _l_arm_thicknesses(roi)
            # Outer L-polygon
            if eq == 3:
                outer = [(bx,bw+bx),(bx+bw,by),(bx+bw,by+t_h),(bx+t_v,by+t_h),(bx+t_v,by+bh),(bx,by+bh)]
                outer = np.array([(bx,by),(bx+bw,by),(bx+bw,by+t_h),(bx+t_v,by+t_h),(bx+t_v,by+bh),(bx,by+bh)], np.int32)
                cline = np.array([(bx+bw,by+t_h//2),(bx+t_v//2,by+t_h//2),(bx+t_v//2,by+bh)], np.int32)
            elif eq == 2:
                outer = np.array([(bx,by),(bx+bw,by),(bx+bw,by+bh),(bx+bw-t_v,by+bh),(bx+bw-t_v,by+t_h),(bx,by+t_h)], np.int32)
                cline = np.array([(bx,by+t_h//2),((bx+bw-t_v//2),by+t_h//2),(bx+bw-t_v//2,by+bh)], np.int32)
            elif eq == 1:
                outer = np.array([(bx,by),(bx+t_v,by),(bx+t_v,by+bh-t_h),(bx+bw,by+bh-t_h),(bx+bw,by+bh),(bx,by+bh)], np.int32)
                cline = np.array([(bx+t_v//2,by),(bx+t_v//2,by+bh-t_h//2),(bx+bw,by+bh-t_h//2)], np.int32)
            else:
                outer = np.array([(bx+bw-t_v,by),(bx+bw,by),(bx+bw,by+bh),(bx,by+bh),(bx,by+bh-t_h),(bx+bw-t_v,by+bh-t_h)], np.int32)
                cline = np.array([(bx+bw-t_v//2,by),(bx+bw-t_v//2,by+bh-t_h//2),(bx,by+bh-t_h//2)], np.int32)
            cv2.polylines(canvas, [outer], True,  (0,160,0), 2)
            cv2.polylines(canvas, [cline], False, (0,160,0), 2)

    cv2.imwrite(out_path, canvas)
    print(f'Preview → {out_path}')


def main():
    ap = argparse.ArgumentParser(
        description='Color-annotated floor plan PNG → DXF')
    ap.add_argument('input', help='Input PNG (blue=walls, red=doors, green=windows)')
    ap.add_argument('-o', '--output', default=None, help='Output DXF path')
    ap.add_argument('--width-mm', type=float, default=None,
                    help='Real-world floor plan width in mm')
    ap.add_argument('--scale', type=float, default=None,
                    help='mm per pixel — overrides --width-mm')
    ap.add_argument('--debug', action='store_true',
                    help='Save debug mask PNGs')
    args = ap.parse_args()

    out_dxf = args.output or args.input.rsplit('.', 1)[0] + '.dxf'

    img = cv2.imread(args.input)
    if img is None:
        sys.exit(f'Cannot read: {args.input}')
    H, W = img.shape[:2]
    print(f'Image: {W}×{H} px')

    hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue  = cv2.inRange(hsv, BLUE_LO,  BLUE_HI)
    red   = cv2.bitwise_or(cv2.inRange(hsv, RED_LO1, RED_HI1),
                           cv2.inRange(hsv, RED_LO2, RED_HI2))
    green = cv2.inRange(hsv, GREEN_LO, GREEN_HI)

    # Scale from blue region extent
    cols = np.where(blue.any(axis=0))[0]
    rows = np.where(blue.any(axis=1))[0]
    if not len(cols):
        sys.exit('No blue pixels found — check colour thresholds.')

    x0 = int(cols[0])
    y0 = int(rows[-1])
    plan_px_w = int(cols[-1]) - x0
    plan_px_h = y0 - int(rows[0])
    plan_bbox = (x0, int(rows[0]), int(cols[-1]), y0)   # pixel bounds of floor plan

    if args.scale:
        scale = args.scale
    elif args.width_mm:
        scale = args.width_mm / plan_px_w
    else:
        print('Warning: no scale given — 1 px = 1 mm')
        scale = 1.0

    print(f'Scale: {scale:.4f} mm/px  →  '
          f'{plan_px_w * scale:.0f} × {plan_px_h * scale:.0f} mm')

    # Clean-up + smooth
    # Use small close kernel so door gaps in walls are preserved
    blue_c  = smooth_edges(morph_clean(blue,  close_k=3, open_k=3, close_iter=1), sigma=2.0)
    red_c   = smooth_edges(morph_clean(red,   close_k=3, open_k=3, close_iter=1), sigma=1.5)
    green_c = morph_clean(green, close_k=5, open_k=3, close_iter=2)

    # Detail mask: non-wall/door/window dark content — no smoothing to preserve
    # individual strokes; RETR_CCOMP captures inner contours too
    detail_mask = make_detail_mask(img, blue, red, green)

    if args.debug:
        base = args.input.rsplit('.', 1)[0]
        cv2.imwrite(base + '_dbg_blue.png',   blue_c)
        cv2.imwrite(base + '_dbg_red.png',    red_c)
        cv2.imwrite(base + '_dbg_green.png',  green_c)
        cv2.imwrite(base + '_dbg_detail.png', detail_mask)

    # Build DXF
    doc = ezdxf.new('R2010')
    doc.units = 4   # mm
    doc.header['$LTSCALE'] = 1
    msp = doc.modelspace()
    doc.layers.add('WALLS',   color=7,  lineweight=35)
    doc.layers.add('DOORS',   color=1,  lineweight=25)
    doc.layers.add('WINDOWS', color=4,  lineweight=25)
    doc.layers.add('DETAILS', color=8,  lineweight=13)   # gray, thin
    # Dense dashed linetype for door arcs (80 mm dash, 40 mm gap)
    _lt = doc.linetypes.new('DOOR_DASH', dxfattribs={'description': 'Door arc dashes'})
    _lt.setup_pattern([120.0, 80.0, -40.0])

    # Walls: small eps preserves door-jamb notches (only ~15px deep in mask)
    nw = draw_contours(msp, blue_c, 'WALLS', scale, x0, y0,
                       min_area=400, eps_frac=0.001,
                       retrieve=cv2.RETR_EXTERNAL, ortho=True)

    # Doors: geometric arc + straight panel line
    nd = add_doors_arc(msp, red_c, 'DOORS', scale, x0, y0)

    # Windows: closed rectangle outline + centre line
    nwin = add_windows_rect(msp, green_c, 'WINDOWS', scale, x0, y0)

    # Details: everything else, vectorized as-is
    ndet = add_detail_layer(msp, detail_mask, 'DETAILS', scale, x0, y0)

    print(f'Extracted: walls={nw}, doors={nd}, windows={nwin}, details={ndet}')
    doc.saveas(out_dxf)
    print(f'Saved: {out_dxf}')

    prev_path = out_dxf.rsplit('.', 1)[0] + '_preview.png'
    render_preview(img, blue_c, red_c, green_c, detail_mask, prev_path)


if __name__ == '__main__':
    main()
