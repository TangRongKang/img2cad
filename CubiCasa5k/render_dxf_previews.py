#!/usr/bin/env python3
"""
Render all DXF files in a folder as preview PNGs.
Each layer is drawn in the RGB colour encoded in its name (e.g. Layer_R82_G94_B156).
Output: <name>_preview.png next to each DXF.
"""
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import ezdxf


def parse_layer_color(name):
    m = re.search(r'R(\d+)_G(\d+)_B(\d+)', name)
    if m:
        r, g, b = map(int, m.groups())
        return (b, g, r)  # OpenCV BGR
    if name in ('Defpoints',):
        return None
    return (0, 0, 0)


def iter_points(entity):
    """Yield (x, y) points from LWPOLYLINE / LINE / ARC-like entities."""
    dxftype = entity.dxftype()
    if dxftype == 'LWPOLYLINE':
        pts = list(entity.vertices_in_wcs())
        # vertices_in_wcs returns (x,y,z) for some versions; trim to 2D
        pts = [(p[0], p[1]) for p in pts]
        if entity.closed and pts:
            pts.append(pts[0])
        return pts
    if dxftype == 'LINE':
        s = entity.dxf.start
        e = entity.dxf.end
        return [(s[0], s[1]), (e[0], e[1])]
    if dxftype == 'ARC':
        c = entity.dxf.center
        r = entity.dxf.radius
        sa = np.radians(entity.dxf.start_angle)
        ea = np.radians(entity.dxf.end_angle)
        if ea < sa:
            ea += 2 * np.pi
        n = max(3, int(np.degrees(ea - sa) / 2))
        pts = []
        for a in np.linspace(sa, ea, n + 1):
            pts.append((c[0] + r * np.cos(a), c[1] + r * np.sin(a)))
        return pts
    if dxftype == 'CIRCLE':
        c = entity.dxf.center
        r = entity.dxf.radius
        n = 64
        pts = [(c[0] + r * np.cos(a), c[1] + r * np.sin(a))
               for a in np.linspace(0, 2 * np.pi, n + 1)]
        return pts
    return []


def render_dxf(dxf_path, out_path, size=2400, margin=40):
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    all_pts = []
    polys = []
    for ent in msp:
        pts = iter_points(ent)
        if len(pts) < 2:
            continue
        layer = ent.dxf.layer
        color = parse_layer_color(layer)
        if color is None:
            continue
        all_pts.extend(pts)
        polys.append((np.array(pts, dtype=np.float64), color))

    if not all_pts:
        print(f'No drawable entities in {dxf_path.name}')
        return

    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    w = max_x - min_x
    h = max_y - min_y
    if w == 0 or h == 0:
        print(f'Zero extent in {dxf_path.name}')
        return

    scale = min((size - 2 * margin) / w, (size - 2 * margin) / h)

    # Fit to canvas while preserving aspect ratio and adding margin
    img_w = int(w * scale) + 2 * margin
    img_h = int(h * scale) + 2 * margin
    canvas = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

    def to_px(x, y):
        px = int((x - min_x) * scale) + margin
        py = int((max_y - y) * scale) + margin
        return (px, py)

    for pts, color in polys:
        pts_px = np.array([to_px(x, y) for x, y in pts], dtype=np.int32).reshape(-1, 1, 2)
        closed = bool(np.allclose(pts[0], pts[-1]))
        cv2.polylines(canvas, [pts_px], closed, color, 2, cv2.LINE_AA)

    cv2.imwrite(str(out_path), canvas)
    print(f'{dxf_path.name} → {out_path.name}  ({img_w}x{img_h})')


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('jingpin')
    if not folder.is_dir():
        sys.exit(f'Folder not found: {folder}')
    for dxf in sorted(folder.glob('*.dxf')):
        out = dxf.parent / (dxf.stem + '_preview.png')
        render_dxf(dxf, out)


if __name__ == '__main__':
    main()
