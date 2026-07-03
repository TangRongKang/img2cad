#!/usr/bin/env python3
"""
Convert floor plan image to DXF using CubiCasa5K pretrained model.

Usage:
    conda run -n cubicasa python image_to_dxf.py input.png output.dxf \
        [--weights floortrans/models/model_best_val_loss_var.pkl] \
        [--width-mm 13500] [--height-mm 11500] \
        [--size 512] [--threshold 0.2]
"""
import sys
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import ezdxf
from shapely.geometry import (Polygon as SPoly, box as SBox, MultiPolygon,
                               GeometryCollection, LineString, MultiLineString)
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from floortrans.models import get_model
from floortrans.post_prosessing import split_prediction, get_polygons
from floortrans.loaders import RotateNTurns

ROOM_CLASSES = [
    "Background", "Outdoor", "Wall", "Kitchen", "Living Room",
    "Bed Room", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined",
]
ICON_CLASSES = [
    "No Icon", "Window", "Door", "Closet", "Electrical Appliance",
    "Toilet", "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney",
]

DOUBLE_DOOR_THRESHOLD_MM = 1200
INTERIOR_CLASSES = {3, 4, 5, 6, 7, 9, 10, 11}


# ─── Model ───────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_model(weights_path, device):
    model = get_model('hg_furukawa_original', 51)
    n_classes = 44
    model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    model.to(device)
    return model


def preprocess_image(img_path, target_size):
    img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = img.size

    if target_size == 0:
        # Native resolution: round each dimension to nearest multiple of 4, no squash
        new_w = (orig_w // 4) * 4
        new_h = (orig_h // 4) * 4
        if (new_w, new_h) != (orig_w, orig_h):
            img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f'  native resolution: {orig_w}×{orig_h} → {new_w}×{new_h}')
        arr = np.array(img, dtype=np.float32) / 255.0 * 2.0 - 1.0
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
        pad_info = (0, 0, new_w, new_h)
        return tensor, (new_h, new_w), (orig_h, orig_w), pad_info

    size = (target_size // 4) * 4

    # Maintain aspect ratio: scale to fit within size×size, pad remainder with white
    scale = min(size / orig_w, size / orig_h)
    new_w = (int(orig_w * scale) // 4) * 4
    new_h = (int(orig_h * scale) // 4) * 4
    new_w = min(new_w, size)
    new_h = min(new_h, size)

    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    padded = Image.new('RGB', (size, size), (255, 255, 255))
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    padded.paste(img_resized, (pad_x, pad_y))

    arr = np.array(padded, dtype=np.float32) / 255.0 * 2.0 - 1.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    pad_info = (pad_x, pad_y, new_w, new_h)
    print(f'  aspect-ratio padding: {orig_w}×{orig_h} → {new_w}×{new_h} '
          f'(pad {pad_x},{pad_y}) in {size}×{size}')
    return tensor, (size, size), (orig_h, orig_w), pad_info


def run_inference(model, image_tensor, img_size, device, n_classes=44):
    rot = RotateNTurns()
    image_tensor = image_tensor.to(device)
    height, width = img_size
    prediction = torch.zeros([4, n_classes, height, width])
    with torch.no_grad():
        for i, (forward, back) in enumerate([(0, 0), (1, -1), (2, 2), (-1, 1)]):
            rot_image = rot(image_tensor, 'tensor', forward)
            pred = model(rot_image)
            pred = rot(pred, 'tensor', back)
            pred = rot(pred, 'points', back)
            pred = F.interpolate(pred, size=img_size, mode='bilinear', align_corners=True)
            prediction[i] = pred[0].cpu()
    return torch.mean(prediction, 0, True)


# ─── DXF drawing helpers ─────────────────────────────────────────────────────

def _add_hatch(msp, pts, layer, color=7, transparency=0.0):
    h = msp.add_hatch(color=color, dxfattribs={'layer': layer})
    h.set_pattern_fill('SOLID')
    if transparency > 0.0:
        h.transparency = transparency
    h.paths.add_polyline_path(pts + [pts[0]], is_closed=True)
    return h


_MIN_WALL_THICK_MM = 180.0   # enforce minimum wall/opening thickness


def _snap_to_wall(x1, y1, x2, y2, wall_shapes_mm):
    """Expand opening bbox to the full thickness of the nearest wall."""
    win = SBox(x1, y1, x2, y2)
    best_area = 0
    best = None
    for ws, wx1, wy1, wx2, wy2 in wall_shapes_mm:
        a = win.intersection(ws).area
        if a > best_area:
            best_area = a
            best = (wx1, wy1, wx2, wy2)
    if best is None:
        # No wall found — enforce minimum thickness and return
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        if h >= w:  # vertical opening
            half = max(w, _MIN_WALL_THICK_MM) / 2
            return cx - half, y1, cx + half, y2
        else:
            half = max(h, _MIN_WALL_THICK_MM) / 2
            return x1, cy - half, x2, cy + half

    wx1, wy1, wx2, wy2 = best
    if (wy2 - wy1) >= (wx2 - wx1):  # vertical wall → snap x extents
        rx1, ry1, rx2, ry2 = wx1, y1, wx2, y2
    else:                             # horizontal wall → snap y extents
        rx1, ry1, rx2, ry2 = x1, wy1, x2, wy2

    # Enforce minimum thickness
    thick = min(rx2 - rx1, ry2 - ry1)
    if thick < _MIN_WALL_THICK_MM:
        cx, cy = (rx1 + rx2) / 2, (ry1 + ry2) / 2
        if (ry2 - ry1) >= (rx2 - rx1):  # vertical
            half = _MIN_WALL_THICK_MM / 2
            rx1, rx2 = cx - half, cx + half
        else:                              # horizontal
            half = _MIN_WALL_THICK_MM / 2
            ry1, ry2 = cy - half, cy + half
    return rx1, ry1, rx2, ry2


def _draw_window(msp, x1, y1, x2, y2):
    """Architectural window symbol matching reference style.

    3 parallel lines at wall-face positions + center glass line,
    plus 2 perpendicular jamb lines at each end (the wall-window boundary).
    All on WINDOWS layer, same lineweight as walls.
    """
    WA = {'layer': 'WINDOWS'}
    w, h = x2 - x1, y2 - y1

    if w >= h:   # horizontal window in horizontal wall
        # 3 horizontal lines: outer face, center glass, inner face
        msp.add_line((x1, y1),        (x2, y1),        dxfattribs=WA)
        msp.add_line((x1, (y1+y2)/2), (x2, (y1+y2)/2), dxfattribs=WA)
        msp.add_line((x1, y2),        (x2, y2),        dxfattribs=WA)
        # 2 vertical jamb lines (window-end boundary markers)
        msp.add_line((x1, y1), (x1, y2), dxfattribs=WA)
        msp.add_line((x2, y1), (x2, y2), dxfattribs=WA)
    else:        # vertical window in vertical wall
        # 3 vertical lines: left face, center glass, right face
        msp.add_line((x1, y1),        (x1, y2),        dxfattribs=WA)
        msp.add_line(((x1+x2)/2, y1), ((x1+x2)/2, y2), dxfattribs=WA)
        msp.add_line((x2, y1),        (x2, y2),        dxfattribs=WA)
        # 2 horizontal jamb lines (window-end boundary markers)
        msp.add_line((x1, y1), (x2, y1), dxfattribs=WA)
        msp.add_line((x1, y2), (x2, y2), dxfattribs=WA)


def _draw_door(msp, x1, y1, x2, y2, swing_pos=True, arc_lt='DOOR_ARC'):
    """单开门：门板矩形（实线）+ 四分之一弧（虚线）。

    铰链：水平门固定在左端 x1，竖直门固定在右端 x2。
    swing_pos: 水平门 True=向+Y开，False=向-Y开
               竖直门 True=向+X开，False=向-X开
    """
    w, h = x2 - x1, y2 - y1
    DS = {'layer': 'DOORS'}
    DA = {'layer': 'DOORS', 'linetype': arc_lt}

    def rect(pts):
        msp.add_lwpolyline(pts, close=True, dxfattribs=DS)

    def arc(cx, cy, r, a0, a1):
        msp.add_arc(center=(cx, cy), radius=r,
                    start_angle=a0, end_angle=a1, dxfattribs=DA)

    if w >= h:   # ── 水平门 ────────────────────────────────────────────────────
        r = w
        t = max(40.0, min(80.0, r * 0.08))
        if swing_pos:   # 向 +Y 开，铰链贴上侧墙面 y2
            hy = y2
            rect([(x1, hy), (x1+t, hy), (x1+t, hy+r), (x1, hy+r)])
            arc(x1, hy, r, 0, 90)
        else:           # 向 -Y 开，铰链贴下侧墙面 y1
            hy = y1
            rect([(x1, hy), (x1+t, hy), (x1+t, hy-r), (x1, hy-r)])
            arc(x1, hy, r, 270, 360)

    else:        # ── 竖直门 ────────────────────────────────────────────────────
        r = h
        t = max(40.0, min(80.0, r * 0.08))
        if swing_pos:   # 向 +X 开，铰链贴右侧墙面 x2
            hx = x2
            rect([(hx, y1), (hx+r, y1), (hx+r, y1+t), (hx, y1+t)])
            arc(hx, y1, r, 0, 90)
        else:           # 向 -X 开，铰链贴左侧墙面 x1
            hx = x1
            rect([(hx-r, y1), (hx, y1), (hx, y1+t), (hx-r, y1+t)])
            arc(hx, y1, r, 90, 180)


def _determine_swing_brightness(ix1, iy1, ix2, iy2, gray, orig_h, orig_w, sx, sy):
    """根据原图亮度判断门的开启方向。

    亮的一侧 = 室内空间，门向室内开。
    返回 True → 向 +DXF-Y（水平门）或 +DXF-X（竖直门）开。
    """
    ox1 = int(np.clip(ix1 * sx, 0, orig_w - 1))
    ox2 = int(np.clip(ix2 * sx, 0, orig_w - 1))
    oy1 = int(np.clip(iy1 * sy, 0, orig_h - 1))
    oy2 = int(np.clip(iy2 * sy, 0, orig_h - 1))
    ow = max(1, ox2 - ox1)
    oh = max(1, oy2 - oy1)

    def strip_mean(r0, r1, c0, c1):
        r0 = int(np.clip(r0, 0, orig_h - 1))
        r1 = int(np.clip(r1, 0, orig_h - 1))
        c0 = int(np.clip(c0, 0, orig_w - 1))
        c1 = int(np.clip(c1, 0, orig_w - 1))
        if r0 >= r1 or c0 >= c1:
            return 128.0
        return float(gray[r0:r1, c0:c1].mean())

    if ow >= oh:   # 水平门：上下各采样一条横带
        dist = max(int(oh * 1.5), 30)
        sh   = max(oh, 20)
        oy_c = (oy1 + oy2) // 2
        above = strip_mean(oy_c - dist - sh, oy_c - dist, ox1, ox2)
        below = strip_mean(oy_c + dist,      oy_c + dist + sh, ox1, ox2)
        # 图像上方（小 oy）= DXF +Y → swing_pos=True
        return above >= below
    else:          # 竖直门：左右各采样一条纵带
        dist = max(int(ow * 1.5), 30)
        sw   = max(ow, 20)
        ox_c = (ox1 + ox2) // 2
        right = strip_mean(oy1, oy2, ox_c + dist,      ox_c + dist + sw)
        left  = strip_mean(oy1, oy2, ox_c - dist - sw, ox_c - dist)
        # 图像右侧 = DXF +X → swing_pos=True
        return right >= left




def _draw_detail_lines_hires(msp, img_path, orig_size, polygons, model_img_size,
                              width_mm, height_mm):
    """Extract all interior detail geometry (furniture, stairs, fixtures) via
    contour tracing — captures curves (round tables, sofas) as well as lines."""
    import cv2

    orig_h, orig_w = orig_size
    model_h, model_w = model_img_size
    scale_x = orig_w / model_w
    scale_y = orig_h / model_h
    sx = width_mm  / orig_w
    sy = height_mm / orig_h

    pil = Image.open(img_path).convert('RGB')
    if pil.size != (orig_w, orig_h):
        pil = pil.resize((orig_w, orig_h), Image.LANCZOS)
    img_bgr = np.array(pil)[:, :, ::-1].copy()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Dark lines → 255
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Exclude wall / icon polygon regions
    pad = max(12, int(min(orig_w, orig_h) * 0.012))
    excl = np.zeros((orig_h, orig_w), dtype=np.uint8)
    for pol in polygons:
        x1 = max(0,        int(min(pol[:, 0]) * scale_x) - pad)
        y1 = max(0,        int(min(pol[:, 1]) * scale_y) - pad)
        x2 = min(orig_w-1, int(max(pol[:, 0]) * scale_x) + pad)
        y2 = min(orig_h-1, int(max(pol[:, 1]) * scale_y) + pad)
        excl[y1:y2+1, x1:x2+1] = 255
    binary[excl > 0] = 0

    # Canny on slightly blurred gray → 1-px-thin edges (avoids tracing both
    # sides of a thick stroke, which doubles every line in the output)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(blurred, 40, 120)
    canny[excl > 0] = 0
    # Small close to reconnect slightly broken strokes
    kernel_c = np.ones((2, 2), np.uint8)
    canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel_c, iterations=1)

    # Building footprint — exclude dimension annotations outside the walls
    all_xs = np.concatenate([pol[:, 0] for pol in polygons]) * scale_x
    all_ys = np.concatenate([pol[:, 1] for pol in polygons]) * scale_y
    fp_pad = pad * 3
    bldg_x1 = float(all_xs.min()) - fp_pad
    bldg_x2 = float(all_xs.max()) + fp_pad
    bldg_y1 = float(all_ys.min()) - fp_pad
    bldg_y2 = float(all_ys.max()) + fp_pad

    def to_dxf(x, y):
        return float(x) * sx, (orig_h - float(y)) * sy

    # Contour tracing on Canny edges: paths follow line centers, not stroke edges
    contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)

    # Reject contours shorter than ~1.5 % of image width (text strokes / specks)
    min_peri = max(15, int(orig_w * 0.015))

    count = 0
    for cnt in contours:
        peri = cv2.arcLength(cnt, False)
        if peri < min_peri:
            continue

        pts_xy = cnt[:, 0, :]
        cx = float(pts_xy[:, 0].mean())
        cy = float(pts_xy[:, 1].mean())
        if not (bldg_x1 <= cx <= bldg_x2 and bldg_y1 <= cy <= bldg_y2):
            continue

        # Simplify while preserving curve shape
        epsilon = max(1.5, peri * 0.004)
        approx = cv2.approxPolyDP(cnt, epsilon, closed=False)
        if len(approx) < 2:
            continue

        pts_dxf = [to_dxf(p[0][0], p[0][1]) for p in approx]
        is_closed = (np.linalg.norm(
            np.array(approx[0][0], dtype=float) -
            np.array(approx[-1][0], dtype=float)) < 5)
        msp.add_lwpolyline(pts_dxf, close=is_closed,
                           dxfattribs={'layer': 'DETAILS'})
        count += 1

    return count


def _iter_polygons(geom):
    if isinstance(geom, SPoly):
        yield geom
    elif isinstance(geom, (MultiPolygon, GeometryCollection)):
        for part in geom.geoms:
            yield from _iter_polygons(part)


# ─── Main conversion ─────────────────────────────────────────────────────────

def polygons_to_dxf(polygons, types, room_polygons, room_types,
                    img_size, output_path, width_mm, height_mm,
                    img_path=None, orig_size=None, pad_info=None):
    img_h, img_w = img_size

    # Coordinate mapping: model pixel → DXF mm, accounting for aspect-ratio padding
    if pad_info is not None:
        _pad_x, _pad_y, _new_w, _new_h = pad_info
        def px(x, y):
            return ((x - _pad_x) * width_mm / _new_w,
                    (_new_h - (y - _pad_y)) * height_mm / _new_h)
        sx = width_mm  / _new_w   # used only for detail layer scaling
        sy = height_mm / _new_h
    else:
        sx = width_mm / img_w
        sy = height_mm / img_h
        def px(x, y):
            return x * sx, (img_h - y) * sy

    # 加载原图灰度，用于亮度判断门的开启方向
    _gray = _orig_h = _orig_w = _gsx = _gsy = _gpx = _gpy = None
    if img_path and orig_size:
        _orig_h, _orig_w = orig_size
        _pil = Image.open(img_path).convert('L')
        if _pil.size != (_orig_w, _orig_h):
            _pil = _pil.resize((_orig_w, _orig_h), Image.LANCZOS)
        _gray = np.array(_pil)
        if pad_info is not None:
            _pad_x2, _pad_y2, _new_w2, _new_h2 = pad_info
            _gsx = _orig_w / _new_w2
            _gsy = _orig_h / _new_h2
            _gpx, _gpy = _pad_x2, _pad_y2
        else:
            _gsx = _orig_w / img_w
            _gsy = _orig_h / img_h
            _gpx, _gpy = 0, 0

    def pol_mm(pol):
        return [px(p[0], p[1]) for p in pol]

    def bbox(pts):
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)

    # ── Document setup ────────────────────────────────────────────────
    doc = ezdxf.new('R2018')
    doc.header['$INSUNITS'] = 4
    msp = doc.modelspace()

    doc.layers.add('WALLS',    color=7,  lineweight=35)
    doc.layers.add('WINDOWS',  color=4,  lineweight=35)
    doc.layers.add('DOORS',    color=1,  lineweight=35)
    doc.layers.add('FIXTURES', color=6,  lineweight=18)
    doc.layers.add('ROOMS',    color=3,  lineweight=18)
    doc.layers.add('DETAILS',  color=8,  lineweight=13)  # dark gray

    # 门弧虚线：dash=300mm，gap=150mm
    ARC_LT = 'DOOR_ARC'
    if ARC_LT not in doc.linetypes:
        doc.linetypes.add(ARC_LT, [300, 200, -100])

    # ── Pre-compute wall shapes in mm (needed for opening snap) ──────
    wall_shapes_mm = []
    for pol, t in zip(polygons, types):
        if t['type'] == 'wall':
            pts = pol_mm(pol)
            wx1, wy1, wx2, wy2 = bbox(pts)
            wall_shapes_mm.append((SBox(wx1, wy1, wx2, wy2), wx1, wy1, wx2, wy2))

    # ── Classify icon polygons (snap openings to full wall thickness) ─
    opening_shapes = []   # (SBox_mm, cls, x1,y1,x2,y2 mm, ix1,iy1,ix2,iy2 px)
    icon_others    = []

    for pol, t in zip(polygons, types):
        if t['type'] != 'icon':
            continue
        cls = int(t['class'])
        ix1, iy1 = int(min(pol[:,0])), int(min(pol[:,1]))
        ix2, iy2 = int(max(pol[:,0])), int(max(pol[:,1]))
        pts = pol_mm(pol)
        x1, y1, x2, y2 = bbox(pts)
        if cls in (1, 2):
            x1, y1, x2, y2 = _snap_to_wall(x1, y1, x2, y2, wall_shapes_mm)
            opening_shapes.append((SBox(x1,y1,x2,y2), cls,
                                   x1, y1, x2, y2, ix1, iy1, ix2, iy2))
        else:
            icon_others.append((pts, cls))

    opening_union = unary_union([s[0] for s in opening_shapes]) if opening_shapes else None
    door_union    = (unary_union([s[0] for s in opening_shapes if s[1] == 2])
                     if any(s[1] == 2 for s in opening_shapes) else None)

    room_count = 0  # ROOMS layer disabled

    # ── 2. Walls: merge all → subtract doors → draw outline ───────────
    # unary_union handles all T/L/cross junctions cleanly — no crossing lines
    wall_count = 0
    if wall_shapes_mm:
        merged = unary_union([ws[0] for ws in wall_shapes_mm])
        if door_union is not None and not door_union.is_empty:
            try:
                merged = merged.difference(door_union)
            except Exception:
                pass
        for geom in _iter_polygons(merged):
            try:
                coords = list(geom.exterior.coords)
                if len(coords) >= 3:
                    msp.add_lwpolyline(coords, close=True,
                                       dxfattribs={'layer': 'WALLS'})
                    wall_count += 1
                for interior in geom.interiors:
                    ic = list(interior.coords)
                    if len(ic) >= 3:
                        msp.add_lwpolyline(ic, close=True,
                                           dxfattribs={'layer': 'WALLS'})
            except Exception:
                pass

    # ── 3. Windows & Doors ─────────────────────────────────────────────
    win_count = door_count = 0
    for entry in opening_shapes:
        _, cls, x1, y1, x2, y2, ix1, iy1, ix2, iy2 = entry
        if cls == 1:
            _draw_window(msp, x1, y1, x2, y2)
            win_count += 1
        elif cls == 2:
            swing = True
            if _gray is not None:
                swing = _determine_swing_brightness(
                    ix1 - _gpx, iy1 - _gpy, ix2 - _gpx, iy2 - _gpy,
                    _gray, _orig_h, _orig_w, _gsx, _gsy)
            _draw_door(msp, x1, y1, x2, y2, swing_pos=swing, arc_lt=ARC_LT)
            door_count += 1

    # ── 4. Other fixtures ──────────────────────────────────────────────
    fix_count = 0
    for pts, cls in icon_others:
        msp.add_lwpolyline(pts, close=True, dxfattribs={'layer': 'FIXTURES'})
        fix_count += 1

    # ── 5. Detail lines from original high-res image ───────────────────
    detail_count = 0
    # Detail layer disabled — focus on walls/windows/doors first

    doc.saveas(output_path)
    print(f'Saved  -> {output_path}')
    print(f'  walls={wall_count}  windows={win_count}  doors={door_count}'
          f'  fixtures={fix_count}  rooms={room_count}  details={detail_count}')
    print(f'  scale: 1px = {sx:.2f}mm × {sy:.2f}mm')


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Floor plan image → DXF')
    p.add_argument('input',  help='Input image path')
    p.add_argument('output', nargs='?', default=None,
                   help='Output DXF path (default: same dir/name as input with .dxf)')
    p.add_argument('--weights',    default='floortrans/models/model_best_val_loss_var.pkl')
    p.add_argument('--width-mm',   type=float, default=13500)
    p.add_argument('--height-mm',  type=float, default=11500)
    p.add_argument('--size',       type=int,   default=768,
                   help='Inference size (0 = native resolution, no resize)')
    p.add_argument('--threshold',  type=float, default=0.2)
    args = p.parse_args()
    if args.output is None:
        import pathlib
        args.output = str(pathlib.Path(args.input).with_suffix('.dxf'))

    device = get_device()
    print(f'Device: {device}')

    print('Loading model ...')
    model = load_model(args.weights, device)

    print(f'Preprocessing → {args.size}px ...')
    image_tensor, img_size, orig_size, pad_info = preprocess_image(args.input, args.size)
    print(f'  inference {img_size[1]}×{img_size[0]}  original {orig_size[1]}×{orig_size[0]}')

    print('Running inference (4-rotation TTA) ...')
    prediction = run_inference(model, image_tensor, img_size, device)

    print('Post-processing ...')
    heatmaps, rooms, icons = split_prediction(prediction, img_size, [21, 12, 11])
    polygons, types, room_polygons, room_types = get_polygons(
        (heatmaps, rooms, icons), args.threshold, [1, 2]
    )
    print(f'  {len(polygons)} polygons, {len(room_polygons)} rooms')

    print('Writing DXF ...')
    polygons_to_dxf(
        polygons, types, room_polygons, room_types,
        img_size, args.output, args.width_mm, args.height_mm,
        img_path=args.input,
        orig_size=orig_size,
        pad_info=pad_info,
    )


if __name__ == '__main__':
    main()
