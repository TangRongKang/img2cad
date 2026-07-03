#!/usr/bin/env python3
"""Debug door extraction from color-annotated floor plan."""
import sys
from pathlib import Path
import cv2
import numpy as np
import math

sys.path.insert(0, str(Path(__file__).parent))
from color_annotated_to_dxf import (
    BLUE_LO, BLUE_HI, RED_LO1, RED_HI1, RED_LO2, RED_HI2,
    GREEN_LO, GREEN_HI, morph_clean, trace_doors, _door_axes,
    _snap_dir45, _perp_thickness, _build_door
)


def main():
    img_path = Path('output_test/test_annotated_1.png')
    if not img_path.exists():
        img_path = Path('output_test/test_image_06_annotated.png')
    if not img_path.exists():
        img_path = Path('image/test_image_06.png')

    img = cv2.imread(str(img_path))
    if img is None:
        sys.exit(f'Cannot read {img_path}')

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, BLUE_LO, BLUE_HI)
    red = cv2.bitwise_or(cv2.inRange(hsv, RED_LO1, RED_HI1),
                         cv2.inRange(hsv, RED_LO2, RED_HI2))
    green = cv2.inRange(hsv, GREEN_LO, GREEN_HI)

    blue_w = morph_clean(blue, close_k=3, open_k=3, close_iter=1)
    blue_c = morph_clean(blue, close_k=3, open_k=3, close_iter=1)
    red_o = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    red_c = cv2.morphologyEx(red_o, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=2)
    green_c = morph_clean(green, close_k=5, open_k=3, close_iter=2)

    struct = cv2.bitwise_or(blue_w, green_c)

    wall_t = 25  # approximate
    doors = trace_doors(red_c, blue_w, struct, wall_t)

    print(f'Doors detected: {len(doors)}')

    canvas = img.copy()
    for i, dr in enumerate(doors):
        color = (0, 255, 255)
        pts = np.array(dr['opening'], np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, color, 2)
        pts = np.array(dr['leaf'], np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, (255, 0, 255), 2)
        pts = np.array(dr['arc'], np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], False, (0, 0, 255), 2)

    out_dir = Path('output_test/debug')
    out_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir / 'doors_debug.png'), canvas)
    cv2.imwrite(str(out_dir / 'red_mask.png'), red_c)
    print(f'Saved door debug to {out_dir}')


if __name__ == '__main__':
    main()
