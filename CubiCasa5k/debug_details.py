#!/usr/bin/env python3
"""Debug detail extraction from color-annotated floor plan."""
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from color_annotated_to_dxf import (
    BLUE_LO, BLUE_HI, RED_LO1, RED_HI1, RED_LO2, RED_HI2,
    GREEN_LO, GREEN_HI, make_detail_mask, trace_details, morph_clean
)


def main():
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

    detail_mask = make_detail_mask(img, blue, red, green)
    details = trace_details(detail_mask)

    print(f'Detail contours: {len(details)}')

    # Draw extracted polylines on a white canvas
    canvas = np.ones_like(img) * 255
    for i, poly in enumerate(details):
        if len(poly) < 2:
            continue
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        color = (0, 0, 255) if len(poly) >= 3 else (255, 0, 0)
        cv2.polylines(canvas, [pts], True, color, 1)
        # mark first vertex
        cv2.circle(canvas, tuple(pts[0][0]), 2, (0, 255, 0), -1)

    out_dir = Path('output_test/debug')
    out_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir / 'detail_mask.png'), detail_mask)
    cv2.imwrite(str(out_dir / 'detail_contours.png'), canvas)
    print(f'Saved debug images to {out_dir}')


if __name__ == '__main__':
    main()
