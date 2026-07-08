#!/usr/bin/env python3
"""
floorplan_to_dxf.py

端到端户型图 → DXF 转换 pipeline：
    普通黑白/灰度户型图 → 知末 AI 彩色标注 → DXF

Usage:
    python floorplan_to_dxf.py input.png --width-mm 13700
    python floorplan_to_dxf.py input.png --width-mm 13700 --output-dir ./out
    python floorplan_to_dxf.py input.png --annotated input_annotated.png --width-mm 13700
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
from PIL import Image

# 项目内模块
from znzmo_client import ZnzmoClient
from color_annotated_to_dxf import convert_annotated_image

# 知末 AI 彩色标注提示词（通用，不针对单张图打补丁）：
#   ① 原地重新着色，绝不重绘/简化/增删几何；
#   ② 墙/窗 → 蓝/绿实心，门与其它图元一律保留为黑线；
#   ③ 所有墙（含最外圈外轮廓）都必须着色；
#   ④ 显式处理深色/黑底 CAD 源图。
ANNOTATE_PROMPT = (
    "对这张户型图进行『原地重新着色』：严格保持原图的几何结构、比例和视角完全不变，"
    "只改变颜色，绝不重绘、简化、移动、新增或删除任何结构或图元。规则如下：\n"
    "1. 墙体 → 蓝色实心填充：所有承重墙、隔墙，以及建筑最外圈的外轮廓墙，"
    "全部都要填成蓝色，不得遗漏，不得保留为黑色或其它颜色。\n"
    "2. 窗 → 绿色实心填充：所有窗户。\n"
    "3. 门不要单独标注颜色，保持为白底黑色线条，与家具、洁具等其它图元一同处理。\n"
    "4. 其余所有图元（家具、洁具、橱柜、楼梯、电梯、家电、阳台构件等）一律保留为"
    "白底黑色线条，必须逐一完整保留，不得删除或简化。\n"
    "5. 只删除文字类信息：房间名称、尺寸标注、标高、轴号、引线、图框、标题栏、"
    "指北针、水印等文字和数字；不要删除任何图形线条。\n"
    "6. 输出为白色背景；若原图是深色/黑色背景的 CAD 图，转为白底后仍须逐一保留全部"
    "原始线条与图元，再按上述规则着色。\n"
    "7. 输出与原图同尺寸、像素对齐——这是对原图重新着色，而不是重新绘制一张图。"
)


def get_annotated_image(args):
    """返回标注图路径（已存在的直接用，否则调用知末 API 生成）。"""
    if args.annotated:
        annotated_path = Path(args.annotated)
        if not annotated_path.exists():
            sys.exit(f'Annotated image not found: {annotated_path}')
        print(f'Using existing annotated image: {annotated_path}')
        return str(annotated_path)

    if not args.api_url:
        sys.exit('Error: --api-url is required when --annotated is not provided.')

    client = ZnzmoClient(
        base_url=args.api_url,
        api_key=args.api_key or os.getenv('ZNZMO_IMAGE_API_KEY', ''),
        server_ip=args.server_ip,
        timeout=args.timeout,
    )

    input_path = Path(args.input)
    ref_img = Image.open(input_path).convert('RGB')
    input_size = ref_img.size  # (W, H)

    print(f'Calling Znzmo API to annotate: {input_path.name}')
    images, message = client.generate_image(
        model=args.model,
        prompt=ANNOTATE_PROMPT,
        aspect_ratio='auto',
        image_size='4k',
        batch_size=1,
        image_list=[ref_img],
    )

    if not images:
        sys.exit('Znzmo returned no images.')

    ann_img = images[0]
    # Keep the AI-returned high-res image for conversion.  Scale is recomputed
    # from the annotated image's own pixel width, so real-world dimensions stay
    # correct while benefiting from the higher resolution.
    print(f'  AI returned annotated image size: {ann_img.size}')

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / f'{input_path.stem}_annotated.png'
    ann_img.save(annotated_path, 'PNG')
    print(f'Saved annotated image: {annotated_path}  (msg={message})')
    return str(annotated_path)


def main():
    ap = argparse.ArgumentParser(description='Floor plan image → DXF (via AI color annotation)')
    ap.add_argument('input', help='Input floor plan image (PNG/JPG)')
    ap.add_argument('-o', '--output-dir', default=None,
                    help='Output directory (default: same as input)')
    ap.add_argument('--annotated', default=None,
                    help='Use existing color-annotated image instead of calling API')
    ap.add_argument('--width-mm', type=float, default=13700.0,
                    help='Real-world floor plan width in mm (default: 13700)')
    ap.add_argument('--scale', type=float, default=None,
                    help='mm per pixel (alternative to --width-mm)')

    # Znzmo API 参数
    ap.add_argument('--api-url',
                    default='https://api.znzmo.cn/ai-draw/third-api/ai-draw-api/dispatch/getAgentResult',
                    help='Znzmo API endpoint')
    ap.add_argument('--api-key', default='',
                    help='Znzmo API key (or set ZNZMO_IMAGE_API_KEY env var)')
    ap.add_argument('--server-ip', default='',
                    help='Override API domain resolution to this IP')
    ap.add_argument('--timeout', type=int, default=180,
                    help='API timeout in seconds (0 = no timeout)')
    ap.add_argument('--model', default='nanoBanana',
                    help='Znzmo model name')

    ap.add_argument('--no-details', action='store_true',
                    help='Skip the DETAILS layer in final DXF')
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f'Input not found: {input_path}')

    # 1. 获取/生成彩色标注图
    annotated_path = get_annotated_image(args)

    # 2. 确定输出 DXF 路径
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dxf = output_dir / f'{input_path.stem}.dxf'

    # 3. 转换为 DXF
    print(f'Converting annotated image to DXF: {output_dxf}')
    img_bgr = cv2.imread(annotated_path)
    if img_bgr is None:
        sys.exit(f'Cannot read annotated image: {annotated_path}')

    convert_annotated_image(
        img_bgr,
        str(output_dxf),
        width_mm=args.width_mm,
        scale=args.scale,
        no_details=args.no_details,
        save_preview=True,
    )

    print('Done.')


if __name__ == '__main__':
    main()
