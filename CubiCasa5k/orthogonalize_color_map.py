#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
色块图正交化工具
将色块图转换为横平竖直的多边形（不一定是矩形），保持轮廓清晰规整
"""
import cv2
import numpy as np
import argparse
from pathlib import Path
from scipy.ndimage import label
from skimage import measure
from shapely.geometry import Polygon
from shapely.ops import unary_union

# 房间类别和颜色定义
room_classes = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", 
               "Bed Room", "Bath", "Entry", "Railing", "Storage", 
               "Garage", "Undefined"]

room_colors = {
    0: (220, 220, 220),  # Background - #DCDCDC
    1: (179, 222, 105),  # Outdoor - #b3de69
    2: (0, 0, 0),        # Wall - #000000
    3: (141, 211, 199),  # Kitchen - #8dd3c7
    4: (253, 180, 98),   # Living Room - #fdb462
    5: (252, 205, 229),  # Bed Room - #fccde5
    6: (128, 177, 211),  # Bath - #80b1d3
    7: (128, 128, 128),  # Entry - #808080
    8: (251, 128, 114),  # Railing - #fb8072
    9: (105, 105, 105),  # Storage - #696969
    10: (87, 122, 77),   # Garage - #577a4d
    11: (255, 255, 179), # Undefined - #ffffb3
}


def color_to_class_id(color_map):
    """
    将颜色图转换为类别ID图（向量化实现，高效）
    
    Args:
        color_map: (H, W, 3) RGB颜色图
    
    Returns:
        segmentation: (H, W) 类别ID图像
    """
    H, W = color_map.shape[:2]
    
    # 批量转换（向量化实现）
    color_flat = color_map.reshape(-1, 3).astype(np.float32)
    
    # 构建所有类别的颜色数组
    color_array = np.array([room_colors[i] for i in range(len(room_classes))], dtype=np.float32)  # (12, 3)
    
    # 对每个像素，计算到所有类别颜色的距离
    # color_flat: (N, 3), color_array: (12, 3)
    # 使用广播计算所有距离: (N, 12)
    diff = color_flat[:, np.newaxis, :] - color_array[np.newaxis, :, :]  # (N, 12, 3)
    distances = np.sum(diff ** 2, axis=2)  # (N, 12) - 每个像素到每个类别的距离
    
    # 找到距离最小的类别
    seg_flat = np.argmin(distances, axis=1).astype(np.int32)
    
    segmentation = seg_flat.reshape(H, W)
    return segmentation


def simplify_to_orthogonal(polygon_points, grid_size=1):
    """
    将多边形简化为横平竖直的多边形（只保留水平和垂直的边）
    使用改进的算法减少锯齿
    
    Args:
        polygon_points: (N, 2) 多边形顶点坐标 [[x1, y1], [x2, y2], ...]
        grid_size: 网格大小，用于对齐坐标（默认1，即像素级对齐）
    
    Returns:
        orthogonal_points: (M, 2) 简化后的横平竖直多边形顶点
    """
    if len(polygon_points) < 3:
        return polygon_points
    
    # 对齐到网格
    aligned_points = np.round(polygon_points / grid_size) * grid_size
    
    # 移除重复点
    unique_points = []
    prev_point = None
    for point in aligned_points:
        if prev_point is None or not np.allclose(point, prev_point, atol=0.1):
            unique_points.append(point)
            prev_point = point
    
    if len(unique_points) < 3:
        return polygon_points
    
    unique_points = np.array(unique_points)
    
    # 改进的简化算法：先分析整体方向，再分段处理
    simplified = []
    
    i = 0
    while i < len(unique_points):
        start_point = unique_points[i]
        simplified.append(start_point)
        
        # 向前查找，找到方向改变的点
        if i + 1 >= len(unique_points):
            break
        
        # 计算初始方向
        next_point = unique_points[(i + 1) % len(unique_points)]
        dx = next_point[0] - start_point[0]
        dy = next_point[1] - start_point[1]
        
        # 判断主要方向
        is_horizontal = abs(dx) > abs(dy)
        
        # 沿着当前方向继续，直到方向改变
        j = i + 1
        while j < len(unique_points):
            curr_point = unique_points[j]
            prev_point = unique_points[j - 1]
            
            dx_seg = curr_point[0] - prev_point[0]
            dy_seg = curr_point[1] - prev_point[1]
            
            # 判断当前段的方向
            curr_is_horizontal = abs(dx_seg) > abs(dy_seg)
            
            # 如果方向改变，停止
            if curr_is_horizontal != is_horizontal:
                # 添加方向改变点
                if is_horizontal:
                    # 之前是水平，现在要垂直：保持x不变
                    simplified.append(np.array([prev_point[0], curr_point[1]]))
                else:
                    # 之前是垂直，现在要水平：保持y不变
                    simplified.append(np.array([curr_point[0], prev_point[1]]))
                break
            
            j += 1
        
        # 如果到达末尾，添加最后一个点
        if j >= len(unique_points):
            end_point = unique_points[-1]
            if is_horizontal:
                simplified.append(np.array([end_point[0], start_point[1]]))
            else:
                simplified.append(np.array([start_point[0], end_point[1]]))
            break
        
        i = j
    
    # 移除连续重复的点
    final_points = []
    for i, point in enumerate(simplified):
        if i == 0 or not np.allclose(final_points[-1], point, atol=0.1):
            final_points.append(point)
    
    # 确保闭合
    if len(final_points) > 0:
        if not np.allclose(final_points[0], final_points[-1], atol=0.1):
            final_points.append(final_points[0])
    
    return np.array(final_points)


def extract_orthogonal_polygons(segmentation, min_area=100, grid_size=1, smooth_contour=True):
    """
    从分割结果中提取横平竖直的多边形
    
    Args:
        segmentation: (H, W) 类别ID图像
        min_area: 最小区域面积
        grid_size: 网格大小，用于对齐坐标
        smooth_contour: 是否先平滑轮廓再简化
    
    Returns:
        polygons_by_class: {class_id: [polygon1, polygon2, ...], ...}
    """
    polygons_by_class = {}
    unique_classes = np.unique(segmentation)
    
    for class_id in unique_classes:
        if class_id == 0:  # 跳过Background
            continue
        
        # 创建二值掩码
        mask = (segmentation == class_id).astype(np.uint8)
        
        # 先进行形态学平滑，减少锯齿
        if smooth_contour:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 标记连通域
        labeled_mask, num_features = label(mask)
        
        polygons_by_class[class_id] = []
        
        # 对每个连通域提取轮廓并简化
        for i in range(1, num_features + 1):
            component_mask = (labeled_mask == i).astype(np.uint8)
            area = np.sum(component_mask)
            
            if area < min_area:
                continue
            
            # 提取轮廓（使用更高的阈值以获得更平滑的轮廓）
            contours = measure.find_contours(component_mask, 0.5)
            
            if len(contours) == 0:
                continue
            
            # 使用最大的轮廓（外轮廓）
            largest_contour = max(contours, key=len)
            
            # 转换为 (x, y) 格式（注意measure返回的是 (row, col)，即 (y, x)）
            contour_points = np.array([[p[1], p[0]] for p in largest_contour])
            
            # 如果轮廓点太多，先进行Douglas-Peucker简化
            if len(contour_points) > 100:
                # 使用简单的距离阈值简化
                simplified_contour = [contour_points[0]]
                for k in range(1, len(contour_points) - 1):
                    prev_point = simplified_contour[-1]
                    curr_point = contour_points[k]
                    next_point = contour_points[k + 1]
                    
                    # 计算点到直线的距离
                    dx = next_point[0] - prev_point[0]
                    dy = next_point[1] - prev_point[1]
                    if dx == 0 and dy == 0:
                        continue
                    
                    # 点到直线的距离
                    dist = abs((dy * curr_point[0] - dx * curr_point[1] + next_point[0] * prev_point[1] - next_point[1] * prev_point[0]) / np.sqrt(dx*dx + dy*dy))
                    
                    # 如果距离大于阈值，保留这个点
                    if dist > 2.0:
                        simplified_contour.append(curr_point)
                
                simplified_contour.append(contour_points[-1])
                contour_points = np.array(simplified_contour)
            
            # 简化为横平竖直的多边形
            orthogonal_points = simplify_to_orthogonal(contour_points, grid_size)
            
            if len(orthogonal_points) >= 3:
                polygons_by_class[class_id].append(orthogonal_points)
    
    return polygons_by_class


def render_polygons_to_segmentation(polygons_by_class, shape):
    """
    将多边形渲染为分割图像（使用skimage.draw.polygon高效填充）
    
    Args:
        polygons_by_class: {class_id: [polygon1, polygon2, ...], ...}
        shape: (H, W) 输出图像尺寸
    
    Returns:
        segmentation: (H, W) 类别ID图像
    """
    H, W = shape
    segmentation = np.zeros((H, W), dtype=np.int32)
    
    # 按类别优先级绘制（非Background类别优先，然后按面积从大到小）
    all_polygons = []
    for class_id, polygons in polygons_by_class.items():
        for polygon in polygons:
            if len(polygon) < 3:
                continue
            # 计算多边形面积（近似）
            try:
                poly = Polygon(polygon)
                area = poly.area if poly.is_valid else 0
            except:
                area = 0
            all_polygons.append((class_id, polygon, area))
    
    # 排序：非Background优先，然后按面积从大到小
    all_polygons.sort(key=lambda x: (0 if x[0] == 0 else 1, -x[2], x[0]))
    
    # 绘制多边形
    from skimage.draw import polygon as draw_polygon
    
    for class_id, polygon, _ in all_polygons:
        if len(polygon) < 3:
            continue
        
        try:
            # 提取坐标（注意：skimage.draw.polygon需要 (row, col) 格式，即 (y, x)）
            # polygon是 (x, y) 格式，需要转换为 (y, x)
            y_coords = polygon[:, 1].astype(int)
            x_coords = polygon[:, 0].astype(int)
            
            # 确保坐标在图像范围内
            y_coords = np.clip(y_coords, 0, H - 1)
            x_coords = np.clip(x_coords, 0, W - 1)
            
            # 使用skimage.draw.polygon填充多边形
            rr, cc = draw_polygon(y_coords, x_coords, shape=(H, W))
            
            # 只填充未被其他非Background类别占用的位置
            if class_id == 0:
                mask = (segmentation[rr, cc] == 0)
                segmentation[rr[mask], cc[mask]] = class_id
            else:
                mask = (segmentation[rr, cc] == 0) | (segmentation[rr, cc] == class_id)
                segmentation[rr[mask], cc[mask]] = class_id
                
        except Exception as e:
            print(f"  警告：绘制多边形时出错 (class_id={class_id}): {e}")
            continue
    
    return segmentation


def class_id_to_color(segmentation):
    """
    将类别ID图转换为颜色图
    
    Args:
        segmentation: (H, W) 类别ID图像
    
    Returns:
        color_map: (H, W, 3) RGB颜色图
    """
    H, W = segmentation.shape
    color_map = np.zeros((H, W, 3), dtype=np.uint8)
    
    for class_id in range(len(room_classes)):
        mask = (segmentation == class_id)
        color_map[mask] = room_colors[class_id]
    
    return color_map


def orthogonalize_color_map(input_path, output_path=None, min_area=100, grid_size=1):
    """
    正交化色块图（转换为横平竖直的多边形）
    
    Args:
        input_path: 输入色块图路径
        output_path: 输出路径（如果为None，自动生成）
        min_area: 最小区域面积（像素数）
        grid_size: 网格大小，用于对齐坐标（默认1，即像素级）
    """
    print(f"正在读取色块图: {input_path}")
    # 读取图像（BGR格式）
    color_map_bgr = cv2.imread(input_path)
    if color_map_bgr is None:
        raise ValueError(f"无法读取图片: {input_path}")
    
    # 转换为RGB
    color_map = cv2.cvtColor(color_map_bgr, cv2.COLOR_BGR2RGB)
    print(f"  图像尺寸: {color_map.shape[1]}x{color_map.shape[0]}")
    
    # 转换为类别ID
    print("正在转换为类别ID...")
    segmentation = color_to_class_id(color_map)
    unique_classes = np.unique(segmentation)
    print(f"  检测到 {len(unique_classes)} 个类别: {[room_classes[c] for c in unique_classes]}")
    
    # 提取横平竖直的多边形
    print(f"正在提取横平竖直的多边形 (min_area={min_area}, grid_size={grid_size})...")
    polygons_by_class = extract_orthogonal_polygons(
        segmentation, 
        min_area=min_area,
        grid_size=grid_size,
        smooth_contour=True
    )
    
    # 统计多边形数量
    total_polygons = sum(len(polys) for polys in polygons_by_class.values())
    print(f"  提取到 {total_polygons} 个多边形")
    for class_id, polygons in polygons_by_class.items():
        if len(polygons) > 0:
            print(f"    {room_classes[class_id]}: {len(polygons)} 个多边形")
    
    # 渲染为分割图像
    print("正在渲染横平竖直的多边形...")
    orthogonal_seg = render_polygons_to_segmentation(
        polygons_by_class,
        segmentation.shape
    )
    
    # 转换回颜色图
    print("正在生成正交化色块图...")
    orthogonal_color_map = class_id_to_color(orthogonal_seg)
    
    # 生成输出路径
    if output_path is None:
        input_path_obj = Path(input_path)
        output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_orthogonal{input_path_obj.suffix}")
    
    # 保存（转换为BGR）
    orthogonal_color_map_bgr = cv2.cvtColor(orthogonal_color_map, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, orthogonal_color_map_bgr)
    print(f"✓ 已保存: {output_path}")
    
    return orthogonal_color_map, output_path


def main():
    parser = argparse.ArgumentParser(description='色块图正交化工具 - 生成横平竖直的多边形色块图')
    parser.add_argument('--input', type=str, required=True,
                        help='输入色块图路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出路径（默认：输入文件名_orthogonal.png）')
    parser.add_argument('--min-area', type=int, default=100,
                        help='最小区域面积（像素数），小于此值的区域将被忽略')
    parser.add_argument('--grid-size', type=int, default=1,
                        help='网格大小，用于对齐坐标（默认1，即像素级对齐）')
    
    args = parser.parse_args()
    
    print("="*60)
    print("色块图正交化工具")
    print("="*60)
    print(f"输入文件: {args.input}")
    print(f"最小区域面积: {args.min_area}")
    print(f"网格大小: {args.grid_size}")
    print("="*60)
    
    try:
        orthogonal_color_map, output_path = orthogonalize_color_map(
            args.input,
            args.output,
            min_area=args.min_area,
            grid_size=args.grid_size
        )
        
        print("\n" + "="*60)
        print("正交化完成！")
        print("="*60)
        print(f"输出文件: {output_path}")
        print("\n提示：生成的色块图由横平竖直的多边形组成，边界清晰规整")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

