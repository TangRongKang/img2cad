#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版户型图色块生成工具 - 支持多种推理优化策略
无需重新训练，通过调整推理参数来优化结果
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
import os
from pathlib import Path

from floortrans.models import get_model
from floortrans.loaders.augmentations import DictToTensor, RotateNTurns
from floortrans.post_prosessing import split_prediction, get_polygons
from floortrans.plotting import polygons_to_image, discrete_cmap

# 房间类别和颜色定义
room_classes = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", 
               "Bed Room", "Bath", "Entry", "Railing", "Storage", 
               "Garage", "Undefined"]

room_colors = {
    0: (220, 220, 220), 1: (179, 222, 105), 2: (0, 0, 0), 3: (141, 211, 199),
    4: (253, 180, 98), 5: (252, 205, 229), 6: (128, 177, 211), 7: (128, 128, 128),
    8: (251, 128, 114), 9: (105, 105, 105), 10: (87, 122, 77), 11: (255, 255, 179),
}


def load_image(image_path, image_size=512):
    """加载并预处理图片"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_height, original_width = img.shape[:2]
    
    img_tensor = torch.from_numpy(img.astype(np.float32))
    img_tensor = img_tensor.permute(2, 0, 1)
    img_tensor = 2 * (img_tensor / 255.0) - 1
    
    img_tensor = F.interpolate(
        img_tensor.unsqueeze(0), 
        size=(image_size, image_size), 
        mode='bilinear', 
        align_corners=False
    ).squeeze(0)
    
    return img_tensor, (original_height, original_width)


def load_model(model_path, n_classes=44, image_size=512, device_id=2):
    """加载训练好的模型"""
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
        print(f"使用GPU: {torch.cuda.get_device_name(device_id)}")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    
    model = get_model('hg_furukawa_original', n_classes=51)
    model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, device


def predict_with_tta(model, img_tensor, device, split=[21, 12, 11], 
                     original_size=None, use_flip=True, use_scale=False, scales=[0.8, 1.0, 1.2]):
    """
    测试时增强 (Test Time Augmentation) 预测
    
    Args:
        model: 训练好的模型
        img_tensor: 输入图片张量
        device: 设备
        split: 分割配置
        original_size: 原始尺寸
        use_flip: 是否使用翻转增强
        use_scale: 是否使用多尺度
        scales: 多尺度比例列表
    """
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        test_output = model(img_tensor)
    
    if original_size is not None:
        height, width = original_size[0], original_size[1]
    else:
        height, width = test_output.shape[2], test_output.shape[3]
    
    img_size = (height, width)
    n_classes = test_output.shape[1]
    
    # 基础旋转融合
    rot = RotateNTurns()
    rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
    all_predictions = []
    
    # 1. 标准旋转融合
    with torch.no_grad():
        for r in rotations:
            forward, back = r
            rot_image = rot(img_tensor, 'tensor', forward)
            pred = model(rot_image)
            pred = rot(pred, 'tensor', back)
            pred = rot(pred, 'points', back)
            pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=True)
            all_predictions.append(pred[0])
        
        # 2. 水平翻转增强
        if use_flip:
            flipped_img = torch.flip(img_tensor, [3])  # 水平翻转
            for r in rotations:
                forward, back = r
                rot_image = rot(flipped_img, 'tensor', forward)
                pred = model(rot_image)
                pred = rot(pred, 'tensor', back)
                pred = rot(pred, 'points', back)
                pred = torch.flip(pred, [2])  # 翻转回来
                pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=True)
                all_predictions.append(pred[0])
        
        # 3. 多尺度预测
        if use_scale and scales:
            for scale in scales:
                if scale == 1.0:
                    continue  # 已经处理过了
                scaled_size = (int(height * scale), int(width * scale))
                scaled_img = F.interpolate(
                    img_tensor, 
                    size=scaled_size, 
                    mode='bilinear', 
                    align_corners=False
                )
                pred = model(scaled_img)
                pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=True)
                all_predictions.append(pred[0])
    
    # 对所有预测取平均
    prediction = torch.stack(all_predictions).mean(dim=0, keepdim=True)
    
    # 提取房间类别
    rooms_logits = prediction[:, split[0]:split[0]+split[1], :, :]
    rooms_prob = F.softmax(rooms_logits, dim=1)
    rooms_pred = torch.argmax(rooms_prob, dim=1)
    
    rooms_pred = rooms_pred.cpu().numpy()[0]
    
    return rooms_pred, rooms_prob.cpu().numpy()[0], prediction, img_size


def predict_rooms(model, img_tensor, device, split=[21, 12, 11], original_size=None):
    """标准预测（保持向后兼容）"""
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        test_output = model(img_tensor)
    
    if original_size is not None:
        height, width = original_size[0], original_size[1]
    else:
        height, width = test_output.shape[2], test_output.shape[3]
    
    img_size = (height, width)
    n_classes = test_output.shape[1]
    
    rot = RotateNTurns()
    rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
    pred_count = len(rotations)
    prediction = torch.zeros([pred_count, n_classes, height, width], device=device)
    
    with torch.no_grad():
        for i, r in enumerate(rotations):
            forward, back = r
            rot_image = rot(img_tensor, 'tensor', forward)
            pred = model(rot_image)
            pred = rot(pred, 'tensor', back)
            pred = rot(pred, 'points', back)
            pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=True)
            prediction[i] = pred[0]
    
    prediction = torch.mean(prediction, 0, True)
    
    rooms_logits = prediction[:, split[0]:split[0]+split[1], :, :]
    rooms_prob = F.softmax(rooms_logits, dim=1)
    rooms_pred = torch.argmax(rooms_prob, dim=1)
    rooms_pred = rooms_pred.cpu().numpy()[0]
    
    return rooms_pred, rooms_prob.cpu().numpy()[0], prediction, img_size


def apply_probability_threshold(rooms_prob, threshold=0.5):
    """
    应用概率阈值过滤
    
    Args:
        rooms_prob: 房间概率 (C, H, W)
        threshold: 概率阈值，低于此值的预测设为Background
    """
    max_prob = np.max(rooms_prob, axis=0)
    max_class = np.argmax(rooms_prob, axis=0)
    
    # 如果最大概率低于阈值，设为Background
    mask = max_prob < threshold
    max_class[mask] = 0  # Background
    
    return max_class


def apply_morphology_smoothing(prediction, kernel_size=3):
    """
    应用形态学平滑处理
    
    Args:
        prediction: 预测结果 (H, W)
        kernel_size: 核大小
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # 开运算：去除小噪点
    opened = cv2.morphologyEx(prediction.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    # 闭运算：填充小洞
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    return closed.astype(np.int32)


def apply_confidence_smoothing(rooms_prob, kernel_size=5):
    """
    对概率图进行平滑处理
    
    Args:
        rooms_prob: 房间概率 (C, H, W)
        kernel_size: 平滑核大小
    """
    from scipy.ndimage import uniform_filter
    
    smoothed = np.zeros_like(rooms_prob)
    for i in range(rooms_prob.shape[0]):
        smoothed[i] = uniform_filter(rooms_prob[i], size=kernel_size)
    
    return smoothed


def create_color_map(rooms_pred, original_size=None):
    """将房间类别ID映射到颜色"""
    height, width = rooms_pred.shape
    color_map = np.zeros((height, width, 3), dtype=np.uint8)
    
    for class_id in range(len(room_classes)):
        mask = (rooms_pred == class_id)
        color_map[mask] = room_colors[class_id]
    
    if original_size is not None:
        color_map = cv2.resize(
            color_map, 
            (original_size[1], original_size[0]), 
            interpolation=cv2.INTER_NEAREST
        )
    
    return color_map


def create_polygon_segmentation(prediction, img_size, split=[21, 12, 11], 
                                threshold=0.2, opening_types=[1, 2]):
    """
    使用后处理生成矢量多边形分割图
    
    Args:
        prediction: 完整预测结果
        img_size: 图像尺寸
        split: 分割配置
        threshold: 多边形提取阈值 (可调整: 0.1-0.5)
        opening_types: 开启类型列表 [1=Window, 2=Door]
    """
    if prediction.is_cuda:
        prediction = prediction.cpu()
    
    heatmaps, rooms, icons = split_prediction(prediction, img_size, split)
    
    polygons, types, room_polygons, room_types = get_polygons(
        (heatmaps, rooms, icons), threshold, opening_types
    )
    
    height, width = img_size
    pol_room_seg, pol_icon_seg = polygons_to_image(
        polygons, types, room_polygons, room_types, height, width
    )
    
    return pol_room_seg, pol_icon_seg


def overlay_on_original(original_img, color_map, alpha=0.6):
    """将色块图叠加到原始图片上"""
    if original_img.shape[:2] != color_map.shape[:2]:
        color_map = cv2.resize(
            color_map, 
            (original_img.shape[1], original_img.shape[0]), 
            interpolation=cv2.INTER_NEAREST
        )
    
    overlay = cv2.addWeighted(original_img, 1-alpha, color_map, alpha, 0)
    return overlay


def main():
    parser = argparse.ArgumentParser(description='增强版户型图色块生成工具')
    parser.add_argument('--image', type=str, required=True, help='输入户型图路径')
    parser.add_argument('--model', type=str, 
                        default='floortrans/models/model_best_val_loss_var.pkl',
                        help='模型权重文件路径')
    parser.add_argument('--image-size', type=int, default=512, help='模型输入图片尺寸')
    parser.add_argument('--n-classes', type=int, default=44, help='模型输出类别数')
    parser.add_argument('--device-id', type=int, default=2, help='GPU设备ID')
    
    # 推理优化参数
    parser.add_argument('--use-tta', action='store_true', 
                        help='使用测试时增强（翻转+多尺度）')
    parser.add_argument('--use-flip', action='store_true', default=True,
                        help='使用水平翻转增强')
    parser.add_argument('--use-scale', action='store_true',
                        help='使用多尺度预测')
    parser.add_argument('--scales', type=float, nargs='+', default=[0.8, 1.0, 1.2],
                        help='多尺度比例列表')
    
    # 后处理参数
    parser.add_argument('--polygon-threshold', type=float, default=0.2,
                        help='多边形提取阈值 (0.1-0.5，默认0.2)')
    parser.add_argument('--prob-threshold', type=float, default=0.0,
                        help='概率阈值，低于此值的预测设为Background (0.0-1.0)')
    parser.add_argument('--use-morphology', action='store_true',
                        help='使用形态学平滑处理')
    parser.add_argument('--morph-kernel', type=int, default=3,
                        help='形态学核大小 (3, 5, 7)')
    parser.add_argument('--use-confidence-smooth', action='store_true',
                        help='对概率图进行平滑处理')
    parser.add_argument('--smooth-kernel', type=int, default=5,
                        help='概率平滑核大小')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, 
                        default='/mnt/workspace/yangxiaohang/cubicase5k/CubiCasa5k/result',
                        help='输出目录')
    parser.add_argument('--save-all', action='store_true',
                        help='保存所有中间结果')
    
    args = parser.parse_args()
    
    result_dir = args.output_dir
    os.makedirs(result_dir, exist_ok=True)
    
    input_path = Path(args.image)
    base_name = input_path.stem
    
    discrete_cmap()
    
    print("="*60)
    print("增强版户型图推理工具")
    print("="*60)
    print(f"输入图片: {args.image}")
    print(f"模型路径: {args.model}")
    print(f"\n推理参数:")
    print(f"  - TTA增强: {args.use_tta}")
    print(f"  - 翻转增强: {args.use_flip}")
    print(f"  - 多尺度: {args.use_scale}")
    if args.use_scale:
        print(f"  - 尺度列表: {args.scales}")
    print(f"\n后处理参数:")
    print(f"  - 多边形阈值: {args.polygon_threshold}")
    print(f"  - 概率阈值: {args.prob_threshold}")
    print(f"  - 形态学平滑: {args.use_morphology}")
    print(f"  - 概率平滑: {args.use_confidence_smooth}")
    print("="*60)
    
    print("\n正在加载模型...")
    model, device = load_model(args.model, args.n_classes, args.image_size, args.device_id)
    
    print(f"\n正在加载图片: {args.image}")
    img_tensor, original_size = load_image(args.image, args.image_size)
    
    print("\n正在进行推理...")
    if args.use_tta:
        rooms_pred, rooms_prob, full_prediction, img_size = predict_with_tta(
            model, img_tensor, device, 
            split=[21, 12, 11], 
            original_size=original_size,
            use_flip=args.use_flip,
            use_scale=args.use_scale,
            scales=args.scales
        )
        print("  ✓ 使用测试时增强")
    else:
        rooms_pred, rooms_prob, full_prediction, img_size = predict_rooms(
            model, img_tensor, device, 
            split=[21, 12, 11], 
            original_size=original_size
        )
        print("  ✓ 使用标准旋转融合")
    
    # 应用概率阈值
    if args.prob_threshold > 0:
        rooms_pred = apply_probability_threshold(rooms_prob, args.prob_threshold)
        print(f"  ✓ 应用概率阈值: {args.prob_threshold}")
    
    # 概率平滑
    if args.use_confidence_smooth:
        rooms_prob_smooth = apply_confidence_smoothing(rooms_prob, args.smooth_kernel)
        rooms_pred = np.argmax(rooms_prob_smooth, axis=0)
        print(f"  ✓ 应用概率平滑: kernel={args.smooth_kernel}")
    
    # 形态学平滑
    if args.use_morphology:
        rooms_pred = apply_morphology_smoothing(rooms_pred, args.morph_kernel)
        print(f"  ✓ 应用形态学平滑: kernel={args.morph_kernel}")
    
    print("\n正在生成色块图...")
    color_map = create_color_map(rooms_pred, original_size)
    
    output_color_map = os.path.join(result_dir, f'{base_name}_color_map_advanced.png')
    color_map_bgr = cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_color_map, color_map_bgr)
    print(f"  ✓ 已保存: {output_color_map}")
    
    # 生成多边形分割
    print("\n正在生成矢量多边形分割图...")
    pol_room_seg, pol_icon_seg = create_polygon_segmentation(
        full_prediction, img_size, 
        split=[21, 12, 11], 
        threshold=args.polygon_threshold,
        opening_types=[1, 2]
    )
    
    polygon_color_map = create_color_map(pol_room_seg, original_size)
    output_polygon = os.path.join(result_dir, f'{base_name}_color_map_polygon_advanced.png')
    polygon_color_map_bgr = cv2.cvtColor(polygon_color_map, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_polygon, polygon_color_map_bgr)
    print(f"  ✓ 已保存: {output_polygon}")
    
    # 生成叠加图
    print("\n正在生成叠加图...")
    original_img = cv2.imread(args.image)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    overlay = overlay_on_original(original_img, color_map)
    output_overlay = os.path.join(result_dir, f'{base_name}_overlay_advanced.png')
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_overlay, overlay_bgr)
    print(f"  ✓ 已保存: {output_overlay}")
    
    polygon_overlay = overlay_on_original(original_img, polygon_color_map)
    output_overlay_polygon = os.path.join(result_dir, f'{base_name}_overlay_polygon_advanced.png')
    polygon_overlay_bgr = cv2.cvtColor(polygon_overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_overlay_polygon, polygon_overlay_bgr)
    print(f"  ✓ 已保存: {output_overlay_polygon}")
    
    # 统计信息
    print("\n房间类别统计:")
    unique, counts = np.unique(rooms_pred, return_counts=True)
    total_pixels = rooms_pred.size
    for class_id, count in zip(unique, counts):
        percentage = (count / total_pixels) * 100
        print(f"  {room_classes[class_id]:<15}: {percentage:6.2f}%")


if __name__ == '__main__':
    main()

