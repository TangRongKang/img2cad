#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化版户型图推理工具 - 解决"未定义"类别过多和漏检问题
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


def apply_class_weights(rooms_logits, class_weights=None):
    """
    应用类别权重来调整预测
    
    Args:
        rooms_logits: 房间logits (C, H, W)
        class_weights: 类别权重字典，例如 {11: 0.5} 表示降低Undefined的权重
    """
    if class_weights is None:
        return rooms_logits
    
    # 创建权重张量
    weights = torch.ones(rooms_logits.shape[0], device=rooms_logits.device)
    for class_id, weight in class_weights.items():
        if 0 <= class_id < rooms_logits.shape[0]:
            weights[class_id] = weight
    
    # 应用权重
    weighted_logits = rooms_logits * weights.view(-1, 1, 1)
    return weighted_logits


def apply_temperature_scaling(rooms_logits, temperature=1.0):
    """
    温度缩放：调整softmax的"温度"来改变概率分布
    
    注意：温度缩放是对所有类别一视同仁的，它会放大或缩小所有类别logits之间的相对差异。
    单独使用temperature < 1.0不能解决Undefined过多的问题，因为：
    - 如果非Undefined类别的logit本来就最高，temperature < 1.0会放大优势
    - 但如果Undefined的logit最高，temperature < 1.0反而会让Undefined更容易被选中
    
    建议：先使用类别权重（undefined-weight）降低Undefined的logit，然后再使用温度缩放。
    
    Args:
        rooms_logits: 房间logits
        temperature: 温度参数
          - < 1.0: 放大logits差异，使高概率更高，低概率更低
          - = 1.0: 标准softmax
          - > 1.0: 缩小logits差异，使分布更平滑（更保守）
    """
    return rooms_logits / temperature


def suppress_undefined_class(rooms_prob, undefined_class_id=11, min_prob=0.3, suppression_factor=0.5):
    """
    抑制"未定义"类别：如果最大概率不是Undefined且概率足够高，则降低Undefined的概率
    
    Args:
        rooms_prob: 房间概率 (C, H, W)
        undefined_class_id: 未定义类别ID
        min_prob: 最小概率阈值，低于此值才考虑抑制
        suppression_factor: 抑制因子，将Undefined概率乘以这个值
    """
    rooms_prob = rooms_prob.copy()
    
    # 找到每个像素的最大概率和对应类别
    max_probs = np.max(rooms_prob, axis=0)
    max_classes = np.argmax(rooms_prob, axis=0)
    
    # 如果最大概率足够高且不是Undefined，则抑制Undefined
    mask = (max_probs >= min_prob) & (max_classes != undefined_class_id)
    rooms_prob[undefined_class_id, mask] *= suppression_factor
    
    # 重新归一化
    rooms_prob = rooms_prob / rooms_prob.sum(axis=0, keepdims=True)
    
    return rooms_prob


def apply_min_prob_threshold(rooms_prob, min_prob=0.2, fallback_class=0):
    """
    应用最小概率阈值：如果最大概率低于阈值，使用fallback类别而不是argmax
    
    Args:
        rooms_prob: 房间概率 (C, H, W)
        min_prob: 最小概率阈值
        fallback_class: 当概率太低时使用的类别（通常是Background）
    """
    max_probs = np.max(rooms_prob, axis=0)
    max_classes = np.argmax(rooms_prob, axis=0)
    
    # 如果最大概率低于阈值，使用fallback类别
    mask = max_probs < min_prob
    max_classes[mask] = fallback_class
    
    return max_classes


def predict_rooms_optimized(model, img_tensor, device, split=[21, 12, 11], 
                            original_size=None,
                            class_weights=None,
                            temperature=1.0,
                            suppress_undefined=True,
                            min_prob_threshold=0.0,
                            use_flip=False):
    """
    优化的房间预测函数
    
    Args:
        model: 模型
        img_tensor: 输入图片
        device: 设备
        split: 分割配置
        original_size: 原始尺寸
        class_weights: 类别权重，例如 {11: 0.5} 降低Undefined权重
        temperature: 温度缩放参数
        suppress_undefined: 是否抑制Undefined类别
        min_prob_threshold: 最小概率阈值
        use_flip: 是否使用水平翻转增强
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
    
    # 旋转融合（4次旋转，效果最好）
    rot = RotateNTurns()
    rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
    all_predictions = []
    
    with torch.no_grad():
        # 1. 标准旋转融合（4次旋转）
        for r in rotations:
            forward, back = r
            rot_image = rot(img_tensor, 'tensor', forward)
            pred = model(rot_image)
            pred = rot(pred, 'tensor', back)
            pred = rot(pred, 'points', back)
            pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=True)
            all_predictions.append(pred[0])
        
        # 2. 水平翻转增强（如果启用）
        if use_flip:
            flipped_img = torch.flip(img_tensor, [3])  # 水平翻转输入图像 (1, C, H, W) 的宽度维度
            for r in rotations:
                forward, back = r
                rot_image = rot(flipped_img, 'tensor', forward)
                pred = model(rot_image)
                pred = rot(pred, 'tensor', back)
                pred = rot(pred, 'points', back)
                pred = torch.flip(pred, [3])  # 水平翻转回来 (1, C, H, W) 的宽度维度
                pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=True)
                all_predictions.append(pred[0])
    
    # 对所有预测结果取平均
    prediction = torch.stack(all_predictions).mean(dim=0, keepdim=True)  # (1, n_classes, H, W)
    
    # 提取房间类别部分
    rooms_logits = prediction[:, split[0]:split[0]+split[1], :, :]  # (1, 12, H, W)
    rooms_logits = rooms_logits.squeeze(0)  # (12, H, W)
    
    # 应用类别权重
    if class_weights:
        rooms_logits = apply_class_weights(rooms_logits, class_weights)
    
    # 应用温度缩放
    if temperature != 1.0:
        rooms_logits = apply_temperature_scaling(rooms_logits, temperature)
    
    # Softmax得到概率
    rooms_prob = F.softmax(rooms_logits, dim=0)  # (12, H, W)
    rooms_prob_np = rooms_prob.cpu().numpy()
    
    # 抑制Undefined类别
    if suppress_undefined:
        rooms_prob_np = suppress_undefined_class(
            rooms_prob_np, 
            undefined_class_id=11,
            min_prob=0.3,
            suppression_factor=0.3  # 将Undefined概率降低到30%
        )
    
    # 应用最小概率阈值
    if min_prob_threshold > 0:
        rooms_pred = apply_min_prob_threshold(
            rooms_prob_np,
            min_prob=min_prob_threshold,
            fallback_class=0  # Background
        )
    else:
        rooms_pred = np.argmax(rooms_prob_np, axis=0)
    
    return rooms_pred, rooms_prob_np, prediction, img_size


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
                                threshold=0.2, opening_types=[1, 2],
                                optimized_rooms_prob=None, optimized_rooms_pred=None):
    """
    使用后处理生成矢量多边形分割图
    
    Args:
        prediction: 原始预测tensor
        img_size: 图片尺寸
        split: 分割配置
        threshold: 多边形提取阈值
        opening_types: 开口类型
        optimized_rooms_prob: 优化后的房间概率分布 (12, H, W)，如果提供则使用此概率而不是原始预测
        optimized_rooms_pred: 优化后的房间预测结果 (H, W)，如果提供则直接使用此结果生成one-hot概率
    """
    if prediction.is_cuda:
        prediction = prediction.cpu()
    
    heatmaps, rooms, icons = split_prediction(prediction, img_size, split)
    
    # 如果提供了优化后的预测结果，直接使用它生成one-hot概率分布
    if optimized_rooms_pred is not None:
        print(f"  使用优化后的预测结果，尺寸: {optimized_rooms_pred.shape}")
        print(f"  rooms 尺寸: {rooms.shape}")
        # 将预测结果插值到目标尺寸
        if optimized_rooms_pred.shape != rooms.shape[1:]:
            print(f"  尺寸不匹配，进行插值: {optimized_rooms_pred.shape} -> {rooms.shape[1:]}")
            import torch.nn.functional as F
            pred_tensor = torch.from_numpy(optimized_rooms_pred).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
            pred_tensor = F.interpolate(
                pred_tensor,
                size=(rooms.shape[1], rooms.shape[2]),
                mode='nearest',  # 使用nearest保持类别ID不变
                align_corners=False
            ).squeeze(0).squeeze(0).numpy().astype(np.int32)  # (H, W)
        else:
            print(f"  尺寸匹配，直接使用")
            pred_tensor = optimized_rooms_pred.astype(np.int32)
        
        # 将类别ID转换为one-hot概率分布
        n_classes = rooms.shape[0]
        rooms_onehot = np.zeros((n_classes, pred_tensor.shape[0], pred_tensor.shape[1]), dtype=np.float32)
        for class_id in range(n_classes):
            rooms_onehot[class_id, pred_tensor == class_id] = 1.0
        
        rooms = rooms_onehot
        print(f"  生成的 one-hot 概率分布尺寸: {rooms.shape}")
    # 如果提供了优化后的概率，使用它替换原始的rooms概率
    elif optimized_rooms_prob is not None:
        # 确保尺寸匹配
        if optimized_rooms_prob.shape[1:] != rooms.shape[1:]:
            # 需要插值到相同尺寸，使用nearest保持argmax结果一致
            import torch.nn.functional as F
            # 先对概率做argmax得到类别
            pred_from_prob = np.argmax(optimized_rooms_prob, axis=0)
            pred_tensor = torch.from_numpy(pred_from_prob).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
            pred_tensor = F.interpolate(
                pred_tensor,
                size=(rooms.shape[1], rooms.shape[2]),
                mode='nearest',  # 使用nearest保持类别ID不变
                align_corners=False
            ).squeeze(0).squeeze(0).numpy().astype(np.int32)  # (H, W)
            
            # 将类别ID转换为one-hot概率分布
            n_classes = rooms.shape[0]
            rooms_onehot = np.zeros((n_classes, pred_tensor.shape[0], pred_tensor.shape[1]), dtype=np.float32)
            for class_id in range(n_classes):
                rooms_onehot[class_id, pred_tensor == class_id] = 1.0
            
            rooms = rooms_onehot
        else:
            rooms = optimized_rooms_prob
    
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


def analyze_predictions(rooms_prob, rooms_pred):
    """分析预测结果，输出统计信息"""
    print("\n" + "="*60)
    print("预测结果分析")
    print("="*60)
    
    # 统计每个类别的像素数和平均概率
    total_pixels = rooms_pred.size
    unique_classes, counts = np.unique(rooms_pred, return_counts=True)
    
    print(f"\n类别分布:")
    for class_id, count in zip(unique_classes, counts):
        percentage = (count / total_pixels) * 100
        avg_prob = rooms_prob[class_id, rooms_pred == class_id].mean() if np.any(rooms_pred == class_id) else 0
        print(f"  {room_classes[class_id]:<15}: {percentage:6.2f}%  (平均概率: {avg_prob:.3f})")
    
    # 分析Undefined类别
    undefined_mask = rooms_pred == 11
    if np.any(undefined_mask):
        undefined_probs = rooms_prob[:, undefined_mask]
        print(f"\n未定义类别分析:")
        print(f"  - 像素数: {np.sum(undefined_mask)} ({np.sum(undefined_mask)/total_pixels*100:.2f}%)")
        print(f"  - 平均概率: {undefined_probs[11].mean():.3f}")
        print(f"  - 最大概率: {undefined_probs[11].max():.3f}")
        print(f"  - 最小概率: {undefined_probs[11].min():.3f}")
        
        # 检查是否有其他类别概率更高但被选为Undefined的情况
        other_max_probs = np.max(undefined_probs[:11], axis=0)
        better_than_undefined = np.sum(other_max_probs > undefined_probs[11])
        if better_than_undefined > 0:
            print(f"  ⚠️  有 {better_than_undefined} 个像素存在其他类别概率更高但仍被选为Undefined")


def main():
    parser = argparse.ArgumentParser(description='优化版户型图推理工具 - 解决未定义类别过多问题')
    parser.add_argument('--image', type=str, required=True, help='输入户型图路径')
    parser.add_argument('--model', type=str, 
                        default='floortrans/models/model_best_val_loss_var.pkl',
                        help='模型权重文件路径')
    parser.add_argument('--image-size', type=int, default=512, help='模型输入图片尺寸')
    parser.add_argument('--n-classes', type=int, default=44, help='模型输出类别数')
    parser.add_argument('--device-id', type=int, default=2, help='GPU设备ID')
    
    # 优化参数
    parser.add_argument('--undefined-weight', type=float, default=0.3,
                        help='Undefined类别权重 (0.0-1.0，越小越抑制，默认0.3)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='温度缩放参数 (0.5-2.0，<1更自信，>1更保守，默认1.0)')
    suppress_group = parser.add_mutually_exclusive_group()
    suppress_group.add_argument('--suppress-undefined', action='store_true', default=True,
                                help='抑制Undefined类别（默认启用）')
    suppress_group.add_argument('--no-suppress-undefined', dest='suppress_undefined', action='store_false',
                                help='禁用Undefined抑制')
    parser.add_argument('--min-prob', type=float, default=0.0,
                        help='最小概率阈值，低于此值使用Background (0.0-1.0)')
    parser.add_argument('--use-flip', action='store_true',
                        help='使用水平翻转增强（在4次旋转基础上，再对翻转图像进行4次旋转，共8次预测）')
    
    # 后处理参数
    parser.add_argument('--polygon-threshold', type=float, default=0.2,
                        help='多边形提取阈值')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, 
                        default='/mnt/workspace/yangxiaohang/cubicase5k/CubiCasa5k/result',
                        help='输出目录')
    parser.add_argument('--analyze', action='store_true',
                        help='分析预测结果')
    
    args = parser.parse_args()
    
    result_dir = args.output_dir
    os.makedirs(result_dir, exist_ok=True)
    
    input_path = Path(args.image)
    base_name = input_path.stem
    
    discrete_cmap()
    
    print("="*60)
    print("优化版户型图推理工具")
    print("="*60)
    print(f"输入图片: {args.image}")
    print(f"模型路径: {args.model}")
    print(f"\n优化参数:")
    print(f"  - Undefined权重: {args.undefined_weight}")
    print(f"  - 温度缩放: {args.temperature}")
    print(f"  - 抑制Undefined: {args.suppress_undefined}")
    print(f"  - 最小概率阈值: {args.min_prob}")
    print(f"  - 翻转增强: {args.use_flip}")
    print("="*60)
    
    print("\n正在加载模型...")
    model, device = load_model(args.model, args.n_classes, args.image_size, args.device_id)
    
    print(f"\n正在加载图片: {args.image}")
    img_tensor, original_size = load_image(args.image, args.image_size)
    
    print("\n正在进行优化推理...")
    class_weights = {11: args.undefined_weight} if args.undefined_weight < 1.0 else None
    
    rooms_pred, rooms_prob, full_prediction, img_size = predict_rooms_optimized(
        model, img_tensor, device, 
        split=[21, 12, 11], 
        original_size=original_size,
        class_weights=class_weights,
        temperature=args.temperature,
        suppress_undefined=args.suppress_undefined,
        min_prob_threshold=args.min_prob,
        use_flip=args.use_flip
    )
    
    # 分析预测结果
    if args.analyze:
        analyze_predictions(rooms_prob, rooms_pred)
    
    print("\n正在生成色块图...")
    color_map = create_color_map(rooms_pred, original_size)
    
    output_color_map = os.path.join(result_dir, f'{base_name}_color_map_optimized.png')
    color_map_bgr = cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_color_map, color_map_bgr)
    print(f"  ✓ 已保存: {output_color_map}")
    
    # 生成多边形分割（使用优化后的预测结果）
    print("\n正在生成矢量多边形分割图...")
    # 确保 rooms_pred 的尺寸与 img_size 匹配
    # rooms_pred 的尺寸是 (img_size[0], img_size[1])，即 (original_height, original_width)
    # 这与 split_prediction 输出的 rooms 尺寸应该匹配
    print(f"  rooms_pred 尺寸: {rooms_pred.shape}")
    print(f"  img_size: {img_size}")
    # 直接使用优化后的预测结果，确保与色块图一致
    pol_room_seg, pol_icon_seg = create_polygon_segmentation(
        full_prediction, img_size, 
        split=[21, 12, 11], 
        threshold=args.polygon_threshold,
        opening_types=[1, 2],
        optimized_rooms_pred=rooms_pred  # 直接使用优化后的预测结果
    )
    
    polygon_color_map = create_color_map(pol_room_seg, original_size)
    output_polygon = os.path.join(result_dir, f'{base_name}_color_map_polygon_optimized.png')
    polygon_color_map_bgr = cv2.cvtColor(polygon_color_map, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_polygon, polygon_color_map_bgr)
    print(f"  ✓ 已保存: {output_polygon}")
    
    # 生成叠加图
    print("\n正在生成叠加图...")
    original_img = cv2.imread(args.image)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    overlay = overlay_on_original(original_img, color_map)
    output_overlay = os.path.join(result_dir, f'{base_name}_overlay_optimized.png')
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_overlay, overlay_bgr)
    print(f"  ✓ 已保存: {output_overlay}")
    
    polygon_overlay = overlay_on_original(original_img, polygon_color_map)
    output_overlay_polygon = os.path.join(result_dir, f'{base_name}_overlay_polygon_optimized.png')
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

