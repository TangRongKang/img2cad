"""
生成户型图的色块特征图
输入：一张户型图
输出：色块图（不同房间类型用不同颜色表示）
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from floortrans.models import get_model
from floortrans.loaders.augmentations import DictToTensor, RotateNTurns
from floortrans.post_prosessing import split_prediction, get_polygons
from floortrans.plotting import polygons_to_image, discrete_cmap

# 房间类别列表（12个类别，对应模型输出的split[1]=12个通道）
# 与示例代码 samples.ipynb 保持一致
room_classes = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", 
               "Bed Room", "Bath", "Entry", "Railing", "Storage", 
               "Garage", "Undefined"]

# 为每个房间类别定义颜色（RGB格式，0-255）
# 使用 discrete_cmap() 中定义的颜色，与示例代码保持一致
room_colors = {
    0: (220, 220, 220),      # Background - #DCDCDC
    1: (179, 222, 105),      # Outdoor - #b3de69
    2: (0, 0, 0),            # Wall - #000000
    3: (141, 211, 199),      # Kitchen - #8dd3c7
    4: (253, 180, 98),       # Living Room - #fdb462
    5: (252, 205, 229),      # Bed Room - #fccde5
    6: (128, 177, 211),       # Bath - #80b1d3
    7: (128, 128, 128),      # Entry - #808080
    8: (251, 128, 114),      # Railing - #fb8072
    9: (105, 105, 105),      # Storage - #696969
    10: (87, 122, 77),       # Garage - #577a4d
    11: (255, 255, 179),     # Undefined - #ffffb3
}


def load_image(image_path, image_size=512):
    """
    加载并预处理图片
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_height, original_width = img.shape[:2]
    
    # 转换为tensor并归一化到[-1, 1]
    img_tensor = torch.from_numpy(img.astype(np.float32))
    img_tensor = img_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    img_tensor = 2 * (img_tensor / 255.0) - 1  # 归一化到[-1, 1]
    
    # 调整大小到模型输入尺寸
    img_tensor = F.interpolate(
        img_tensor.unsqueeze(0), 
        size=(image_size, image_size), 
        mode='bilinear', 
        align_corners=False
    ).squeeze(0)
    
    return img_tensor, (original_height, original_width)


def load_model(model_path, n_classes=44, image_size=512, device_id=2):
    """
    加载训练好的模型
    """
    # 确定设备（优先使用GPU，A100等）
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
        print(f"使用GPU: {torch.cuda.get_device_name(device_id)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(device_id).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    
    # 创建模型
    model = get_model('hg_furukawa_original', n_classes=51)
    
    # 修改输出层以匹配训练时的配置
    model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
    
    # 加载权重（根据设备选择加载位置）
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    
    # 将模型移到指定设备
    model = model.to(device)
    model.eval()
    
    return model, device


def predict_rooms(model, img_tensor, device, split=[21, 12, 11], original_size=None):
    """
    使用模型预测房间类别（使用旋转融合提升精度）
    返回：原始预测结果和完整输出（用于后处理）
    """
    # 添加batch维度
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    
    # 移到指定设备（GPU或CPU）
    img_tensor = img_tensor.to(device)
    
    # 先进行一次推理以获取模型输出尺寸和类别数
    with torch.no_grad():
        test_output = model(img_tensor)
    
    # 确定旋转融合的目标尺寸
    # 如果提供了原始尺寸，使用原始尺寸；否则使用模型输出尺寸
    if original_size is not None:
        height, width = original_size[0], original_size[1]
    else:
        height, width = test_output.shape[2], test_output.shape[3]
    
    img_size = (height, width)
    n_classes = test_output.shape[1]  # 获取类别数
    
    # 旋转融合：使用4个旋转角度进行预测并取平均
    rot = RotateNTurns()
    rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
    pred_count = len(rotations)
    
    # 存储所有旋转的预测结果
    prediction = torch.zeros([pred_count, n_classes, height, width], device=device)
    
    with torch.no_grad():
        for i, r in enumerate(rotations):
            forward, back = r
            # 先旋转图像
            rot_image = rot(img_tensor, 'tensor', forward)
            # 进行预测
            pred = model(rot_image)
            # 将预测结果旋转回来
            pred = rot(pred, 'tensor', back)
            # 修复热图（heatmaps）- 交换热图通道以匹配旋转
            pred = rot(pred, 'points', back)
            # 确保尺寸正确（插值到目标尺寸）
            pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=True)
            # 添加到预测结果中
            prediction[i] = pred[0]
            print(f"完成旋转 {i+1}/{pred_count}: {r}")
    
    # 对所有旋转的预测结果取平均
    prediction = torch.mean(prediction, 0, True)  # (1, n_classes, H, W)
    
    # 提取房间类别部分（split[1]=12个通道）
    # prediction shape: (1, 44, H, W)
    # 房间类别在索引 [split[0]:split[0]+split[1]] = [21:33]
    rooms_logits = prediction[:, split[0]:split[0]+split[1], :, :]  # (1, 12, H, W)
    
    # Softmax得到概率
    rooms_prob = F.softmax(rooms_logits, dim=1)
    
    # Argmax得到类别ID
    rooms_pred = torch.argmax(rooms_prob, dim=1)  # (1, H, W)
    
    # 转回CPU和numpy
    rooms_pred = rooms_pred.cpu().numpy()[0]  # (H, W)
    
    return rooms_pred, rooms_prob.cpu().numpy()[0], prediction, img_size  # 返回预测类别、概率、完整输出和尺寸


def create_color_map(rooms_pred, original_size=None):
    """
    将房间类别ID映射到颜色，生成色块图
    """
    height, width = rooms_pred.shape
    
    # 创建RGB图像
    color_map = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 为每个类别ID分配颜色
    for class_id in range(len(room_classes)):
        mask = (rooms_pred == class_id)
        color_map[mask] = room_colors[class_id]
    
    # 如果需要恢复到原始尺寸
    if original_size is not None:
        color_map = cv2.resize(
            color_map, 
            (original_size[1], original_size[0]), 
            interpolation=cv2.INTER_NEAREST  # 使用最近邻插值保持类别边界清晰
        )
    
    return color_map


def create_polygon_segmentation(prediction, img_size, split=[21, 12, 11], threshold=0.2):
    """
    使用后处理生成矢量多边形分割图
    """
    # 确保 prediction 在 CPU 上（split_prediction 需要 CPU tensor）
    if prediction.is_cuda:
        prediction = prediction.cpu()
    
    # 分割预测结果
    heatmaps, rooms, icons = split_prediction(prediction, img_size, split)
    
    # 获取多边形（all_opening_types=[1, 2] 表示 Window 和 Door）
    polygons, types, room_polygons, room_types = get_polygons(
        (heatmaps, rooms, icons), threshold, [1, 2]
    )
    
    # 将多边形转换为图像
    height, width = img_size
    pol_room_seg, pol_icon_seg = polygons_to_image(
        polygons, types, room_polygons, room_types, height, width
    )
    
    return pol_room_seg, pol_icon_seg


def overlay_on_original(original_img, color_map, alpha=0.6):
    """
    将色块图叠加到原始图片上
    """
    # 确保尺寸一致
    if original_img.shape[:2] != color_map.shape[:2]:
        color_map = cv2.resize(
            color_map, 
            (original_img.shape[1], original_img.shape[0]), 
            interpolation=cv2.INTER_NEAREST
        )
    
    # 叠加
    overlay = cv2.addWeighted(original_img, 1-alpha, color_map, alpha, 0)
    return overlay


def main():
    parser = argparse.ArgumentParser(description='生成户型图的色块特征图')
    parser.add_argument('--image', type=str, required=True,
                        help='输入户型图路径')
    parser.add_argument('--model', type=str, 
                        default='floortrans/models/model_best_val_loss_var.pkl',
                        help='模型权重文件路径')
    parser.add_argument('--image-size', type=int, default=512,
                        help='模型输入图片尺寸')
    parser.add_argument('--n-classes', type=int, default=44,
                        help='模型输出类别数')
    parser.add_argument('--device-id', type=int, default=2,
                        help='GPU设备ID（默认使用第三张卡，索引2）')
    
    args = parser.parse_args()
    
    # 确定输出目录和文件名
    result_dir = '/mnt/workspace/yangxiaohang/cubicase5k/CubiCasa5k/result'
    os.makedirs(result_dir, exist_ok=True)
    
    # 从输入图片路径提取文件名（不含扩展名）
    input_path = Path(args.image)
    base_name = input_path.stem  # 获取不含扩展名的文件名
    
    # 生成输出文件路径
    output_color_map = os.path.join(result_dir, f'{base_name}_color_map.png')
    output_polygon = os.path.join(result_dir, f'{base_name}_color_map_polygon.png')
    output_overlay = os.path.join(result_dir, f'{base_name}_overlay.png')
    output_overlay_polygon = os.path.join(result_dir, f'{base_name}_overlay_polygon.png')
    
    # 初始化颜色映射
    discrete_cmap()
    
    print("正在加载模型...")
    model, device = load_model(args.model, args.n_classes, args.image_size, args.device_id)
    
    print(f"正在加载图片: {args.image}")
    img_tensor, original_size = load_image(args.image, args.image_size)
    
    print("正在进行推理...")
    rooms_pred, rooms_prob, full_prediction, img_size = predict_rooms(
        model, img_tensor, device, split=[21, 12, 11], original_size=original_size
    )
    
    print("正在生成原始预测色块图...")
    color_map = create_color_map(rooms_pred, original_size)
    
    # 保存原始预测色块图
    color_map_bgr = cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_color_map, color_map_bgr)
    print(f"原始预测色块图已保存到: {output_color_map}")
    
    # 生成矢量多边形分割图
    print("正在生成矢量多边形分割图...")
    pol_room_seg, pol_icon_seg = create_polygon_segmentation(
        full_prediction, img_size, split=[21, 12, 11], threshold=0.2
    )
    
    # 将多边形分割结果转换为颜色图
    polygon_color_map = create_color_map(pol_room_seg, original_size)
    
    # 保存矢量多边形色块图
    polygon_color_map_bgr = cv2.cvtColor(polygon_color_map, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_polygon, polygon_color_map_bgr)
    print(f"矢量多边形色块图已保存到: {output_polygon}")
    
    # 生成叠加图
    print("正在生成叠加图...")
    original_img = cv2.imread(args.image)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # 原始预测叠加图
    overlay = overlay_on_original(original_img, color_map)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_overlay, overlay_bgr)
    print(f"原始预测叠加图已保存到: {output_overlay}")
    
    # 矢量多边形叠加图
    polygon_overlay_img = overlay_on_original(original_img, polygon_color_map)
    polygon_overlay_bgr = cv2.cvtColor(polygon_overlay_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_overlay_polygon, polygon_overlay_bgr)
    print(f"矢量多边形叠加图已保存到: {output_overlay_polygon}")
    
    # 打印统计信息
    print("\n房间类别统计:")
    unique, counts = np.unique(rooms_pred, return_counts=True)
    total_pixels = rooms_pred.size
    for class_id, count in zip(unique, counts):
        percentage = (count / total_pixels) * 100
        print(f"  {room_classes[class_id]}: {percentage:.2f}%")


if __name__ == '__main__':
    main()
    # # 打印cuda
    # print(torch.cuda.is_available())
    # device_id = 2  # 使用第三张卡
    # print(f"GPU设备 {device_id}: {torch.cuda.get_device_name(device_id)}")


