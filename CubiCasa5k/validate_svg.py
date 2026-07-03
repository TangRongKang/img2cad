#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证SVG标注文件是否正确
"""
import argparse
import os
import sys
from pathlib import Path

try:
    from floortrans.loaders.svg_loader import FloorplanSVG
    import cv2
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在正确的conda环境中运行此脚本")
    sys.exit(1)


def validate_svg_folder(folder_path, data_root):
    """
    验证单个文件夹的SVG标注
    
    Args:
        folder_path: 相对于data_root的文件夹路径
        data_root: 数据根目录
    """
    print(f"\n{'='*60}")
    print(f"验证文件夹: {folder_path}")
    print(f"{'='*60}")
    
    full_path = os.path.join(data_root, folder_path.lstrip('/'))
    
    # 检查必需文件
    required_files = ['F1_original.png', 'F1_scaled.png', 'model.svg']
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(full_path, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
            print(f"❌ 缺少文件: {file}")
        else:
            print(f"✅ 找到文件: {file}")
    
    if missing_files:
        print(f"\n⚠️  缺少以下文件: {', '.join(missing_files)}")
        return False
    
    # 尝试加载数据
    try:
        # 创建临时txt文件
        temp_txt = '/tmp/temp_validation.txt'
        with open(temp_txt, 'w') as f:
            f.write(folder_path + '\n')
        
        # 加载数据
        data_loader = FloorplanSVG(data_root, temp_txt, format='txt')
        sample = data_loader[0]
        
        print(f"\n✅ 数据加载成功!")
        print(f"   - 图片形状: {sample['image'].shape}")
        print(f"   - 标签形状: {sample['label'].shape}")
        print(f"   - 文件夹: {sample['folder']}")
        
        # 检查标签值范围
        label = sample['label']
        if label.dim() == 3:
            room_label = label[0].numpy()
            icon_label = label[1].numpy() if label.shape[0] > 1 else None
            
            print(f"\n   房间标签统计:")
            unique_rooms = set(room_label.flatten().tolist())
            print(f"   - 唯一值: {sorted(unique_rooms)}")
            print(f"   - 值范围: {room_label.min()} - {room_label.max()}")
            
            if icon_label is not None:
                print(f"\n   图标标签统计:")
                unique_icons = set(icon_label.flatten().tolist())
                print(f"   - 唯一值: {sorted(unique_icons)}")
                print(f"   - 值范围: {icon_label.min()} - {icon_label.max()}")
        
        # 清理临时文件
        os.remove(temp_txt)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_dataset_list(txt_file, data_root):
    """
    验证数据集列表中的所有样本
    
    Args:
        txt_file: 数据集列表文件路径
        data_root: 数据根目录
    """
    if not os.path.exists(txt_file):
        print(f"❌ 文件不存在: {txt_file}")
        return
    
    print(f"\n{'='*60}")
    print(f"验证数据集列表: {txt_file}")
    print(f"{'='*60}")
    
    # 读取文件夹列表
    with open(txt_file, 'r') as f:
        folders = [line.strip() for line in f if line.strip()]
    
    print(f"\n找到 {len(folders)} 个样本")
    
    success_count = 0
    fail_count = 0
    
    for i, folder in enumerate(folders, 1):
        print(f"\n[{i}/{len(folders)}] 验证: {folder}")
        if validate_svg_folder(folder, data_root):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"验证完成!")
    print(f"   ✅ 成功: {success_count}/{len(folders)}")
    print(f"   ❌ 失败: {fail_count}/{len(folders)}")
    print(f"{'='*60}")


def create_svg_template(output_path, width=512, height=512):
    """
    创建SVG标注模板
    
    Args:
        output_path: 输出文件路径
        width: 图片宽度
        height: 图片高度
    """
    template = f'''<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" 
     height="{height}" 
     width="{width}" 
     viewBox="0 0 {width} {height}">
    <defs/>
    <g id="Model" class="Model v1-1">
        <g class="Floor">
            <g id="Floor-1" class="Floorplan Floor-1">
                <!-- 示例: 添加一个厨房房间 -->
                <g id="room-1" class="Space Kitchen" fill="#ffffff" stroke="#ffffff" 
                   style="fill-opacity: 1; stroke-opacity: 1; stroke-width: 0.2;">
                    <polygon points="100,100 300,100 300,300 100,300"/>
                </g>
                
                <!-- 示例: 添加墙体 -->
                <g id="Wall" class="Wall External" fill="#000000" stroke="#000000" 
                   style="fill-opacity: 1; stroke-opacity: 1; stroke-width: 0.2;">
                    <polygon points="0,0 {width},0 {width},{height} 0,{height}"/>
                </g>
                
                <!-- 示例: 添加窗户 -->
                <g id="Window" class="Window Regular" fill="#f0f0ff" stroke="#000000" 
                   style="fill-opacity: 1; stroke-width: 1; stroke-opacity: 1;">
                    <polygon points="200,0 250,0 250,50 200,50"/>
                </g>
                
                <!-- 示例: 添加门 -->
                <g id="Door" class="Door Swing Beside" fill="#ffffff" stroke="#000000" 
                   style="fill-opacity: 1; stroke-width: 1; stroke-opacity: 1;">
                    <polygon points="150,0 200,0 200,50 150,50"/>
                </g>
            </g>
        </g>
    </g>
</svg>'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(template)
    
    print(f"✅ SVG模板已创建: {output_path}")
    print(f"\n支持的房间类型:")
    print("  - Kitchen, LivingRoom, Bedroom, Bath")
    print("  - Entry, Storage, Garage, Outdoor")
    print("  - Wall, Railing, Undefined")


def main():
    parser = argparse.ArgumentParser(description='验证SVG标注文件')
    parser.add_argument('--mode', type=str, choices=['validate', 'template', 'list'],
                       default='validate',
                       help='运行模式: validate(验证), template(创建模板), list(验证列表)')
    parser.add_argument('--folder', type=str,
                       help='要验证的文件夹路径（相对于data-root）')
    parser.add_argument('--txt', type=str,
                       help='数据集列表文件路径（train.txt, val.txt等）')
    parser.add_argument('--data-root', type=str, default='data/cubicasa5k/',
                       help='数据根目录')
    parser.add_argument('--template-output', type=str,
                       help='SVG模板输出路径')
    parser.add_argument('--width', type=int, default=512,
                       help='模板图片宽度')
    parser.add_argument('--height', type=int, default=512,
                       help='模板图片高度')
    
    args = parser.parse_args()
    
    if args.mode == 'validate':
        if not args.folder:
            print("❌ 请使用 --folder 指定要验证的文件夹")
            return
        validate_svg_folder(args.folder, args.data_root)
    
    elif args.mode == 'template':
        if not args.template_output:
            print("❌ 请使用 --template-output 指定输出路径")
            return
        create_svg_template(args.template_output, args.width, args.height)
    
    elif args.mode == 'list':
        if not args.txt:
            print("❌ 请使用 --txt 指定数据集列表文件")
            return
        validate_dataset_list(args.txt, args.data_root)


if __name__ == '__main__':
    main()

