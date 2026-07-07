#!/usr/bin/env python3
"""
测试知末(Znzmo)生图服务

使用方法:
    cd /mnt/workspace/yangxiaohang/agent/dingzi_code_deepagents_platform
    python scripts/test_znzmo_image.py

环境变量:
    ZNZMO_IMAGE_API_KEY: 知末API密钥（可选）
"""
import os
import sys
import time
import base64
import io
from datetime import datetime
from pathlib import Path

from PIL import Image

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from znzmo_client import ZnzmoClient




def save_image(img, output_dir: str, prefix: str = "znzmo") -> str:
    """保存图片到本地"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    img.save(filepath, "PNG")
    return filepath


def test_text_to_image(client: ZnzmoClient, output_dir: str):
    """测试文生图"""
    print("\n" + "="*60)
    print("测试 1: 文生图 (不带参考图)")
    print("="*60)

    prompt = "一只可爱的橘猫，坐在窗台上晒太阳，插画风格，高清画质"
    model = "nanoBanana"
    batch_size = 2

    print(f"提示词: {prompt}")
    print(f"模型: {model}")
    print(f"生成数量: {batch_size}")
    print("-"*60)

    start_time = time.time()
    try:
        images, message = client.generate_image(
            model=model,
            prompt=prompt,
            batch_size=batch_size,
            aspect_ratio="auto",
            image_size="auto"
        )
        elapsed = time.time() - start_time

        print(f"✅ 生图成功! 耗时: {elapsed:.2f}秒")
        print(f"返回消息: {message}")
        print(f"生成图片数量: {len(images)}")

        # 保存图片
        saved_paths = []
        for i, img in enumerate(images):
            filepath = save_image(img, output_dir, f"znzmo_text2img_{i+1}")
            saved_paths.append(filepath)
            print(f"  图片 {i+1} 已保存: {filepath}")

        return True, saved_paths

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ 生图失败! 耗时: {elapsed:.2f}秒")
        print(f"错误: {e}")
        return False, []


# 户型图门/墙/窗彩色标注提示词。
# 设计原则（通用，不针对单张图打补丁）：
#   ① 原地重新着色，绝不重绘/简化/增删几何 —— 防止模型把整张图“重画”成简版；
#   ② 墙/门/窗→蓝/红/绿实心，其余图元一律保留为黑线 —— 只删文字，不删图形；
#   ③ 所有墙（含最外圈外轮廓）都必须着色 —— 防止漏墙；
#   ④ 显式处理深色/黑底 CAD 源图 —— 转白底但完整保留所有原始图元。
ANNOTATE_PROMPT = (
    "对这张户型图进行『原地重新着色』：严格保持原图的几何结构、比例和视角完全不变，"
    "只改变颜色，绝不重绘、简化、移动、新增或删除任何结构或图元。规则如下：\n"
    "1. 墙体 → 蓝色实心填充：所有承重墙、隔墙，以及建筑最外圈的外轮廓墙，"
    "全部都要填成蓝色，不得遗漏，不得保留为黑色或其它颜色。\n"
    "2. 门 → 红色实心填充：包含门扇和开门弧线所在的整个门洞和平移门。\n"
    "3. 窗 → 绿色实心填充：所有窗户。\n"
    "4. 其余所有图元（家具、洁具、橱柜、楼梯、电梯、家电、阳台构件等）一律保留为"
    "白底黑色线条，必须逐一完整保留，不得删除或简化。\n"
    "5. 只删除文字类信息：房间名称、尺寸标注、标高、轴号、引线、图框、标题栏、"
    "指北针、水印等文字和数字；不要删除任何图形线条。\n"
    "6. 输出为白色背景；若原图是深色/黑色背景的 CAD 图，转为白底后仍须逐一保留全部"
    "原始线条与图元，再按上述规则着色。\n"
    "7. 输出与原图同尺寸、像素对齐——这是对原图重新着色，而不是重新绘制一张图。"
)


def batch_annotate(client: ZnzmoClient, input_dir: str, output_dir: str,
                   model: str = "nanoBanana"):
    """对 input_dir 下的每张户型图批量做门墙窗彩色标注，结果存到 output_dir。"""
    print("\n" + "=" * 60)
    print("批量标注：门墙窗彩色化")
    print("=" * 60)

    exts = {".png", ".jpg", ".jpeg"}
    files = sorted(p for p in Path(input_dir).iterdir() if p.suffix.lower() in exts)
    os.makedirs(output_dir, exist_ok=True)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"待处理 {len(files)} 张: {[f.name for f in files]}")
    print(f"提示词: {ANNOTATE_PROMPT}")

    results = []
    for idx, fp in enumerate(files, 1):
        print("-" * 60)
        print(f"[{idx}/{len(files)}] {fp.name}")
        start = time.time()
        try:
            ref_img = Image.open(fp).convert("RGB")
            images, message = client.generate_image(
                model=model,
                prompt=ANNOTATE_PROMPT,
                batch_size=1,
                aspect_ratio="auto",
                image_size="auto",
                image_list=[ref_img],
            )
            elapsed = time.time() - start
            out_path = os.path.join(output_dir, f"{fp.stem}_annotated.png")
            images[0].save(out_path, "PNG")
            print(f"  ✅ 成功 ({elapsed:.1f}s) -> {out_path}  msg={message}")
            results.append((fp.name, True, out_path))
        except Exception as e:
            elapsed = time.time() - start
            print(f"  ❌ 失败 ({elapsed:.1f}s): {e}")
            results.append((fp.name, False, str(e)))

    ok = sum(1 for _, s, _ in results if s)
    print("=" * 60)
    print(f"完成: {ok}/{len(results)} 成功")
    for name, s, info in results:
        print(f"  {'✅' if s else '❌'} {name}: {info}")
    return results


def test_different_aspect_ratios(client: ZnzmoClient, output_dir: str):
    """测试不同宽高比"""
    print("\n" + "="*60)
    print("测试 3: 不同宽高比")
    print("="*60)

    prompt = "蓝天白云下的草原，几只绵羊在吃草，风景摄影风格"
    model = "nanoBanana"
    aspect_ratios = ["16:9", "4:3", "1:1", "9:16"]

    results = []
    for ratio in aspect_ratios:
        print(f"\n测试宽高比: {ratio}")
        start_time = time.time()
        try:
            images, message = client.generate_image(
                model=model,
                prompt=prompt,
                batch_size=1,
                aspect_ratio=ratio
            )
            elapsed = time.time() - start_time

            filepath = save_image(images[0], output_dir, f"znzmo_ratio_{ratio.replace(':', '_')}")
            print(f"  ✅ 成功! 耗时: {elapsed:.2f}秒, 尺寸: {images[0].size}, 保存: {filepath}")
            results.append((ratio, True, elapsed, images[0].size))

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  ❌ 失败! 耗时: {elapsed:.2f}秒, 错误: {e}")
            results.append((ratio, False, elapsed, None))

    return results


def main():
    print("知末(Znzmo)生图服务测试脚本")
    print("="*60)

    # 配置
    api_url = "https://api.znzmo.cn/ai-draw/third-api/ai-draw-api/dispatch/getAgentResult"
    api_key = os.getenv("ZNZMO_IMAGE_API_KEY", "")
    timeout = 180  # 3分钟超时

    print(f"API URL: {api_url}")
    print(f"API Key: {'已设置' if api_key else '未设置'}")
    print(f"超时: {timeout}秒")

    if not api_url:
        print("❌ 错误: 未配置知末API地址")
        sys.exit(1)

    # 输入/输出目录（脚本同级的 test/ 作为批跑数据源）
    script_dir = Path(__file__).parent
    input_dir = "/Users/yxh/work/CubiCasa5k/test"
    output_dir = os.path.join(script_dir, "test", "annotated")
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")

    # 初始化客户端
    print("\n初始化 ZnzmoClient...")
    try:
        client = ZnzmoClient(
            base_url=api_url,
            api_key=api_key,
            timeout=timeout
        )
        print("✅ 客户端初始化成功")
    except Exception as e:
        print(f"❌ 客户端初始化失败: {e}")
        sys.exit(1)


    # 批量门墙窗彩色标注
    batch_annotate(client, input_dir, output_dir)




if __name__ == "__main__":
    sys.exit(main())
