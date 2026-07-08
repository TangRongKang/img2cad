# 户型图 → DXF 端到端转换流程说明

## 概述

本项目的核心 pipeline 把一张普通黑白/灰度户型图转换为 CAD 可用的 DXF 文件：

```
input.png
    ↓  知末 AI 原地重新着色（4K）
input_annotated.png
    ↓  按颜色语义提取、正则化、矢量化
input.dxf + input_preview.png
```

最终 DXF 包含三个独立图层：

| 颜色标注 | DXF 图层 | 含义 |
|---------|---------|------|
| 蓝色 | `WALLS` | 墙体（闭合多段线） |
| 绿色 | `WINDOWS` | 窗户（闭合矩形/L 形 + 内部双线） |
| 黑色/深色（含门、家具等） | `DETAILS` | 门、家具、洁具、楼梯等细节 |

> **注意**：当前版本已**不再单独识别门**，门不再被标注为红色，而是与家具等其它深色图元一起进入 `DETAILS` 层处理。

---

## 相关文件

| 文件 | 作用 |
|------|------|
| `floorplan_to_dxf.py` | 入口脚本：调用知末 API 生成彩色标注图，再转 DXF |
| `color_annotated_to_dxf.py` | 核心转换库：颜色分割 → 墙体/窗户/细节提取 → DXF 输出 |
| `znzmo_client.py` | 知末 API 客户端封装 |
| `render_dxf_previews.py` | 把已有 `.dxf` 批量渲染为 `_preview.png` |

---

## 运行方式

### 1. 自动调用 AI 标注（默认）

```bash
python floorplan_to_dxf.py image/test_image_01.png --width-mm 13700
```

- `--width-mm`：户型图实际宽度（毫米），默认 `13700.0`。
- 会先向知末 API 请求一张 4K 彩色标注图，保存为 `test_image_01_annotated.png`。
- 再基于该标注图生成 `test_image_01.dxf` 和 `test_image_01_preview.png`。

### 2. 使用已有的彩色标注图

```bash
python floorplan_to_dxf.py image/test_image_01.png \
    --annotated image/test_image_01_annotated.png \
    --width-mm 13700
```

### 3. 直接转换彩色标注图

```bash
python color_annotated_to_dxf.py image/test_image_01_annotated.png \
    --width-mm 13700 -o output/test_image_01.dxf
```

### 4. 跳过细节层

```bash
python floorplan_to_dxf.py image/test_image_01.png --width-mm 13700 --no-details
```

---

## 第一步：AI 彩色标注

`floorplan_to_dxf.py` 通过 `ZnzmoClient` 调用知末 API，使用固定的 `ANNOTATE_PROMPT` 对原图进行**原地重新着色**。着色规则如下：

1. **墙体 → 蓝色实心填充**：包括承重墙、隔墙、建筑最外圈外轮廓墙，不得遗漏。
2. **窗户 → 绿色实心填充**：所有窗户。
3. **门不单独标注颜色**：保持为白底黑色线条，与家具等图元一起处理。
4. **其余图元保留为白底黑色线条**：家具、洁具、橱柜、楼梯、电梯、家电、阳台构件等必须逐一完整保留。
5. **只删除文字类信息**：房间名称、尺寸标注、标高、轴号、引线、图框、标题栏、指北针、水印等。
6. **深色/黑色 CAD 源图**：先转为白底，再保留全部原始线条并按规则着色。
7. **输出与原图同尺寸、像素对齐**：是对原图重新着色，而不是重新绘制。

API 参数：

- `image_size='4k'`：使用高清输出，以提升 DXF 元素精度。
- 返回的标注图尺寸可能不等于输入图尺寸；后续比例尺会根据标注图自身的像素宽度重新计算。

---

## 第二步：颜色分割与掩模预处理

`color_annotated_to_dxf.py` 读取标注图后，在 HSV 空间提取颜色：

```python
BLUE_LO  = [ 95, 100,  80] ; BLUE_HI  = [145, 255, 255]
GREEN_LO = [ 40,  80,  80] ; GREEN_HI = [ 85, 255, 255]
```

### 墙体（蓝色）

- `morph_clean(..., close_k=3, open_k=3, close_iter=1)`：小核闭运算 + 开运算，填充细小断口并去除孤立噪点。
- 同时保留两份掩模：
  - `blue_w`：较“锋利”的掩模，用于后续细节层排除。
  - `blue_c`：高斯模糊（`sigma=2.0`）后的平滑掩模，用于墙体轮廓提取和墙厚估计。

### 窗户（绿色）

- `morph_clean(..., close_k=5, open_k=3, close_iter=2)`：略大的闭运算核，用于连接窗框断口。
- 输出 `green_c` 用于后续窗户检测。

---

## 第三步：比例尺与墙厚估计

从蓝色掩模中统计像素范围：

- `x0 = 最左蓝色像素 x`
- `y0 = 最下蓝色像素 y`
- `plan_w = 最右 - 最左蓝色像素 x`

```
scale = width_mm / plan_w    # mm/px
```

坐标转换时以平面图左下角为原点，Y 轴翻转：

```python
mm_x = (px_x - x0) * scale
mm_y = (y0 - px_y) * scale
```

墙厚 `wall_t` 通过扫描线法估计蓝色掩模上水平/垂直 run-length 的中位数得到，用于后续所有正则化的容差。

---

## 第四步：墙体（WALLS）矢量化

### 4.1 轮廓提取

- `cv2.findContours(blue_c, RETR_CCOMP, CHAIN_APPROX_SIMPLE)`
- 过滤小轮廓（`min_area=400`）。

### 4.2 顶点简化

- `cv2.approxPolyDP`
- `eps = max(wall_t * 0.12, 3.0)`：去除像素锯齿，但保留门套、窗套等真实拐角。

### 4.3 方向正则化

`regularize_dirs(...)` 把每条边吸附到最近的 `0° / 45° / 90° / 135°` 方向：

- 夹角容差 `ang_tol = 18°`；偏离过大的边保持原方向。
- 每个顶点重新计算为相邻两条吸附后边直线的交点。
- `max_pull = max(wall_t * 0.6, 4.0)`：防止短斜边/毛刺导致顶点飞出原始轮廓形成尖刺。

### 4.4 坐标聚类对齐

仅对正交顶点（相邻两边均为水平/垂直）的 X/Y 坐标做聚类：

- 容差 `tol = max(2, min(round(wall_t * 0.15), 6))` 像素。
- 将相近坐标统一为均值，消除相邻墙段之间的 1–2px 偏差。

### 4.5 输出

- 闭合 `LWPOLYLINE`，图层 `WALLS`，颜色 7（白色/黑色），线宽 0.35mm。
- 同时输出 `x_lines`、`y_lines` 作为“墙格线”，供窗户端面吸附。

---

## 第五步：窗户（WINDOWS）矢量化

`trace_windows(green_c, x_lines, y_lines, wall_t)` 对每个绿色区域分类处理。

### 5.1 直窗（轴对齐，填充率 ≥ 0.65）

`_straight_window(...)`：

- 将窗户矩形的两对面吸附到最近的墙格线（`face_tol = 0.9 * wall_t`）。
- 将两端吸附到墙格线（`end_tol = 0.7 * wall_t`）。
- 输出闭合矩形 `poly`。
- 在厚度方向的 **1/3 和 2/3 处**绘制两条与长边平行的内部线（窗户符号惯例）。

### 5.2 旋转直窗

- 若长轴偏离正交网格超过 `15°`，且旋转矩形填充率 ≥ 0.55，则使用 `cv2.minAreaRect` 提取旋转矩形框。
- 输出旋转矩形轮廓 + 中心单线。

### 5.3 L 形转角窗（填充率 < 0.65）

1. 通过象限像素计数找到 L 形的“空缺角”。
2. 计算水平臂厚度 `t_h` 和垂直臂厚度 `t_v`。
3. 分别对两臂调用 `_straight_window`，得到两个矩形。
4. 用 `shapely` 取并集，合并为一个 6 顶点 L 形闭合多边形。
5. 输出连续 L 形中心线：一臂远端 → 转角 → 另一臂远端。

### 5.4 输出

- 闭合 `LWPOLYLINE`（窗户外轮廓），图层 `WINDOWS`，颜色 4（青色），线宽 0.25mm。
- 内部双线/中心线作为非闭合 `LWPOLYLINE` 输出到同一图层。

---

## 第六步：门（当前已取消独立图层）

当前 pipeline 中：

- AI 标注提示词已明确要求**门不单独着色**。
- `convert_annotated_image()` 中 `doors = []`。
- DXF 中**不再输出 `DOORS` 图层**。
- 门扇、门洞弧线等所有门相关线条均作为普通深色内容，随 `DETAILS` 层一起处理。

因此，门的几何结构与家具、洁具等保持一致风格：单线中心线/轮廓，经骨架化、连接、共线合并、正交修正后输出。

---

## 第七步：细节层（DETAILS）矢量化

### 7.1 细节掩模

`make_detail_mask(img_bgr, blue_w, green_c)`：

1. 灰度化原图，`cv2.threshold(gray, 220, 255, THRESH_BINARY_INV)` 提取所有深色前景。
2. 对墙/窗掩模分别做 `15×15` 矩形膨胀，覆盖墙/窗周围的反走样深色光晕。
3. 用 `bitwise_and(fg, not(color_union))` 得到纯细节掩模。
4. 轻微 `2×2` 闭运算，弥合细线 1px 断口。

### 7.2 骨架化

- `skimage.morphology.skeletonize` 将粗笔画细化为 1 像素宽中心线。
- 避免旧方法中“提取轮廓边界”导致的双线问题。

### 7.3 骨架 → 多段线

`_skeleton_to_polylines(...)`：

- 计算每个骨架像素的 8 邻域度数，识别端点（度 1）和分叉点（度 > 2）。
- 从端点出发追踪到下一个端点/分叉点。
- 处理分叉点之间的分支。
- 处理孤立闭合环。
- 过滤过短路径（默认 `min_length=1`）。

### 7.4 后处理

1. `_reconnect_polylines(...)`：在 `5px` 距离内且端点方向大致相对时，重新连接断开的细线。
2. `cv2.approxPolyDP`：用 `eps_frac=0.001` 轻微平滑曲线，保留真实拐角。
3. `_merge_collinear_segments(..., angle_tol_deg=10.0)`：合并连续几乎共线的边，去除骨架锯齿。
4. `_orthogonalize_details(..., angle_tol_deg=2.0)`：把偏离水平/垂直不超过 `2°` 的边修正为严格正交。

### 7.5 输出

- 若首末点重合则输出闭合 `LWPOLYLINE`，否则输出非闭合 `LWPOLYLINE`。
- 图层 `DETAILS`，颜色 8（灰色），线宽 0.13mm。

---

## 第八步：DXF 输出与预览

### DXF

`emit_dxf(...)` 创建 ezdxf R2010 文档：

- 单位：`doc.units = 4`（毫米）。
- `$LTSCALE = 1`。
- 图层：

| 图层 | 颜色 | 线宽 |
|------|------|------|
| `WALLS`   | 7 | 0.35mm |
| `WINDOWS` | 4 | 0.25mm |
| `DETAILS` | 8 | 0.13mm |

所有坐标均已按第三步的 `scale`、`x0`、`y0` 转换为毫米。

### 预览 PNG

`render_preview(...)` 生成与标注图同尺寸的预览：

- 墙体：黑色，2px
- 窗户：绿色，2px
- 细节：灰色，1px

文件名：`{output_dxf_prefix}_preview.png`。

---

## 第九步：批量 DXF → 预览图

若已有 `jingpin/` 等文件夹的 `.dxf` 需要预览：

```bash
python render_dxf_previews.py jingpin/
```

会为每个 `.dxf` 生成同名的 `_preview.png`。

---

## 各层处理策略对比

| 图层 | 提取方式 | 简化/平滑 | 正则化 | 实体类型 |
|------|---------|-----------|--------|---------|
| `WALLS`   | 蓝色掩模轮廓 | `approxPolyDP` + 高斯平滑 | 正交化（0/45/90/135°）+ 坐标聚类 | 闭合 `LWPOLYLINE` |
| `WINDOWS` | 绿色掩模轮廓 | — | 吸附到墙格线 | 闭合 `LWPOLYLINE` + 内部双线 |
| `DETAILS` | 骨架化深色内容 | `approxPolyDP` + 共线合并 | 2° 内修正为正交 | 闭合/非闭合 `LWPOLYLINE` |

---

## 常见问题速查

| 现象 | 可能原因 / 处理 |
|------|----------------|
| 墙体边缘出现一圈细节线 | 墙/窗掩模膨胀不够；当前已用 `15×15` 膨胀排除光晕 |
| 窗户内部只有一条线 | 旧逻辑；当前已改为 1/3 处双内平行线 |
| 门没有单独图层 | 这是预期行为；门已并入 `DETAILS` |
| 细节层有锯齿/双线 | 已用骨架化 + 共线合并 + 正交修正处理 |
| 输出 DXF 尺寸不对 | 检查 `--width-mm` 或 `--scale`；默认宽度为 `13700mm` |
