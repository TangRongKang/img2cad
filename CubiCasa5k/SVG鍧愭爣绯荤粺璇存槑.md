# SVG坐标系统说明

## 重要结论

**SVG中的坐标不是直接的原图像素坐标，而是SVG自己的坐标系统。代码会根据图像尺寸自动进行缩放映射。**

---

## SVG坐标系统

### 1. SVG的viewBox和尺寸

查看SVG文件的第一行：
```xml
<svg ... height="871.0599994659424" width="1476.150016784668" 
     viewBox="0 0 1476.150016784668 871.0599994659424">
```

这表示：
- SVG的**逻辑坐标系统**范围是 `0 0 1476.15 871.06`
- 所有`<polygon points="...">`中的坐标都在这个逻辑坐标系中

### 2. 实际图像尺寸

数据集中有两种图像：
- **F1_scaled.png**: 通常是512x512或类似尺寸（用于训练）
- **F1_original.png**: 原始尺寸（可能更大，如1476x871）

---

## 代码如何处理坐标映射

### 关键代码流程

1. **读取图像** (`svg_loader.py`):
   ```python
   fplan = cv2.imread('F1_scaled.png')
   height, width, nchannel = fplan.shape  # 例如: 512, 512
   ```

2. **创建House对象** (`house.py`):
   ```python
   house = House(svg_path, height, width)  # 传入图像的实际尺寸
   ```

3. **解析SVG坐标** (`svg_utils.py`):
   ```python
   def get_polygon(e):
       points = pol.getAttribute("points").split(' ')
       # 例如: "1293.38,36.56 1448.57,36.56 ..."
       for a in points:
           y, x = a.split(',')  # 注意：SVG是x,y，代码转换为y,x
           X = np.append(X, np.round(float(x)))  # 行坐标
           Y = np.append(Y, np.round(float(y)))  # 列坐标
       rr, cc = polygon(X, Y)  # 转换为像素掩码
       return rr, cc
   ```

4. **坐标裁剪** (`house.py`):
   ```python
   def _clip_outside(self, rr, cc):
       # 将超出图像边界的坐标裁剪到图像范围内
       rr = np.clip(rr, 0, self.height - 1)
       cc = np.clip(cc, 0, self.width - 1)
       return rr, cc
   ```

---

## 坐标映射关系

### 情况1: 使用F1_scaled.png（训练时）

假设：
- SVG viewBox: `1476.15 x 871.06`
- F1_scaled.png: `512 x 512`

**代码会自动将SVG坐标映射到512x512图像**：
- SVG坐标 `(1293.38, 36.56)` → 图像坐标 `(约448, 约21)`
- 映射公式：`图像坐标 = SVG坐标 × (图像尺寸 / SVG尺寸)`

### 情况2: 使用F1_original.png（原始尺寸）

如果使用`original_size=True`：
- 代码先读取F1_scaled.png生成标签
- 然后读取F1_original.png
- 使用插值将标签从scaled尺寸缩放到original尺寸

```python
if self.original_size:
    # 读取原图
    fplan = cv2.imread('F1_original.png')
    height_org, width_org = fplan.shape[:2]
    
    # 缩放标签
    label = torch.nn.functional.interpolate(
        label, size=(height_org, width_org), mode='nearest'
    )
    
    # 缩放热图坐标
    coef_width = float(width_org) / float(width)
    coef_height = float(height_org) / float(height)
    for key, value in heatmaps.items():
        heatmaps[key] = [(int(round(x*coef_width)), 
                         int(round(y*coef_height))) 
                        for x, y in value]
```

---

## 如何标注

### 方法1: 直接使用SVG坐标系统（推荐）

**不需要考虑原图尺寸**，直接在SVG的逻辑坐标系中标注：

1. 打开SVG文件（使用Inkscape等工具）
2. SVG会自动显示在viewBox坐标系中
3. 绘制多边形时，坐标会自动在SVG坐标系中
4. 保存后，代码会根据实际图像尺寸自动映射

**示例**：
```xml
<!-- SVG viewBox是 1476.15 x 871.06 -->
<g class="Space Kitchen">
    <polygon points="100,100 300,100 300,200 100,200 100,100"/>
    <!-- 这些坐标在SVG坐标系中，代码会自动映射到图像尺寸 -->
</g>
```

### 方法2: 根据原图尺寸计算（不推荐）

如果你知道原图尺寸，可以手动计算：

```python
# 假设原图是 1476 x 871
# SVG viewBox是 1476.15 x 871.06
# 那么坐标基本1:1对应

# 但如果是512x512的图像：
svg_x = 1293.38
svg_y = 36.56
image_x = svg_x * (512 / 1476.15)  # ≈ 448
image_y = svg_y * (512 / 871.06)   # ≈ 21
```

**但这种方法不推荐**，因为：
- 不同图像可能有不同的缩放比例
- 代码已经自动处理了映射
- 容易出错

---

## 重要注意事项

### 1. SVG坐标格式

SVG中坐标格式是 `"x,y x,y x,y ..."`（注意是逗号分隔，空格分隔点）

代码解析时：
```python
y, x = a.split(',')  # 注意：代码中x和y是反的！
# 这是因为图像坐标系：行(y)在前，列(x)在后
```

### 2. 坐标裁剪

如果SVG坐标超出图像边界，代码会自动裁剪：
```python
rr = np.clip(rr, 0, self.height - 1)
cc = np.clip(cc, 0, self.width - 1)
```

### 3. 坐标精度

SVG坐标可以是小数（如`1293.38`），代码会四舍五入：
```python
X = np.append(X, np.round(float(x)))
```

---

## 实际标注建议

### 推荐工作流程

1. **使用Inkscape打开SVG文件**
   - Inkscape会自动使用SVG的viewBox作为坐标系
   - 你可以直接看到标注区域

2. **同时打开对应的图像文件作为参考**
   - 在Inkscape中导入图像作为背景层
   - 确保图像和SVG对齐

3. **绘制多边形**
   - 使用多边形工具绘制房间边界
   - 坐标会自动保存在SVG坐标系中

4. **设置房间类型**
   - 选中多边形，设置`class="Space Kitchen"`等

5. **保存SVG文件**
   - 代码读取时会自动处理坐标映射

### 验证标注

可以使用代码验证标注是否正确：

```python
from floortrans.loaders.house import House
import cv2

# 读取图像和SVG
img = cv2.imread('F1_scaled.png')
height, width = img.shape[:2]

house = House('model.svg', height, width)

# 检查房间类型
print("房间类型:", house.room_types)
print("图像尺寸:", height, width)
print("SVG坐标已映射到图像尺寸")
```

---

## 总结

1. **SVG坐标不是原图像素坐标**，而是SVG自己的逻辑坐标系
2. **代码会自动映射**：根据图像实际尺寸，将SVG坐标映射到图像像素
3. **标注时不需要考虑原图尺寸**：直接在SVG坐标系中标注即可
4. **推荐使用Inkscape**：它会自动处理坐标系，你只需要绘制多边形

**简单回答**：
- ❌ 不是原图坐标
- ✅ 是SVG的逻辑坐标系统
- ✅ 代码会自动映射到图像尺寸
- ✅ 标注时直接使用SVG坐标系即可

