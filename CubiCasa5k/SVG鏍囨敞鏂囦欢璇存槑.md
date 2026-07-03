# SVG标注文件说明

## 文件结构

SVG文件是XML格式的矢量图形文件，包含了户型图的所有标注信息。

---

## 主要信息类型

### 1. 房间（Space）

**格式**:
```xml
<g id="[唯一ID]" class="Space [房间类型] [子类型]">
    <polygon points="x1,y1 x2,y2 x3,y3 ..."/>
    <g class="SpaceDimensionsLabel">
        <g class="NameLabel">
            <text>房间名称</text>
        </g>
    </g>
</g>
```

**示例** (从model.svg):
```xml
<g id="853e2235-4100-46f2-90a0-8a3af729de74" 
   class="Space LivingRoom">
    <polygon points="1277.38,418.79 741.84,418.79 741.84,418.81 571.67,418.81 571.67,37.56 1277.38,37.56 "/>
    <g class="NameLabel">
        <text>OH</text>  <!-- 房间名称（可选） -->
    </g>
</g>
```

**房间类型**（class属性中的第二个单词）:
- `LivingRoom` - 客厅
- `Kitchen` - 厨房
- `Bedroom` - 卧室
- `Bath` - 浴室
- `Entry` / `Lobby` - 入口/大厅
- `Outdoor` - 户外
- `Balcony` - 阳台
- `Storage` - 储藏室
- `Garage` - 车库
- 等等...

**关键点**:
- `polygon points` 定义了房间的多边形边界（坐标格式：`x,y x,y x,y ...`）
- 房间类型通过 `class="Space [类型]"` 指定
- 可以包含子类型，如 `"Space Outdoor Balcony"` 表示户外阳台

---

### 2. 墙体（Wall）

**格式**:
```xml
<g id="Wall" class="Wall [类型]">
    <polygon points="..."/>
</g>
```

**类型**:
- `External` - 外墙
- 无类型 - 内墙

**示例**:
```xml
<g id="Wall" class="Wall External">
    <polygon points="118.00,21.56 1293.38,21.56 1277.38,37.56 133.97,37.56 "/>
</g>
```

---

### 3. 门（Door）

**格式**:
```xml
<g id="Door" class="Door [类型]">
    <polygon points="..."/>
    <g class="Threshold">...</g>
    <g class="Panel">...</g>
</g>
```

**类型**:
- `Swing Opposite` - 对开门
- `Swing Beside` - 侧开门
- 等等...

**示例**:
```xml
<g id="Door" class="Door Swing Opposite">
    <polygon points="1293.38,326.93 1293.38,412.15 1277.38,412.15 1277.38,326.93 "/>
</g>
```

---

### 4. 窗（Window）

**格式**:
```xml
<g id="Window" class="Window [类型]">
    <polygon points="..."/>
    <g class="Glass">...</g>
</g>
```

**类型**:
- `Regular` - 普通窗
- 等等...

**示例**:
```xml
<g id="Window" class="Window Regular">
    <polygon points="1293.38,45.11 1293.38,319.73 1277.38,319.73 1277.38,45.11 "/>
</g>
```

---

### 5. 固定家具（FixedFurniture）

**格式**:
```xml
<g class="FixedFurniture [类型]">
    <g class="BoundaryPolygon">
        <polygon points="..."/>
    </g>
    <g class="Name">
        <text>名称缩写</text>
    </g>
</g>
```

**常见类型**:
- `Closet` (CL) - 衣柜
- `Sink` (SINK) - 水槽
- `Toilet` - 马桶
- `Shower` - 淋浴
- `Refrigerator` (REF) - 冰箱
- `Dishwasher` (DW) - 洗碗机
- `WashingMachine` (WM) - 洗衣机
- `BaseCabinet` (CB) - 底柜
- `WallCabinet` (UC) - 吊柜
- 等等...

**示例**:
```xml
<g class="FixedFurniture Toilet">
    <g class="BoundaryPolygon">
        <polygon points="0,0 41,0 41,71 0,71"/>
    </g>
</g>
```

---

### 6. 栏杆（Railing）

**格式**:
```xml
<g id="Railing" class="Railing">
    <polygon points="..."/>
</g>
```

---

## 房间类型映射

代码中使用的房间类型映射（`floortrans/loaders/house.py`）：

### 所有房间类型（all_rooms）
完整的房间类型列表包含60+种类型，从"Background"到"UserDefined"。

### 选中的房间类型（rooms_selected）
实际使用的12种房间类型（映射到类别0-11）：
- 0: Background
- 1: Outdoor
- 2: Wall
- 3: Kitchen
- 4: LivingRoom
- 5: Bedroom
- 6: Bath
- 7: Entry
- 8: Railing
- 9: Storage
- 10: Garage
- 11: Undefined

---

## 如何进行标注

### 方法1: 使用SVG编辑器（推荐）

1. **使用Inkscape或Adobe Illustrator**
   - 打开SVG文件
   - 使用多边形工具绘制房间边界
   - 设置class属性为 `"Space [房间类型]"`

2. **标注步骤**:
   ```
   1. 选择多边形工具
   2. 绘制房间轮廓（点击多个点形成闭合多边形）
   3. 选中多边形，查看属性
   4. 设置class为 "Space LivingRoom"（或"Space Kitchen"等）
   5. 保存文件
   ```

### 方法2: 直接编辑SVG文件（高级用户）

1. **添加新房间**:
   ```xml
   <g id="unique-id-here" class="Space Kitchen">
       <polygon points="x1,y1 x2,y2 x3,y3 x4,y4 x1,y1"/>
       <g class="SpaceDimensionsLabel">
           <g class="NameLabel">
               <text>K</text>
           </g>
       </g>
   </g>
   ```

2. **修改房间类型**:
   - 找到对应的 `<g>` 标签
   - 修改 `class` 属性，例如：
     - `class="Space LivingRoom"` → `class="Space Kitchen"`
   
3. **修改房间边界**:
   - 找到 `<polygon points="..."/>`
   - 修改points属性中的坐标

### 方法3: 使用标注工具

如果你有CubiCasa的原始标注工具，可以直接使用。

---

## 标注注意事项

### 1. 坐标系统
- SVG使用像素坐标
- 原点(0,0)在左上角
- X轴向右，Y轴向下

### 2. 多边形格式
- 坐标格式：`"x1,y1 x2,y2 x3,y3 ..."`
- 最后一个点通常与第一个点相同（形成闭合）
- 坐标值可以是小数

### 3. 房间类型命名
- 必须使用代码中定义的类型名
- 区分大小写：`LivingRoom` 不是 `livingroom`
- 如果类型不存在，会被映射到 `Undefined`

### 4. 墙体处理
- 墙体不应该被包含在房间多边形内
- 房间多边形应该紧贴墙体内侧

### 5. 重叠处理
- 不同房间之间不应重叠
- 房间和墙体不应重叠（除非是门）

---

## 示例：如何标注一个新房间

假设你要标注一个厨房：

```xml
<g id="kitchen-001" class="Space Kitchen">
    <!-- 厨房多边形：四个点的矩形 -->
    <polygon points="100,100 300,100 300,200 100,200 100,100"/>
    
    <!-- 房间标签（可选） -->
    <g transform="matrix(1,0,0,1,200,150)" class="SpaceDimensionsLabel">
        <g class="NameLabel">
            <text x="0" y="0em" fill="#000000">K</text>
        </g>
    </g>
</g>
```

---

## 代码如何读取SVG

查看 `floortrans/loaders/svg_utils.py` 和 `floortrans/loaders/house.py`：

1. **解析XML**: 使用 `minidom.parse(path)` 解析SVG
2. **查找房间**: 搜索 `class` 包含 `"Space "` 的元素
3. **提取多边形**: 从 `<polygon>` 标签提取 `points` 属性
4. **映射类别**: 使用 `get_room_number()` 将房间类型名称映射到类别ID
5. **生成掩码**: 使用 `skimage.draw.polygon()` 将多边形转换为像素掩码

---

## 快速参考

### 常用房间类型
- `Space LivingRoom` - 客厅
- `Space Kitchen` - 厨房  
- `Space Bedroom` - 卧室
- `Space Bath` - 浴室
- `Space Entry` - 入口
- `Space Storage` - 储藏室
- `Space Garage` - 车库
- `Space Outdoor` - 户外

### 常用图标类型
- `FixedFurniture Toilet` - 马桶
- `FixedFurniture Sink` - 水槽
- `FixedFurniture Shower` - 淋浴
- `FixedFurniture Refrigerator` - 冰箱

---

## 验证标注

使用代码验证标注是否正确：

```python
from floortrans.loaders.house import House
from floortrans.loaders.svg_utils import get_labels

# 读取SVG并验证
house = House('path/to/model.svg', height=512, width=512)
walls, icons = get_labels('path/to/model.svg', height=512, width=512)

# 检查房间类型
print("房间类型:", house.room_types)
print("图标类型:", house.icon_types)
```

