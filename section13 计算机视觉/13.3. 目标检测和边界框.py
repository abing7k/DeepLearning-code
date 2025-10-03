import time
import torch
from d2l import torch as d2l

# =========================
# 1. 边界框转换函数
# =========================
# 从 (左上x, 左上y, 右下x, 右下y) → (中心x, 中心y, 宽度, 高度)
def box_corner_to_center(boxes):
    """
    boxes: Tensor [n,4] or [4]，每个框是 (x1, y1, x2, y2)
    return: Tensor [n,4]，每个框是 (cx, cy, w, h)
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

# 从 (中心x, 中心y, 宽度, 高度) → (左上x, 左上y, 右下x, 右下y)
def box_center_to_corner(boxes):
    """
    boxes: Tensor [n,4] or [4]，每个框是 (cx, cy, w, h)
    return: Tensor [n,4]，每个框是 (x1, y1, x2, y2)
    """
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

# =========================
# 2. Matplotlib 辅助绘图
# =========================
def bbox_to_rect(bbox, color):
    """
    将边界框(左上x,左上y,右下x,右下y)格式
    转换成 matplotlib 的 Rectangle 格式
    """
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]),
        width=bbox[2] - bbox[0],
        height=bbox[3] - bbox[1],
        fill=False, edgecolor=color, linewidth=2
    )

# =========================
# 3. 主程序
# =========================
def main():
    start_time = time.time()

    # ---- 设备检测 ----
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # ---- 设置图像显示大小 ----
    d2l.set_figsize()

    # ---- 读取示例图像 ----
    # 确保 ../img/catdog.jpg 路径存在，或修改为你本地图片路径
    img = d2l.plt.imread('../img/catdog.jpg')

    # ---- 显示原始图像 ----
    d2l.plt.imshow(img)
    d2l.plt.title("Original Image")
    d2l.plt.axis('off')
    d2l.plt.show()

    # ---- 定义狗和猫的边界框 ----
    # [左上x, 左上y, 右下x, 右下y]
    dog_bbox = [60.0, 45.0, 378.0, 516.0]
    cat_bbox = [400.0, 112.0, 655.0, 493.0]

    # ---- 验证边界框转换正确性 ----
    boxes = torch.tensor((dog_bbox, cat_bbox))
    print("转换验证结果：")
    print(box_center_to_corner(box_corner_to_center(boxes)) == boxes)

    # ---- 在图像上画出边界框 ----
    fig = d2l.plt.imshow(img)
    fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))  # 狗：蓝色
    fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))   # 猫：红色
    d2l.plt.title("Dog (blue) & Cat (red) Bounding Boxes")
    d2l.plt.axis('off')
    d2l.plt.show()

    # ---- 打印执行时间 ----
    print(f"Execution finished in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
