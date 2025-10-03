# 13_4_anchors_multibox.py
# 重构并整理自《动手学深度学习》第13.4节：锚框（anchors）、IoU、标注、NMS 与 multibox
# 保持原始算法逻辑不变，增加 main()、设备检测（cuda/mps/cpu）、注释及示例输出展示。
# 说明：此脚本假定环境中已安装 d2l（d2ltorch）和 matplotlib，并且有一张示例图片位于 ../img/catdog.jpg
# 如果示例图片不存在，会使用占位随机图像以便脚本在 PyCharm 中可运行。
import os


import time
import torch
from d2l import torch as d2l
import numpy as np
from PIL import Image



# 精简输出精度（与书中一致）
torch.set_printoptions(2)

# ----------------------------- 工具函数（保持书上实现） -----------------------------

def multibox_prior(data, sizes, ratios):
    """生成以每个像素为中心具有不同形状的锚框
    返回形状: (1, num_anchors, 4)，坐标范围为相对值(0~1): (xmin, ymin, xmax, ymax)
    """
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)
    # 偏移量，将中心移动到像素中心
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height
    steps_w = 1.0 / in_width
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)
    # 宽高（先计算 boxes_per_pixel 个宽、高）
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:]))) * in_height / in_width
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # half width/height
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)


def box_iou(boxes1, boxes2):
    """计算两个边界框列表成对的 IoU（交并比）
    boxes 格式为 (xmin, ymin, xmax, ymax)，取值范围为相对坐标0~1
    返回形状: (len(boxes1), len(boxes2))
    """
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas


def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """将最接近的真实边界框分配给锚框，返回每个锚框分配的真实边界框的索引（无分配为 -1）"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 计算 IoU
    jaccard = box_iou(anchors, ground_truth)
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map


def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """对锚框偏移量进行变换（用于训练标签）
    使用书中给出的缩放因子：x,y *10；w,h 的 log 缩放因子 5
    """
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset


def multibox_target(anchors, labels):
    """使用真实边界框标记锚框，返回 (bbox_offset, bbox_mask, class_labels)
    - bbox_offset: (batch_size, num_anchors*4) 每个锚框的4个偏移量（未分配为0）
    - bbox_mask: (batch_size, num_anchors*4) 掩码，负类的偏移不参与损失
    - class_labels: (batch_size, num_anchors) 类别索引（0为背景，1..为真实类）
    """
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)


def offset_inverse(anchors, offset_preds):
    """根据锚框与预测偏移量恢复预测的边界框（multibox检测时使用）
    返回格式为 (xmin, ymin, xmax, ymax)
    """
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox


def nms(boxes, scores, iou_threshold):
    """非极大值抑制（NMS）：返回保留索引的张量（按 scores 降序）"""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1:
            break
        iou = box_iou(boxes[i, :].reshape(-1, 4), boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)


def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5, pos_threshold=0.009999999):
    """使用非极大值抑制来对预测边界框进行后处理，返回形状 (batch_size, num_anchors, 6)
    每个预测的6个值为: (class_id, confidence, xmin, ymin, xmax, ymax)
    class_id 为 -1 表示背景或被抑制的预测
    """
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        # 对每个锚框找出非背景类的最高置信度及其类别
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1), conf.unsqueeze(1), predicted_bb), dim=1)
        out.append(pred_info)
    return torch.stack(out)


# ----------------------------- main: 调用并展示书上的示例 -----------------------------

def detect_device():
    """检测可用设备：优先 cuda，其次 mps（Apple），否则 cpu"""
    # if torch.cuda.is_available():
    #     return torch.device('cuda')
    # # PyTorch 的 mps 后端（Apple Silicon）检查
    # try:
    #     if getattr(torch, 'has_mps', False) and torch.has_mps:
    #         return torch.device('mps')
    # except Exception:
    #     pass
    return torch.device('cpu')


def load_image_or_placeholder(path, expected_size=None):
    """尝试加载图片；若失败则返回随机占位图像（可在无图像环境下运行）"""
    if os.path.exists(path):
        img = d2l.plt.imread(path)
        h, w = img.shape[:2]
        return img, h, w
    else:
        print(f"警告: 未找到图片 {path}，将使用随机占位图像以确保脚本可运行。")
        if expected_size is None:
            h, w = 561, 728
        else:
            h, w = expected_size
        img = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
        return img, h, w

# 简单包装 show_bboxes（保持与书中一致）
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """显示所有边界框（辅助函数，用于绘图）"""
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj
    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))

def main():
    device = detect_device()
    print(f"Using device: {device}")

    # 载入图片（书中示例路径）
    img_path = os.path.join('..', 'img', 'catdog.jpg')
    img, h, w = load_image_or_placeholder(img_path)
    print('图像高,宽:', h, w)

    # 生成一个随机输入（与书中示例一致）
    X = torch.rand(size=(1, 3, h, w), device=device)
    Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
    print('anchors shape Y:', Y.shape)

    # 将锚框重塑为 (h, w, boxes_per_pixel, 4)
    boxes = Y.reshape(h, w, 5, 4)
    # 输出 (250,250) 位置的第一个锚框（书上示例）
    if h > 250 and w > 250:
        print('boxes[250,250,0,:] =', boxes[250, 250, 0, :])

    # 绘图：展示以某个像素为中心的锚框
    d2l.set_figsize()
    bbox_scale = torch.tensor((w, h, w, h))
    fig = d2l.plt.imshow(img)
    show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
                ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2', 's=0.75, r=0.5'])
    d2l.plt.show()

    # ---------------- 示例：IoU、分配与标签 ----------------
    ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                                 [1, 0.55, 0.2, 0.9, 0.88]], device=device)
    anchors = torch.tensor([[0.0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                            [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                            [0.57, 0.3, 0.92, 0.9]], device=device)
    fig = d2l.plt.imshow(img)
    show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
    show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])
    d2l.plt.show()

    labels = multibox_target(anchors.unsqueeze(dim=0), ground_truth.unsqueeze(dim=0))
    print('\nmultibox_target 返回的类别标签:')
    print(labels[2])  # 类标签
    print('\nmultibox_target 返回的掩码 mask:')
    print(labels[1])
    print('\nmultibox_target 返回的偏移量 offset:')
    print(labels[0])

    # ---------------- 示例：NMS 与 multibox_detection ----------------
    anchors_n = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                              [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]], device=device)
    offset_preds = torch.tensor([0] * anchors_n.numel(), device=device)
    cls_probs = torch.tensor([[0] * 4, [0.9, 0.8, 0.7, 0.1], [0.1, 0.2, 0.3, 0.9]], device=device)

    fig = d2l.plt.imshow(img)
    show_bboxes(fig.axes, anchors_n * bbox_scale, ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
    d2l.plt.show()

    output = multibox_detection(cls_probs.unsqueeze(dim=0),
                                offset_preds.unsqueeze(dim=0),
                                anchors_n.unsqueeze(dim=0),
                                nms_threshold=0.5)
    print('\nmultibox_detection 输出:')
    print(output)

    # 绘制最终保留下来的预测边界框
    fig = d2l.plt.imshow(img)
    for i in output[0].detach().cpu().numpy():
        if i[0] == -1:
            continue
        label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
        show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)
    d2l.plt.show()




if __name__ == '__main__':
    start_time = time.time()
    main()
    # 该脚本不包含训练过程；若后续加入训练，请在训练段落打印训练时间。
    elapsed = time.time() - start_time
    print(f"脚本运行耗时: {elapsed:.2f} 秒")
