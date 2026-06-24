import cv2
import os


# 读取YOLO的txt格式标注文件
def parse_yolo_file(file_path, img_path):
    img = cv2.imread(img_path)

    h, w, _ = img.shape

    gt_boxes = []
    with open(file_path, 'r') as file:
        for line in file:
            # 分割行数据
            data = line.strip().split()
            if len(data) != 5:
                continue  # 跳过格式不正确的行

            # 将字符串数据转换为浮点数，并分别赋值给变量
            cls = float(data[0])
            # 检查类别是否为0（我们只关心类别为0的边界框）
            if cls == 0:
                x_, y_, w_, h_ = eval(data[1]), eval(data[2]), eval(data[3]), eval(data[4])
                x1 = w * x_ - 0.5 * w * w_
                x2 = w * x_ + 0.5 * w * w_
                y1 = h * y_ - 0.5 * h * h_
                y2 = h * y_ + 0.5 * h * h_

                # 添加到gt_boxes列表中
                gt_boxes.append((x1, y1, x2, y2))

    return gt_boxes


def calculate_iou(box1, box2):
    """
    计算两个矩形框的交并比（IoU）。
    box1, box2: (x1, y1, x2, y2)
    """
    x1_inter, y1_inter = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2_inter, y2_inter = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    iou = inter_area / float(box1_area + box2_area - inter_area)

    return iou


def fit_rate(gt_boxes, detections):
    # 计算贴合的检测框数量
    fit_count = 0  # 贴合的检测框数量(之前)
    for gt_box in gt_boxes:
        max_iou = 0
        for det_box in detections:
            iou_value = calculate_iou(gt_box, det_box)
            if iou_value > max_iou:
                max_iou = iou_value
        if max_iou >= 0.85:
            fit_count += 1

    fit_rate = fit_count / len(gt_boxes)
    return fit_rate,fit_count
