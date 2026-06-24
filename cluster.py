import os
from datetime import datetime

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from fit_rate import parse_yolo_file, fit_rate, calculate_iou
from plot import plot_clustering_dendrogram
from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from iou_stats import process_third_model, generate_combined_iou_histogram
from batch_error_calculator import init_batch_errors, add_filtered_error, add_supplementary_error, calculate_batch_errors

# 全局变量用于统计所有图片的IOU分布
all_iou_before_filtering = []
all_iou_after_completion = []
all_iou_third_model = []  # 新增第三个model的IOU统计




def calculate_cluster_centers(coords, labels):
    """
    计算簇中心点

    参数:
    coords -- 坐标数据，形状为(n_samples, 1)
    labels -- 聚类标签

    返回:
    (坐标, 有效簇数量, 重新标记的标签, 簇中心点, 保留的索引)
    """
    unique_labels, counts = np.unique(labels, return_counts=True)

    # 找出所有簇
    valid_labels = unique_labels

    # 所有点都是有效点
    mask = np.isin(labels, valid_labels)
    filtered_coords = coords[mask]
    filtered_labels = labels[mask]
    kept_indices = np.where(mask)[0]  # 记录保留点的原始索引

    # 计算每个簇的中心点
    centroids = []
    for label in valid_labels:
        cluster_points = coords[labels == label]  # 使用原始数据计算中心点
        centroids.append(np.mean(cluster_points))

    # 将中心点按大小排序
    sorted_indices = np.argsort(np.array(centroids).flatten())
    sorted_centroids = np.array(centroids)[sorted_indices]

    # 重新标记簇标签为连续整数
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(valid_labels[sorted_indices])}
    new_labels = np.array([label_mapping[label] for label in filtered_labels])

    return filtered_coords, len(valid_labels), new_labels, sorted_centroids, kept_indices


def find_optimal_clusters(data):
    """
    通过轮廓系数选择最佳聚类数量
    从叶子节点开始向上聚合(从多簇到少簇)

    参数:
    data -- 要聚类的数据

    返回:
    最佳聚类数量
    """
    if len(data) <= 2:
        return 1

    max_clusters = min(30, len(data) - 1)
    scores = []

    # 从最大簇数开始向下尝试到1个簇（包含1个簇）
    for n in range(max_clusters, 0, -1):
        clustering = AgglomerativeClustering(n_clusters=n).fit(data)
        labels = clustering.labels_

        # 对于1个簇的情况，轮廓系数为0，直接记录
        if n == 1:
            scores.append({
                'n_clusters': n,
                'silhouette': 0.0  # 1个簇时轮廓系数为0
            })
        # 对于2个及以上簇，计算轮廓系数
        elif len(np.unique(labels)) >= 2:
            # 仅计算轮廓系数
            sil_score = silhouette_score(data, labels)
            scores.append({
                'n_clusters': n,
                'silhouette': sil_score
            })

    if not scores:  # 如果没有有效的分数
        return 1

    # 找出轮廓系数最大的结果（排除1个簇的情况，除非没有其他选择）
    valid_scores = [score for score in scores if score['n_clusters'] > 1]

    if valid_scores:  # 如果有2个及以上簇的有效分数
        best_result = max(valid_scores, key=lambda x: x['silhouette'])
    else:  # 如果只有1个簇的分数
        best_result = scores[0]

    # print(f"最佳聚类数: {best_result['n_clusters']}")
    # print(f"轮廓系数: {best_result['silhouette']:.4f}")

    return best_result['n_clusters']


def cluster_image(img_path, model, label_file):
    """
    处理单个图像，返回处理后的图像和行列数

    参数:
    img_path -- 图像路径
    model -- 目标检测模型

    返回:
    (图像, 行数, 列数)
    """
    global all_iou_before_filtering, all_iou_after_completion
    img = cv2.imread(img_path)
    img01 = img.copy()
    img02 = img.copy()
    img03 = img.copy()
    height, width, _ = img.shape

    gt_boxes = parse_yolo_file(label_file, img_path)
    # 进行推理
    results = model.predict(img, max_det=950, conf=0.3, iou=0.8,line_width=1)
    detections = []

    detected_widths = []
    detected_heights = []
    detections_copy01 = []  # 存储过滤前的中心点
    detections_copy02 = []  # 存储过滤后的中心点
    filtered_detections = []

    # 初始化用于存储面积的列表
    box_areas = []
    color = (255, 0, 0)
    # 遍历每个检测框，输出中心点
    for result in results:
        boxes = result.boxes
        names = result.names
        num = len(boxes.cls.cpu().numpy().astype(int))
        print(num)
        if num >= 1:

            for i in range(num):
                xyxy = boxes.xyxy.cpu().numpy().astype(int)[i]
                xywh = boxes.xywh.cpu().numpy().astype(int)[i]
                cls = boxes.cls.cpu().numpy().astype(int)[i]
                conf = boxes.conf.cpu().numpy()[i]
                if cls == 0:
                    x1 = xyxy[0]
                    y1 = xyxy[1]
                    x2 = xyxy[2]
                    y2 = xyxy[3]
                    x_center = xywh[0]
                    y_center = xywh[1]
                    box_width = xywh[2]
                    box_height = xywh[3]
                    # 存储过滤前的中心点
                    detections_copy01.append((x1, y1, x2, y2))
                    # 计算当前框与所有真实框的最大IOU
                    max_iou = 0
                    for gt_box in gt_boxes:
                        iou = calculate_iou(xyxy, gt_box)
                        if iou > max_iou:
                            max_iou = iou
                            best_gt_box = gt_box

                    # 添加到全局统计
                    all_iou_before_filtering.append(max_iou)
                    # 计算当前框的面积
                    current_area = (x2 - x1) * (y2 - y1)

                    # 检查是否已经有重叠框
                    keep = True
                    for det in filtered_detections:
                        iou = calculate_iou(xyxy, det['bbox'])
                        if iou > 0.4 and det['confidence'] > conf:
                            keep = False
                            break

                    if keep:
                        filtered_detections.append({
                            'bbox': xyxy,
                            'confidence': conf,
                            'area': current_area,
                            'center': (x_center, y_center),
                            'width': box_width,
                            'height': box_height,
                            'class': cls  # 保存类别信息
                        })
                        box_areas.append(current_area)
                        detected_widths.append(box_width)  # 保存的宽高是去掉重叠框后的检测框宽高
                        detected_heights.append(box_height)

                    cv2.rectangle(img01, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(img01, (x_center, y_center), 3, color, -1)  # 半径为4，填充颜色
                    cv2.putText(img01, f" {conf:.2f}", (x1, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
            #         {names.get(cls)}
            cv2.imwrite("./result1.jpg", img01)

        # # 绘制过滤重叠框后的检测框
    for det in filtered_detections:
        x1, y1, x2, y2 = det['bbox']
        x_center, y_center = det['center']
        # cls = det['class']  # 从det中获取类别
        conf = det['confidence']  # 从det中获取置信度
        cv2.rectangle(img02, (x1, y1), (x2, y2), color, 2)
        cv2.circle(img02, (x_center, y_center), 3, color, -1)  # 半径为4，填充颜色
        cv2.putText(img02, f"{conf:.2f}", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
        # {names.get(cls)}
        # 最后保存一次图片（所有检测框绘制完成后）
    cv2.imwrite("./result2.jpg", img02)

    # 边缘错检过滤
    # edge_threshold = 0.002  # 图像边缘0.2%区域
    # img_width, img_height = img.shape[1], img.shape[0]
    #
    # # 执行边缘过滤并记录被移除的框
    # edge_filtered_detections = []
    # removed_edge_boxes = []
    #
    # # 先计算尺寸中值用于判断异常尺寸（使用当前所有检测框）
    current_widths = [det['width'] for det in filtered_detections]
    current_heights = [det['height'] for det in filtered_detections]
    median_width = np.median(current_widths) if current_widths else 0
    median_height = np.median(current_heights) if current_heights else 0
    #
    # # 如果中值为0，设置为1避免除零错误
    # if median_width == 0:
    #     median_width = 1
    # if median_height == 0:
    #     median_height = 1
    #
    # # print(f"用于边缘过滤的尺寸中值 - 宽度: {median_width:.1f}, 高度: {median_height:.1f}")
    #
    # for det in filtered_detections:
    #     # 判断是否在边缘
    #     is_edge = (det['bbox'][0] < edge_threshold * img_width or
    #                det['bbox'][2] > (1 - edge_threshold) * img_width or
    #                det['bbox'][1] < edge_threshold * img_height or  # 添加上下边缘判断
    #                det['bbox'][3] > (1 - edge_threshold) * img_height)
    #
    #     # 判断是否为异常尺寸
    #     is_abnormal_size = not ((0.9 * median_width <= det['width'] <= 1.1 * median_width) and
    #                             (0.9 * median_height <= det['height'] <= 1.1 * median_height))
    #
    #     # 新增条件：既是边缘框又是异常尺寸才过滤
    #     if is_edge and is_abnormal_size:
    #         removed_edge_boxes.append(det)  # 记录被移除的边缘检测框
    #         # print(f"移除边缘异常框: 位置({det['bbox'][0]}, {det['bbox'][1]})-({det['bbox'][2]}, {det['bbox'][3]}), "
    #         #       f"尺寸{det['width']}×{det['height']}, 宽/高中值倍数: {det['width'] / median_width:.2f}/{det['height'] / median_height:.2f}")
    #     else:
    #         edge_filtered_detections.append(det)  # 保留检测框
    #
    #     # 也可以添加只记录不删除的边缘正常尺寸框
    #     # if is_edge and not is_abnormal_size:
    #         # print(f"边缘但尺寸正常框: 位置({det['bbox'][0]}, {det['bbox'][1]})-({det['bbox'][2]}, {det['bbox'][3]}), "
    #         #       f"尺寸{det['width']}×{det['height']}, 宽/高中值倍数: {det['width'] / median_width:.2f}/{det['height'] / median_height:.2f}")
    #
    # # 更新检测框列表
    # original_count = len(filtered_detections)
    # filtered_detections = edge_filtered_detections
    # removed_count = original_count - len(filtered_detections)
    #
    # # print(
    # #     f"边缘过滤结果: 原始{original_count}个框, 移除{removed_count}个边缘异常框, 剩余{len(filtered_detections)}个框")
    #
    #
    # # 最终条件判断
    # if filtered_detections:
    #     # 提取所有检测框信息
    #     centers = [det['center'] for det in filtered_detections]
    #     x_centers = [c[0] for c in centers]  # [100, 200, 300, 100, 200]
    #     y_centers = [c[1] for c in centers]  # [150, 150, 150, 250, 250]
    #
    #     # 分别对x和y坐标进行排序
    #     x_sorted = sorted(x_centers)
    #     y_sorted = sorted(y_centers)
    #
    #     # 计算相邻点之间的距离
    #     x_distances = np.diff(x_sorted)
    #     y_distances = np.diff(y_sorted)
    #     # print(x_distances)
    #     # print(y_distances)
    #     # 第一步：过滤极端异常值（超过10倍中值的距离）
    #     # 先计算初始中值
    #     initial_median_x = np.median(x_distances) if len(x_distances) > 0 else 0
    #     initial_median_y = np.median(y_distances) if len(y_distances) > 0 else 0
    #
    #     # 过滤掉超过10倍中值的极端异常距离
    #     extreme_threshold = 10.0
    #     if initial_median_x == 0:
    #         initial_median_x = 1
    #         # print("警告：X方向过滤前中值为0，已调整为1")
    #     if initial_median_y == 0:
    #         initial_median_y = 1
    #         # print("警告：Y方向过滤前中值为0，已调整为1")
    #
    #     # 对x_distances进行过滤
    #     filtered_x_distances = []
    #     extreme_x_count = 0
    #     for dist in x_distances:
    #         if initial_median_x > 0 and dist >=extreme_threshold * initial_median_x:
    #             extreme_x_count += 1
    #             # print(
    #             #     f"过滤掉X方向极端异常距离: {dist} (是初始中值{initial_median_x}的{dist / initial_median_x:.1f}倍)")
    #         else:
    #             filtered_x_distances.append(dist)
    #
    #     # 对y_distances进行过滤
    #     filtered_y_distances = []
    #     extreme_y_count = 0
    #     for dist in y_distances:
    #         if initial_median_y > 0 and dist > extreme_threshold * initial_median_y:
    #             extreme_y_count += 1
    #             # print(
    #             #     f"过滤掉Y方向极端异常距离: {dist} (是初始中值{initial_median_y}的{dist / initial_median_y:.1f}倍)")
    #         else:
    #             filtered_y_distances.append(dist)
    #
    #     # 重新计算过滤后的中值
    #     median_x_distance = np.median(filtered_x_distances) if len(filtered_x_distances) > 0 else 0
    #     median_y_distance = np.median(filtered_y_distances) if len(filtered_y_distances) > 0 else 0
    #
    #     # 增加代码：如果中值为0，则改为1，避免除零错误和无效判断
    #     if median_x_distance == 0:
    #         median_x_distance = 1
    #         # print("警告：X方向过滤后中值为0，已调整为1")
    #     if median_y_distance == 0:
    #         median_y_distance = 1
    #         # print("警告：Y方向过滤后中值为0，已调整为1")
    #
    #     # print(
    #     #     f"初始X中值: {initial_median_x:.1f}, 过滤后X中值: {median_x_distance:.1f}, 过滤掉{extreme_x_count}个极端值")
    #     # print(
    #     #     f"初始Y中值: {initial_median_y:.1f}, 过滤后Y中值: {median_y_distance:.1f}, 过滤掉{extreme_y_count}个极端值")
    #
    #     # 第二步：使用过滤后的数据计算正常范围内的异常距离
    #     # 计算x方向异常距离的数量（超过2倍中值距离的视为异常）
    #     distance_threshold = 5  # 2倍中值距离作为阈值
    #
    #     abnormal_x_distance_count = 0
    #     for dist in filtered_x_distances:
    #         if median_x_distance > 0 and dist > distance_threshold * median_x_distance:
    #             abnormal_x_distance_count += 1
    #             # print(f"X方向异常距离: {dist} (是过滤后中值{median_x_distance}的{dist / median_x_distance:.1f}倍)")
    #
    #     # 计算y方向异常距离的数量
    #     abnormal_y_distance_count = 0
    #     for dist in filtered_y_distances:
    #         if median_y_distance > 0 and dist > distance_threshold * median_y_distance:
    #             abnormal_y_distance_count += 1
    #             # print(f"Y方向异常距离: {dist} (是过滤后中值{median_y_distance}的{dist / median_y_distance:.1f}倍)")
    #
    #     # 第三步：同时计算异常尺寸的个数
    #     current_widths = [det['width'] for det in filtered_detections]
    #     current_heights = [det['height'] for det in filtered_detections]
    #     median_width = np.median(current_widths) if current_widths else 0
    #     median_height = np.median(current_heights) if current_heights else 0
    #
    #     # 增加代码：如果宽度或高度中值为0，也调整为1
    #     if median_width == 0:
    #         median_width = 1
    #         # print("警告：宽度中值为0，已调整为1")
    #     if median_height == 0:
    #         median_height = 1
    #         # print("警告：高度中值为0，已调整为1")
    #
    #     # 计算异常尺寸的个数（超出0.9-1.1倍中值范围的框）
    #     abnormal_size_count = len([
    #         det for det in filtered_detections
    #         if not (0.9 * median_width <= det['width'] <= 1.1 * median_width) or
    #            not (0.9 * median_height <= det['height'] <= 1.1 * median_height)
    #     ])
    #
    #     # print(f"=== 网格规律性分析 ===")
    #     # print(f"X方向: 过滤后距离中值={median_x_distance:.1f}, 正常异常距离={abnormal_x_distance_count}")
    #     # print(f"Y方向: 过滤后距离中值={median_y_distance:.1f}, 正常异常距离={abnormal_y_distance_count}")
    #     # print(f"异常尺寸数量: {abnormal_size_count}")
    #     # print(f"过滤掉的极端X异常距离: {extreme_x_count}, 极端Y异常距离: {extreme_y_count}")
    #
    #     # 第四步：判断条件
    #     # 如果x和y方向都有异常距离且异常尺寸框超过10个，则进行异常尺寸过滤
    #     # 注意：这里的异常距离是过滤掉极端值后的"正常范围"内的异常
    #     if abnormal_x_distance_count > 0 and abnormal_y_distance_count > 0 and abnormal_size_count > 10:
    #         # print(f"X和Y方向都有异常距离且异常尺寸框超过10个，进行异常尺寸过滤")
    #         final_detections = []
    #         filtered_out_detections = []
            # final_detections = [
            #     det for det in filtered_detections
            #     if (0.9 * median_width <= det['width'] <= 1.1 * median_width) and
            #        (0.9 * median_height <= det['height'] <= 1.1 * median_height)
            # ]
    final_detections = []
    filtered_out_detections = []
    for det in filtered_detections:
        if (0.9 * median_width <= det['width'] <= 1.1 * median_width) and \
            (0.9 * median_height <= det['height'] <= 1.1 * median_height):
            final_detections.append(det)
        else:
            filtered_out_detections.append(det)

    # 输出被过滤掉的框的数量
    print(f"被过滤掉的框数量: {len(filtered_out_detections)}")
        # else:
            # print(
            #     f"条件不满足（X异常距离: {abnormal_x_distance_count}, Y异常距离: {abnormal_y_distance_count}, 异常尺寸: {abnormal_size_count}），直接使用当前结果")
            # final_detections = filtered_detections
    # else:
    #     final_detections = []

    # 分别计算过滤掉的框中心点坐标与真实框的误差（欧氏距离）
    if filtered_out_detections and gt_boxes:
        match_results = []
        # 遍历每个过滤掉的框
        for filtered_box in filtered_out_detections:
            best_iou = 0
            matched_gt_box = None

            # 遍历所有真实框寻找最佳匹配
            for gt_box in gt_boxes:
                # 计算IOU（需要确保calculate_iou函数支持两种格式）
                iou = calculate_iou(filtered_box['bbox'], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    matched_gt_box = gt_box

            # 如果找到了匹配的真实框（可以设置一个IOU阈值）
            if matched_gt_box and best_iou > 0.4:  # 可以调整阈值
                # 计算真实框的中心点和尺寸
                gt_x1, gt_y1, gt_x2, gt_y2 = matched_gt_box
                gt_width = gt_x2 - gt_x1
                gt_height = gt_y2 - gt_y1
                gt_center_x = (gt_x1 + gt_x2) / 2
                gt_center_y = (gt_y1 + gt_y2) / 2
                gt_center = (gt_center_x, gt_center_y)

                # 获取过滤框的中心点和尺寸
                filtered_center = filtered_box['center']
                filtered_width = filtered_box['width']
                filtered_height = filtered_box['height']

                # 计算尺寸误差（宽度和高度）
                width_error = filtered_width - gt_width  # 宽度误差（正表示过滤框太宽，负表示太窄）
                height_error = filtered_height - gt_height  # 高度误差（正表示过滤框太高，负表示太矮）

                # 计算欧氏距离误差（两个中心点之间的距离）
                x_diff = filtered_center[0] - gt_center_x
                y_diff = filtered_center[1] - gt_center_y
                euclidean_distance = np.sqrt(x_diff ** 2 + y_diff ** 2) if hasattr(np, 'sqrt') else (
                                                                                                                x_diff ** 2 + y_diff ** 2) ** 0.5

                # 存储匹配结果和误差
                match_result = {
                    'type': 'filtered',
                    'box': filtered_box,
                    'gt_box': matched_gt_box,
                    'gt_center': gt_center,
                    'gt_width': gt_width,
                    'gt_height': gt_height,
                    'iou': best_iou,
                    'x_diff': x_diff,  # 保留X方向差值，方便分析
                    'y_diff': y_diff,  # 保留Y方向差值，方便分析
                    'width_error': width_error,  # 新增宽度误差
                    'height_error': height_error,  # 新增高度误差
                    'euclidean_distance': euclidean_distance  # 欧氏距离
                }
                match_results.append(match_result)

                add_filtered_error(
                    x_error=x_diff,
                    y_error=y_diff,
                    euclidean_distance=euclidean_distance,
                    image_path=img_path,
                    iou=best_iou,
                    filtered_center=filtered_center,
                    gt_center=gt_center,
                    filtered_bbox=filtered_box['bbox'],
                    gt_bbox=matched_gt_box,
                    width_error=width_error,  # 需要添加到函数参数
                    height_error=height_error  # 需要添加到函数参数
                )

                # 输出匹配信息
                print(
                    f"[过滤框] 真实框的中心点以及宽和高为{gt_center}, {gt_width:.2f}, {gt_height:.2f}，过滤框的中心点以及宽和高为{filtered_center}, {filtered_width:.2f}, {filtered_height:.2f}")
                print(
                    f"[过滤框] X坐标误差: {x_diff:.2f}像素, Y坐标误差: {y_diff:.2f}像素, 欧氏距离: {euclidean_distance:.2f}像素")
                print(
                    f"[过滤框] 宽度误差: {width_error:.2f}像素, 高度误差: {height_error:.2f}像素")
        # 批量计算误差统计
        if match_results:
            # 获取所有欧氏距离
            euclidean_distances = [result['euclidean_distance'] for result in match_results]
            width_errors = [result['width_error'] for result in match_results]  # 新增宽度误差列表
            height_errors = [result['height_error'] for result in match_results]  # 新增高度误差列表

            # 计算欧氏距离的MAE（平均绝对误差）
            euclidean_mae = np.mean(euclidean_distances) if hasattr(np, 'mean') else sum(euclidean_distances) / len(
                euclidean_distances)

            # 计算欧氏距离的RMSE（均方根误差）
            euclidean_rmse = np.sqrt(np.mean(np.square(euclidean_distances))) if hasattr(np, 'sqrt') else (sum([d ** 2
                                                                                                                for d in
                                                                                                                euclidean_distances]) / len(
                euclidean_distances)) ** 0.5

            # 计算欧氏距离的中位数误差（对异常值更鲁棒）
            euclidean_median = np.median(euclidean_distances) if hasattr(np, 'median') else sorted(euclidean_distances)[
                len(euclidean_distances) // 2]

            # 计算欧氏距离的标准差
            if len(euclidean_distances) > 1:
                euclidean_std = np.std(euclidean_distances) if hasattr(np, 'std') else (sum([(d - euclidean_mae) ** 2
                                                                                             for d in
                                                                                             euclidean_distances]) / (
                                                                                                    len(euclidean_distances) - 1)) ** 0.5
            else:
                euclidean_std = 0

            # 计算最大值和最小值
            euclidean_max = max(euclidean_distances)
            euclidean_min = min(euclidean_distances)

            # 也可以保留X和Y方向的单独统计（可选）
            x_diffs = [result['x_diff'] for result in match_results]
            y_diffs = [result['y_diff'] for result in match_results]
            x_mae = np.mean(np.abs(x_diffs)) if hasattr(np, 'mean') else sum([abs(d) for d in x_diffs]) / len(x_diffs)
            y_mae = np.mean(np.abs(y_diffs)) if hasattr(np, 'mean') else sum([abs(d) for d in y_diffs]) / len(y_diffs)

            # 宽度误差统计
            width_rmse = np.sqrt(np.mean(np.square(width_errors))) if hasattr(np, 'sqrt') else (sum([w ** 2 for w in
                                                                                                     width_errors]) / len(
                width_errors)) ** 0.5
            width_mae = np.mean(np.abs(width_errors)) if hasattr(np, 'mean') else sum(
                [abs(w) for w in width_errors]) / len(width_errors)

            # 高度误差统计
            height_rmse = np.sqrt(np.mean(np.square(height_errors))) if hasattr(np, 'sqrt') else (sum([h ** 2 for h in
                                                                                                       height_errors]) / len(
                height_errors)) ** 0.5
            height_mae = np.mean(np.abs(height_errors)) if hasattr(np, 'mean') else sum(
                [abs(h) for h in height_errors]) / len(height_errors)

            print(f"\n[过滤框统计]")
            print(f"匹配到的过滤框数量: {len(match_results)}")
            print(f"欧氏距离统计:")
            print(f"  - MAE（平均绝对误差）: {euclidean_mae:.2f} 像素")
            print(f"  - RMSE（均方根误差）: {euclidean_rmse:.2f} 像素")
            print(f"  - 中位数误差: {euclidean_median:.2f} 像素")
            print(f"  - 标准差: {euclidean_std:.2f} 像素")
            print(f"  - 最小值: {euclidean_min:.2f} 像素")
            print(f"  - 最大值: {euclidean_max:.2f} 像素")
            print(f"  - 范围: [{euclidean_min:.2f}, {euclidean_max:.2f}] 像素")

            # 可选：输出X和Y方向的MAE
            print(f"\n坐标方向统计:")
            print(f"  - X方向MAE: {x_mae:.2f} 像素")
            print(f"  - Y方向MAE: {y_mae:.2f} 像素")

            print(f"\n尺寸误差统计:")
            print(f"宽度误差:")
            print(f"  - RMSE（均方根误差）: {width_rmse:.2f} 像素")
            print(f"  - MAE（平均绝对误差）: {width_mae:.2f} 像素")

            print(f"高度误差:")
            print(f"  - RMSE（均方根误差）: {height_rmse:.2f} 像素")
            print(f"  - MAE（平均绝对误差）: {height_mae:.2f} 像素")
    # ------------误差计算结束-------------
    # 直接使用IOU重叠过滤后的结果作为最终检测
    # final_detections = filtered_detections

    # # 绘制全部过滤（重叠、边缘、尺寸异常）后的检测框
    for det in final_detections:
        x1, y1, x2, y2 = det['bbox']
        x_center, y_center = det['center']
        # cls = det['class']  # 从det中获取类别
        # conf = det['confidence']  # 从det中获取置信度
        # 存储全部过滤（重叠、边缘、尺寸异常）后的中心点
        detections.append((x_center, y_center))
        detections_copy02.append((x1, y1, x2, y2))

        cv2.rectangle(img03, (x1, y1), (x2, y2), color, 2)
        # cv2.circle(img03, (x_center, y_center), 3, color, -1)  # 半径为4，填充颜色
        # cv2.putText(img, f"{names.get(cls)} {conf:.2f}", (x1, y1 - 7),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 最后保存一次图片（所有检测框绘制完成后）
    # cv2.imwrite("./result3.jpg", img03)

    # 将检测到的中心点转换为numpy数组
    detections_np = np.array(detections)

    if len(detections_np) < 2:
        return img, 1, 1, None, None, None, None  # 如果检测到的点少于2个，返回1行1列

    # 对x坐标进行凝聚层次聚类
    x_coords = detections_np[:, 0].reshape(-1, 1)
    optimal_x = find_optimal_clusters(x_coords)
    x_clustering = AgglomerativeClustering(n_clusters=optimal_x).fit(x_coords)
    x_labels = x_clustering.labels_

    # 计算x坐标簇中心点
    filtered_x_coords, num_columns, x_labels, x_centroids, kept_x_indices = calculate_cluster_centers(x_coords,
                                                                                                      x_labels)
    detections_np = detections_np[kept_x_indices]  # 同步更新

    # 对y坐标进行凝聚层次聚类（使用所有点）
    y_coords = detections_np[:, 1].reshape(-1, 1)

    if len(y_coords) < 2:
        return img, 1, num_columns, x_labels, x_centroids, None, None  # 如果y坐标点不足，返回1行

    optimal_y = find_optimal_clusters(y_coords)
    y_clustering = AgglomerativeClustering(n_clusters=optimal_y).fit(y_coords)
    y_labels = y_clustering.labels_

    # 计算y坐标簇中心点
    filtered_y_coords, num_rows, y_labels, y_centroids, kept_y_indices = calculate_cluster_centers(y_coords, y_labels)
    detections_np = detections_np[kept_y_indices]  # 最终数据

    # 添加树状图绘制
    # 在聚类部分之后添加树状图绘制
    # if len(detections_np) >= 2:
    #     x_coords = detections_np[:, 0].reshape(-1, 1)
    #     y_coords = detections_np[:, 1].reshape(-1, 1)
    #
    #     optimal_x = find_optimal_clusters(x_coords)
    #     optimal_y = find_optimal_clusters(y_coords)
    #
    #     # 绘制x坐标的树状图
    #     Z_x = plot_clustering_dendrogram(x_coords, 'X-coordinate', optimal_x)
    #     if Z_x is not None:
    #         plt.savefig('./x_dendrogram.jpg', dpi=300, bbox_inches='tight')
    #         plt.close()
    #
    #     # 绘制y坐标的树状图
    #     if len(y_coords) >= 2:
    #         Z_y = plot_clustering_dendrogram(y_coords, 'Y-coordinate', optimal_y)
    #         if Z_y is not None:
    #             plt.savefig('./y_dendrogram.jpg', dpi=300, bbox_inches='tight')
    #             plt.close()

    # ------------补全------------#
    # 统计每个类的数量
    x_counts = np.bincount(x_labels, minlength=num_columns)
    y_counts = np.bincount(y_labels, minlength=num_rows)

    # 补全漏检的穴孔
    missing_x_indices = np.where(x_counts < num_rows)[0]
    missing_y_indices = np.where(y_counts < num_columns)[0]

    # 计算已检测框的宽高
    # detected_widths = np.array(detected_widths)
    # detected_heights = np.array(detected_heights)
    # avg_width = np.mean(detected_widths)
    # avg_height = np.mean(detected_heights)
    # 计算已检测框的宽高中值
    # median_width = np.median(detected_widths)
    # median_height = np.median(detected_heights)

    #
    # 列拟合：用y预测x
    column_models = []
    for col in range(num_columns):
        mask = (x_labels == col)
        if np.sum(mask) >= 2:  # 至少2个点
            model = LinearRegression()
            model.fit(detections_np[mask, 1].reshape(-1, 1), detections_np[mask, 0])  # y -> x
            column_models.append(model)
        else:
            column_models.append(None)

    # 行拟合：用x预测y
    row_models = []
    for row in range(num_rows):
        mask = (y_labels == row)
        if np.sum(mask) >= 2:
            model = LinearRegression()
            model.fit(detections_np[mask, 0].reshape(-1, 1), detections_np[mask, 1])  # x -> y
            row_models.append(model)
        else:
            row_models.append(None)

    # 初始化 supplementary_boxes 列表
    supplementary_boxes = []
    # 定义颜色
    green_color = (0, 255, 255)

    # 绘制补全的检测框和中心点
    for missing_y in missing_y_indices:
        for missing_x in missing_x_indices:
            if (missing_y, missing_x) not in zip(y_labels, x_labels):
                # 获取对应行和列的直线方程
                row_model = row_models[missing_y]
                col_model = column_models[missing_x]

                # 只有当行和列都有有效的拟合直线时才进行交点计算
                if row_model is not None and col_model is not None:
                    # 解交点：两条直线方程的交点
                    # 行方程：y = row_coef * x + row_intercept
                    # 列方程：x = col_coef * y + col_intercept
                    row_coef = row_model.coef_[0]
                    row_intercept = row_model.intercept_
                    col_coef = col_model.coef_[0]
                    col_intercept = col_model.intercept_

                    # 求解交点坐标 (x, y)
                    # 从列方程：x = col_coef * y + col_intercept
                    # 代入行方程：y = row_coef * (col_coef * y + col_intercept) + row_intercept
                    # => y = row_coef * col_coef * y + row_coef * col_intercept + row_intercept
                    # => y * (1 - row_coef * col_coef) = row_coef * col_intercept + row_intercept
                    # => y = (row_coef * col_intercept + row_intercept) / (1 - row_coef * col_coef)

                    denominator = 1 - row_coef * col_coef
                    if abs(denominator) > 1e-6:  # 避免分母为零
                        y_centroid = (row_coef * col_intercept + row_intercept) / denominator
                        x_centroid = col_coef * y_centroid + col_intercept

                        x_centroid = int(round(x_centroid))
                        y_centroid = int(round(y_centroid))

                        # 计算检测框位置
                        top_left = (int(round(x_centroid - median_width // 2)), int(round(y_centroid - median_height // 2)))
                        bottom_right = (int(round(x_centroid + median_width // 2)), int(round(y_centroid + median_height // 2)))

                        supplementary_boxes.append((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))

                        # 使用绿色绘制矩形框和中心点
                        cv2.rectangle(img03, top_left, bottom_right, color=green_color, thickness=2)
                        # cv2.circle(img03, (x_centroid, y_centroid), 3, color=green_color, thickness=-1)

    # 若需要画出烟草，则打开
    for result in results:
        boxes = result.boxes
        num = len(boxes.cls)
        for i in range(num):
            xyxy = boxes.xyxy.cpu().numpy().astype(int)[i]
            cls = boxes.cls.cpu().numpy().astype(int)[i]
            if cls == 1:
                cv2.rectangle(img03, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 255, 50), 1)

    # cv2.imwrite("./result4.jpg", img03)

    # 2. 对补充框进行匹配计算
    if supplementary_boxes and gt_boxes:
        supplementary_match_results = []

        # 获取过滤框中已匹配的真实框集合
        filtered_matched_gt_boxes = set()
        for match_result in match_results:  # match_results是过滤框的匹配结果
            filtered_matched_gt_boxes.add(tuple(match_result['gt_box']))  # 转换为元组便于比较

        # 遍历每个补充框
        for supplementary_box in supplementary_boxes:  # supplementary_box是(x1, y1, x2, y2)格式的元组
            best_iou = 0
            matched_gt_box = None

            # 遍历所有真实框寻找最佳匹配
            for gt_box in gt_boxes:
                # 计算IOU
                iou = calculate_iou(supplementary_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    matched_gt_box = gt_box

            # 如果找到了匹配的真实框（可以设置一个IOU阈值）
            if matched_gt_box and best_iou > 0.4:  # 可以调整阈值
                # 检查这个真实框是否已经在过滤框中匹配过
                # if tuple(matched_gt_box) not in filtered_matched_gt_boxes:
                #     # 这是重复的真实框，不参与计算
                #     # 计算补充框的中心点和尺寸
                #     sup_x1, sup_y1, sup_x2, sup_y2 = supplementary_box
                #     sup_width = sup_x2 - sup_x1
                #     sup_height = sup_y2 - sup_y1
                #     sup_center_x = (sup_x1 + sup_x2) / 2
                #     sup_center_y = (sup_y1 + sup_y2) / 2
                #     sup_center = (sup_center_x, sup_center_y)
                #
                #     # 输出排除信息
                #     print(
                #         f"[补充框-排除] 真实框{matched_gt_box}已在过滤框中匹配，补充框{sup_center}, {sup_width:.2f}, {sup_height:.2f}被排除")
                #     continue

                # 计算真实框的中心点和尺寸
                gt_x1, gt_y1, gt_x2, gt_y2 = matched_gt_box
                gt_width = gt_x2 - gt_x1
                gt_height = gt_y2 - gt_y1
                gt_center_x = (gt_x1 + gt_x2) / 2
                gt_center_y = (gt_y1 + gt_y2) / 2
                gt_center = (gt_center_x, gt_center_y)

                # 计算补充框的中心点和尺寸
                sup_x1, sup_y1, sup_x2, sup_y2 = supplementary_box
                sup_width = sup_x2 - sup_x1
                sup_height = sup_y2 - sup_y1
                sup_center_x = (sup_x1 + sup_x2) / 2
                sup_center_y = (sup_y1 + sup_y2) / 2
                sup_center = (sup_center_x, sup_center_y)

                # 计算误差
                x_error = sup_center_x - gt_center_x
                y_error = sup_center_y - gt_center_y
                width_error = sup_width - gt_width  # 宽度误差（正表示预测框太宽，负表示太窄）
                height_error = sup_height - gt_height  # 高度误差（正表示预测框太高，负表示太矮）
                # 计算欧氏距离（综合误差）
                import math
                euclidean_distance = math.sqrt(x_error ** 2 + y_error ** 2)

                # 存储匹配结果和误差
                match_result = {
                    'type': 'supplementary',
                    'box': supplementary_box,
                    'box_center': sup_center,
                    'box_width': sup_width,
                    'box_height': sup_height,
                    'gt_box': matched_gt_box,
                    'gt_center': gt_center,
                    'gt_width': gt_width,
                    'gt_height': gt_height,
                    'iou': best_iou,
                    'x_error': x_error,
                    'y_error': y_error,
                    'width_error': width_error,  # 新增宽度误差
                    'height_error': height_error,  # 新增高度误差
                    'euclidean_distance': euclidean_distance
                }
                supplementary_match_results.append(match_result)

                add_supplementary_error(
                    x_error=x_error,
                    y_error=y_error,
                    euclidean_distance=euclidean_distance,
                    image_path=img_path,
                    iou=best_iou,
                    supplementary_center=sup_center,
                    gt_center=gt_center,
                    supplementary_bbox=supplementary_box,
                    gt_bbox=matched_gt_box,
                    width_error=width_error,  # 可能需要添加到函数参数
                    height_error=height_error  # 可能需要添加到函数参数
                )

                # 输出匹配信息
                print(
                    f"[补充框] 真实框的中心点以及宽和高为{gt_center}, {gt_width:.2f}, {gt_height:.2f}，补充框的中心点以及宽和高为{sup_center}, {sup_width:.2f}, {sup_height:.2f}")
                print(
                    f"[补充框]IOU: {best_iou:.3f},X坐标误差: {x_error:.2f}像素, Y坐标误差: {y_error:.2f}像素, 欧氏距离: {euclidean_distance:.2f}像素")
                print(
                    f"[补充框] 宽度误差: {width_error:.2f}像素, 高度误差: {height_error:.2f}像素")
            else:
                # 如果没有匹配到真实框或IOU太低，也排除
                sup_x1, sup_y1, sup_x2, sup_y2 = supplementary_box
                sup_width = sup_x2 - sup_x1
                sup_height = sup_y2 - sup_y1
                sup_center_x = (sup_x1 + sup_x2) / 2
                sup_center_y = (sup_y1 + sup_y2) / 2
                sup_center = (sup_center_x, sup_center_y)
                print(
                    f"[补充框-排除] 补充框{sup_center}, {sup_width:.2f}, {sup_height:.2f}未匹配到真实框或IOU过低(IOU={best_iou:.3f})")

        # 批量计算误差统计
        if supplementary_match_results:
            supplementary_x_errors = [result['x_error'] for result in supplementary_match_results]
            supplementary_y_errors = [result['y_error'] for result in supplementary_match_results]
            supplementary_width_errors = [result['width_error'] for result in supplementary_match_results]  # 新增宽度误差列表
            supplementary_height_errors = [result['height_error'] for result in supplementary_match_results]  # 新增高度误差列表
            supplementary_euclidean_distances = [result['euclidean_distance'] for result in supplementary_match_results]


            # X坐标误差统计
            x_squared_sum = sum([e ** 2 for e in supplementary_x_errors])
            x_rmse = math.sqrt(x_squared_sum / len(supplementary_x_errors))
            x_mean_error = sum([abs(e) for e in supplementary_x_errors]) / len(supplementary_x_errors)

            # Y坐标误差统计
            y_squared_sum = sum([e ** 2 for e in supplementary_y_errors])
            y_rmse = math.sqrt(y_squared_sum / len(supplementary_y_errors))
            y_mean_error = sum([abs(e) for e in supplementary_y_errors]) / len(supplementary_y_errors)

            # 宽度误差统计（绝对误差）
            width_squared_sum = sum([e ** 2 for e in supplementary_width_errors])
            width_rmse = math.sqrt(width_squared_sum / len(supplementary_width_errors))
            width_mean_error = sum([abs(e) for e in supplementary_width_errors]) / len(supplementary_width_errors)

            # 高度误差统计（绝对误差）
            height_squared_sum = sum([e ** 2 for e in supplementary_height_errors])
            height_rmse = math.sqrt(height_squared_sum / len(supplementary_height_errors))
            height_mean_error = sum([abs(e) for e in supplementary_height_errors]) / len(supplementary_height_errors)

            # 欧氏距离统计
            euclidean_squared_sum = sum([d ** 2 for d in supplementary_euclidean_distances])
            euclidean_rmse = math.sqrt(euclidean_squared_sum / len(supplementary_euclidean_distances))
            euclidean_mae = sum(supplementary_euclidean_distances) / len(supplementary_euclidean_distances)
            euclidean_max = max(supplementary_euclidean_distances)
            euclidean_min = min(supplementary_euclidean_distances)

            print(f"\n[补充框统计]")
            print(f"有效匹配到的补充框数量: {len(supplementary_match_results)}")
            print(f"排除的补充框数量: {len(supplementary_boxes) - len(supplementary_match_results)}")

            print(f"\nX坐标误差统计:")
            print(f"  - RMSE（均方根误差）: {x_rmse:.2f}像素")
            print(f"  - MAE（平均绝对误差）: {x_mean_error:.2f}像素")

            print(f"\nY坐标误差统计:")
            print(f"  - RMSE（均方根误差）: {y_rmse:.2f}像素")
            print(f"  - MAE（平均绝对误差）: {y_mean_error:.2f}像素")

            print(f"\n尺寸误差统计:")
            print(f"宽度误差:")
            print(f"  - RMSE（均方根误差）: {width_rmse:.2f}像素")
            print(f"  - MAE（平均绝对误差）: {width_mean_error:.2f}像素")

            print(f"高度误差:")
            print(f"  - RMSE（均方根误差）: {height_rmse:.2f}像素")
            print(f"  - MAE（平均绝对误差）: {height_mean_error:.2f}像素")

            print(f"\n欧氏距离统计（综合误差）:")
            print(f"  - RMSE（均方根误差）: {euclidean_rmse:.2f}像素")
            print(f"  - MAE（平均绝对误差）: {euclidean_mae:.2f}像素")
            print(f"  - 最大值: {euclidean_max:.2f}像素")
            print(f"  - 最小值: {euclidean_min:.2f}像素")
            print(f"  - 范围: [{euclidean_min:.2f}, {euclidean_max:.2f}]像素")

        else:
            print(f"[补充框统计] 没有有效的补充框匹配结果")

    # 计算贴合的检测框数量
    fit_rate_before, fit_count_before = fit_rate(gt_boxes, detections_copy01)
    print(f"before: fit rate = {fit_rate_before:.4f}")
    # print(len(detections_copy01))
    # print(fit_count_before)

    # 将 supplementary_boxes 中的框添加到 detections_copy02 中
    # detections_copy02就是过滤加再检测漏检目标完成后的最终所有检测框！！！
    detections_copy02.extend(supplementary_boxes)
    fit_rate_after, fit_count_after = fit_rate(gt_boxes, detections_copy02)
    # print(len(detections_copy02))
    print(f"after: fit rate = {fit_rate_after:.4f}")
    # print(fit_count_after)

    # 合并原始检测框和补全的检测框
    all_final_boxes = [det['bbox'] for det in final_detections] + supplementary_boxes

    # 统计每个检测框的IOU
    for box in all_final_boxes:
        max_iou = 0
        for gt_box in gt_boxes:
            iou = calculate_iou(box, gt_box)
            if iou > max_iou:
                max_iou = iou
        all_iou_after_completion.append(max_iou)
    # ------------补全------------#

    return img03, fit_rate_before, fit_rate_after, fit_count_before, fit_count_after, num_rows, num_columns
    # 下行在使用post_process.py或germination_rate01.py时开启
    # return detections_copy02, results


def process_directory(img_dir, model, third_model, result_dir, label_dir):
    """
    遍历目录，处理每个图像并保存结果。
    """
    global all_iou_before_filtering, all_iou_after_completion, all_iou_third_model

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    current_date = datetime.now().strftime("%Y%m%d")

    total_fit_rate_before = 0.0
    total_fit_rate_after = 0.0
    fit_count_before_all = 0
    fit_count_after_all = 0
    total_images = 0

    # 重置全局统计变量
    all_iou_before_filtering = []
    all_iou_after_completion = []
    all_iou_third_model = []

    # 初始化批量误差统计
    init_batch_errors()  # 新增：初始化误差统计

    for filename in os.listdir(img_dir):
        if filename.endswith((".png", ".jpg", ".jpeg", ".JPG")):
            img_path = os.path.join(img_dir, filename)
            label_file = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")

            # 使用第三个model处理图像
            process_third_model(img_path, third_model, label_file, all_iou_third_model)

            # 使用原始model处理图像
            processed_img, fit_rate_before, fit_rate_after, fit_count_before, fit_count_after, rows, cols = cluster_image(
                img_path,
                model,
                label_file)

            total_fit_rate_before += fit_rate_before
            total_fit_rate_after += fit_rate_after
            total_images += 1
            fit_count_before_all += fit_count_before
            fit_count_after_all += fit_count_after
            result_filename = f"{os.path.splitext(filename)[0]}_{rows}x{cols}_{current_date}.jpg"
            result_path = os.path.join(result_dir, result_filename)
            cv2.imwrite(result_path, processed_img)

    # 处理完所有图片后，生成IOU分布直方图
    generate_combined_iou_histogram(all_iou_before_filtering, all_iou_after_completion, all_iou_third_model,
                                    result_dir)
    # 计算批量误差统计
    calculate_batch_errors()

    total_fit_rate_before = total_fit_rate_before / total_images
    total_fit_rate_after = total_fit_rate_after / total_images

    print(f"before: total fit rate = {total_fit_rate_before:.4f}")
    print(f"after: total fit rate = {total_fit_rate_after:.4f}")
    print(f"before:  fit count = {fit_count_before_all}")
    print(f"after:  fit count = {fit_count_after_all}")
    print(f"Processed {total_images} images.")


if __name__ == '__main__':
    model = YOLO("E:/lrx/yolo11/runs/detect/train46/weights/best.pt")
    model2 = YOLO("E:/lrx/yolo11/runs/detect/train50/weights/best.pt")
    #
    # process_directory("datasets/tobacco/images/test"
    #                   , model, model2, "datasets/sort_result/size_test/train46","datasets/tobacco/labels/test")

    # process_directory("datasets/all_tobacco"
    #                   , model, model2, "datasets/sort_result/size_test/train46", "datasets/all_tobacco/txt")

    # 计算尺寸异常框前后误差时打开
    # process_directory("datasets/all_tobacco/0331"
    #                   , model, model2, "datasets/sort_result/size_test/MAE", "datasets/all_tobacco/0331txt")

    img_path = 'E:/lrx/yolo11/datasets/all_tobacco/train_03.jpg'
    label_file = 'E:/lrx/yolo11/datasets/tobacco/labels/test/train_03.txt'
    # label_file = 'E:/lrx/yolo11/datasets/all_tobacco/txt/train_03.txt'
    img, a, b, c, d, e, f = cluster_image(img_path, model, label_file)
    # print(a, b)
