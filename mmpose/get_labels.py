import cv2
import numpy as np
from mmengine.structures import InstanceData
from mmpose.apis.inferencers import base_mmpose_inferencer

# **HSV 颜色阈值**
red_lower1 = np.array([0, 100, 100])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([160, 100, 100])
red_upper2 = np.array([180, 255, 255])

green_lower = np.array([35, 100, 100])
green_upper = np.array([85, 255, 255])


def get_color_mask(image, lower_range, upper_range):
    """获取特定颜色的掩码"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, lower_range, upper_range)

def get_color_ratio(mask):
    """计算颜色占比"""
    return np.sum(mask) / mask.size

def get_labels(frame, bboxes):
    """
    **为每个 `bbox` 分配颜色标签，确保仅有一个 person1 和一个 person2，并保持 bbox 原始顺序**
    """
    color_ratios = []

    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        person_roi = frame[y1:y2, x1:x2]

        # 计算红色 & 绿色的占比
        red_mask1 = get_color_mask(person_roi, red_lower1, red_upper1)
        red_mask2 = get_color_mask(person_roi, red_lower2, red_upper2)
        red_mask = red_mask1 + red_mask2
        green_mask = get_color_mask(person_roi, green_lower, green_upper)

        red_ratio = get_color_ratio(red_mask)
        green_ratio = get_color_ratio(green_mask)

        color_ratios.append((red_ratio, green_ratio, bbox))

    results_dict = {}  # 用于存储 bbox 与标签对应关系

    # **只有一个人，直接分配**
    if len(color_ratios) == 1:
        red_ratio, green_ratio, bbox = color_ratios[0]
        if red_ratio > green_ratio:
            results_dict[tuple(bbox)] = ('person1', red_ratio)
        else:
            results_dict[tuple(bbox)] = ('person2', green_ratio)

    # **有 2 个人，确保只有一个 person1，一个 person2**
    elif len(color_ratios) == 2:
        (r1, g1, bbox1), (r2, g2, bbox2) = color_ratios

        # **如果一个人红，一个人绿，直接分配**
        if r1 > g1 and g2 > r2:
            results_dict[tuple(bbox1)] = ('person1', r1)
            results_dict[tuple(bbox2)] = ('person2', g2)
        elif g1 > r1 and r2 > g2:
            results_dict[tuple(bbox1)] = ('person2', g1)
            results_dict[tuple(bbox2)] = ('person1', r2)
        else:
            # **如果两个人都是红色 or 都是绿色，按红绿比例差值决定 person1**
            if r1 - g1 > r2 - g2:
                results_dict[tuple(bbox1)] = ('person1', r1)
                results_dict[tuple(bbox2)] = ('person2', g2)
            else:
                results_dict[tuple(bbox1)] = ('person2', g1)
                results_dict[tuple(bbox2)] = ('person1', r2)

    # **确保按照原始 bbox 顺序返回 labels**
    results = [results_dict[tuple(bbox)] for bbox in bboxes]

    return results

def merge(data_sample: InstanceData):
    """**在 MMPose `InstanceData` 中注入颜色标签**"""
    instances = data_sample

    if 'bboxes' in instances:
        bboxes = instances.bboxes

        frame_to_use = base_mmpose_inferencer.original_frame
        labels = get_labels(frame_to_use, bboxes)

        instances.labels = labels

    return instances
