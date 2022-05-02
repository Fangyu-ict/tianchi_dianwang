#导入需要用到的模块
import pandas as pd
import numpy as np
import random
import json
import shutil
import os
import matplotlib.pyplot as plt
import cv2
import math
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageChops

# 训练集对应类别标注
train_lable = {
    "badge": 1,
    "offground": 2,
    "ground": 3,
    "safebelt": 4
}
# 测试集图片名字、顺序
test_json_raw = json.load(open("../data/data3/annotations/dianwang_testb_round1.json", "r"))
test_json = json.load(open('../submit/sub3/res_multi3.bbox.json', "r"))
# 提交结果保存路径
results_json_path = "../submit/sub3/res_multi3.json"

# 求勋章的iou
def iou(person_frame, thing_frame, p=0.5, note = 'None'):
    x_min = max(person_frame[0], thing_frame[0])
    y_min = max(person_frame[1], thing_frame[1])
    x_max = min(person_frame[2], thing_frame[2])
    y_max = min(person_frame[3], thing_frame[3])
    # print(x_max-x_min,y_max-y_min)
    # 负值直接为0
    if (x_max - x_min) <= 0 or (y_max - y_min) <= 0:
        return False

    intersection = (x_max - x_min) * (y_max - y_min)
    thing = (thing_frame[2] - thing_frame[0]) * (thing_frame[3] - thing_frame[1])
    # per = (person_frame[2] - person_frame[0]) * (person_frame[3] - person_frame[1])
    # iou = intersection / (thing+per-intersection)
    iou = intersection / thing
    print(note + ': ' + str(iou))
    if iou >= p:
        return True
    else:
        return False

def nms(dets, thresh=0.5):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep



if '__main__' ==__name__:

    id2annos = {}
    for anno in test_json:
        if anno['image_id'] not in id2annos:
            id2annos[anno['image_id']] = []
        id2annos[anno['image_id']].append(anno)

    imgs = test_json_raw['images']
    results = []  # 提交结果

    for i,img in enumerate(imgs):
        print('id: '+  str(i))
        if i not in id2annos:
            continue
        annos = id2annos[i]

        # print(id_s)
        # 将2、3两类取出来，即先判断是否是人（天上的加上地上的）
        offground, ground = [], []  # 人
        badge, safebelt = [], []  # 物体
        for anno in annos:
            label = anno["category_id"]
            score = anno["score"]
            bbox = anno["bbox"]
            # 框的格式化为点的格式x_min,y_min,x_max,y_max
            if label == 2:
                offground.append(np.array(
                    [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], score, label]))
            elif label == 3:
                ground.append(np.array(
                    [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], score, label]))
            elif label == 1:
                badge.append(np.array(
                    [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], score, label]))
            elif label == 4:
                safebelt.append(np.array(
                    [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], score, label]))
            else:
                break

        # 对地上和地下做nms
        if len(offground) > 0 and len(ground) > 0:
            offground = np.array(offground)
            offground[:,4] += 1
            ground = np.array(ground)
            groundset = np.concatenate((offground,ground))
            order_nms = nms(groundset)
            groundset = groundset[order_nms]
            offground, ground = [], []  # 人
            for gr in groundset:
                if gr[-1] == 2:
                    offground.append([gr[0], gr[1], gr[2], gr[3], gr[4]-1, gr[5]])
                if gr[-1] == 3:
                    ground.append([gr[0], gr[1], gr[2], gr[3], gr[4], gr[5]])

        # print(offground)
        # 判断是否为天上的人
        if len(offground) != 0:
            for off in offground:
                offgroundperson = True  # 表示为离地的人,也就是第三类,不是作业人员也不是监督人员
                # 判断是否有勋章
                if len(badge) != 0:
                    for bad in badge:
                        my_iou = iou(off[0:4], bad[0:4], 0.8, note = 'badge')
                        # print(my_iou)
                        if my_iou:
                            offgroundperson = False
                            result = {}
                            result["image_id"] = i
                            result["category_id"] = 1
                            result["bbox"] = [off[0], off[1], off[2], off[3]]
                            result["score"] = float(off[4])
                            results.append(result)

                # 判断是否有穿安全带
                if len(safebelt) != 0:
                    for safe in safebelt:
                        my_iou = iou(off[0:4], safe[0:4], 0.8, note = 'safebelt')
                        # print(my_iou)
                        if my_iou:
                            result = {}
                            result["image_id"] = i
                            result["category_id"] = 2
                            result["bbox"] = [off[0], off[1], off[2], off[3]]
                            result["score"] = float(off[4])
                            results.append(result)

                if offgroundperson:
                    result = {}
                    result["image_id"] = i
                    result["category_id"] = 3
                    result["bbox"] = [off[0], off[1], off[2], off[3]]
                    result["score"] = float(off[4])
                    results.append(result)

        # 判断是否为地上的人
        # 路人是不提交结果的
        if len(ground) != 0:
            for gro in ground:

                # 判断是否有勋章
                if len(badge) != 0:
                    for bad in badge:
                        my_iou = iou(gro[0:4], bad[0:4], 0.8, note = 'badge')
                        # print(my_iou)
                        if my_iou:
                            result = {}
                            result["image_id"] = i
                            result["category_id"] = 1
                            result["bbox"] = [gro[0], gro[1], gro[2], gro[3]]
                            result["score"] = float(gro[4])
                            results.append(result)
                # 判断是否有穿安全带
                if len(safebelt) != 0:
                    for safe in safebelt:
                        my_iou = iou(gro[0:4], safe[0:4], 0.8, note = 'safebelt')
                        # print(my_iou)
                        if my_iou:
                            result = {}
                            result["image_id"] = i
                            result["category_id"] = 2
                            result["bbox"] = [gro[0], gro[1], gro[2], gro[3]]
                            result["score"] = float(gro[4])
                            results.append(result)

    print(len(results))
    json.dump(results, open(results_json_path, 'w'), indent=4)