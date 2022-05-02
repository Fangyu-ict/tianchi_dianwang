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
test_json_raw = json.load(open("../data/data2/annotations/dianwang_testb_round1.json", "r"))
test_json = json.load(open('../submit/sub2/res_nms.bbox.json', "r"))
# 提交结果保存路径
results_json_path = "../submit/sub2/res_nms.json"

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
        # 将1 类取出来，即先判断是否是人（天上的加上地上的）
        person = [] # 人
        badge, clothes, wrongclothes = [], [], [] # 物体
        for anno in annos:
            label = anno["category_id"]
            score = anno["score"]
            bbox = anno["bbox"]
            # 框的格式化为点的格式x_min,y_min,x_max,y_max
            if label == 1:
                badge.append(np.array(
                    [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], score]))
            elif label == 2:
                person.append(np.array(
                    [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], score]))
            elif label == 3:
                clothes.append(np.array(
                    [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], score]))
            elif label == 4:
                wrongclothes.append(np.array(
                    [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], score]))
            else:
                break

        # print(offground)
        # 判断是否为人
        if len(person) != 0:
            for p in person:
                flagpassager = True
                # 判断是否有勋章
                if len(badge) != 0:
                    for bad in badge:
                        my_iou = iou(p[0:4], bad[0:4], 0.8, note = 'badge')
                        # print(my_iou)
                        if my_iou:
                            flagpassager = False
                            result = {}
                            result["image_id"] = i
                            result["category_id"] = 1
                            result["bbox"] = [p[0], p[1], p[2], p[3]]
                            result["score"] = float(p[4])
                            results.append(result)

                wrongclothesflag = False
                # 判断不合规衣服
                if len(wrongclothes) != 0:
                    for wro in wrongclothes:
                        my_iou = iou(p[0:4], wro[0:4], 0.8, note='wrongclothes')
                        # print(my_iou)
                        if my_iou:
                            flagpassager = False
                            wrongclothesflag = True
                            result = {}
                            result["image_id"] = i
                            result["category_id"] = 3
                            result["bbox"] = [p[0], p[1], p[2], p[3]]
                            result["score"] = float(p[4])
                            results.append(result)

                # 判断衣服1
                if len(clothes) != 0 and wrongclothesflag == False:
                    for clo in clothes:
                        my_iou = iou(p[0:4], clo[0:4], 0.8, note = 'clothes')
                        # print(my_iou)
                        if my_iou:
                            flagpassager = False
                            result = {}
                            result["image_id"] = i
                            result["category_id"] = 2
                            result["bbox"] = [p[0], p[1], p[2], p[3]]
                            result["score"] = float(p[4])
                            results.append(result)

                if flagpassager:
                    print('passager!')
                    result = {}
                    result["image_id"] = i
                    result["category_id"] = 3
                    result["bbox"] = [p[0], p[1], p[2], p[3]]
                    result["score"] = float(p[4])
                    results.append(result)

    print(len(results))
    json.dump(results, open(results_json_path, 'w'), indent=4)