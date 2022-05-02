import pandas as pd
import cv2
import codecs
import json
import matplotlib.pyplot as plt
import numpy as np

# %pylab inline

test_json = json.load(open('../submit/sub3/res_3.bbox.json', "r"))

test_df =  pd.read_csv("../data/data3/3_testa_user.csv", header=None)

COLOR_DICT = {
    'badge': (193,255,193),
    'offground': (255,246,143),
    'ground': (250,128,114),
    'safebelt': (0,255,127),
    'wrongclothes': (138,43,226),
    'person': (131,111,255)
}
ID2CATAGORY= {
    1: 'badge',
    2: "offground",
    3: 'ground',
    4: 'safebelt'
}


id2annos = {}
for anno in test_json:
    if anno['image_id'] not in id2annos:
        id2annos[anno['image_id']] = []
    id2annos[anno['image_id']].append(anno)

def show_ann(idx):
    img = cv2.imread('../data/data3/' + test_df[0][idx+1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for anno in id2annos[idx]:
        c = COLOR_DICT[ID2CATAGORY[anno['category_id']]]
        x1, y1, w, h = np.array(anno['bbox']).astype(int)
        ptLeftTop = (x1, y1)
        ptRightBottom = (x1+w, y1+h)
        thickness = 30
        lineType = 2

        cv2.rectangle(img, ptLeftTop, ptRightBottom, c, thickness, lineType)
        img = cv2.putText(img, ID2CATAGORY[anno['category_id']], (x1+30, y1-50), 2, 3, c, 5)

    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.show()
    _ = plt.xticks([]); plt.yticks([])


show_ann(421)




