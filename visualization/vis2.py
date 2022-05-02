import pandas as pd
import cv2
import codecs
import json
import matplotlib.pyplot as plt
import numpy as np

# %pylab inline

train_df = pd.read_csv('../data/data2/2train_rname.csv', header=None, usecols=[4,5])
train_df.columns = ['image_name', 'result']
train_df['result'] = train_df['result'].apply(json.loads)

train_df['image_name'] = '../data/data2/' + train_df['image_name']

COLOR_DICT = {
    'badge': (193,255,193),
    'glove': (255,246,143),
    'clothes': (250,128,114),
    'operatingbar': (0,255,127),
    'wrongclothes': (138,43,226),
    'person': (131,111,255)
}

def show_ann(idx):
    img = cv2.imread(train_df['image_name'].iloc[idx])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for item in train_df['result'].iloc[idx]['items']:
        c = COLOR_DICT[item['labels']['标签']]
        x1, y1, x2, y2 = np.array(item['meta']['geometry']).astype(int)
        ptLeftTop = (x1, y1)
        ptRightBottom = (x2, y2)
        thickness = 30
        lineType = 2

        cv2.rectangle(img, ptLeftTop, ptRightBottom, c, thickness, lineType)
        img = cv2.putText(img, item['labels']['标签'], (x1+30, y1-50), 2, 3, c, 5)

    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.show()
    _ = plt.xticks([]); plt.yticks([])

# show_ann(10)
#
# show_ann(11)
show_ann(312)
show_ann(313)
show_ann(314)
show_ann(315)



bbox_class, bbox = [], []
for items in train_df['result']:
    for item in items['items']:
        bbox_class.append(item['labels']['标签'])
        bbox.append(item['meta']['geometry'])

df = pd.DataFrame({'label':bbox_class, 'bbox': bbox})

print(df['label'].value_counts())

df['width'] = df['bbox'].apply(lambda x: x[2]-x[0])
df['height'] = df['bbox'].apply(lambda x: x[3]-x[1])

df['ratio'] = df['width'] / df['height']

print(df.groupby(['label'])['width'].mean())
print(df.groupby(['label'])['height'].mean())
print(df.groupby(['label'])['ratio'].mean())
