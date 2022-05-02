import json
import os
import cv2

classes_num=4

parent_path = 'data/3_images/'
json_file = './data/annotations/train_annos_coco_0608.json'
with open(json_file) as annos:
    annotations = json.load(annos)

w_box = {}
h_box = {}
count = {}

for i in range(len(annotations['annotations'])):
    # if annotation['category_id'] == 0: # 1表示人这一类
    annotation = annotations['annotations'][i]
    bbox = annotation['bbox']
    if annotation['category_id'] not in w_box:
        w_box[annotation['category_id']] = []
        h_box[annotation['category_id']] = []
        count[annotation['category_id']] = 0
    w_box[annotation['category_id']].append(bbox[2])
    h_box[annotation['category_id']].append(bbox[3])
    count[annotation['category_id']] += 1

import matplotlib.pyplot as plt

# plot and save the w/h ratio
for i in range(classes_num):
    plt.title(str(i))
    plt.scatter(w_box[i], h_box[i])
    plt.savefig('w_h_graph_'+str(i)+'.png')
    plt.show()

    print(count[i])

# import numpy as np
# import pandas as pd
# # import pandas.Series
# from glob import glob
# from PIL import Image
# import cv2
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import seaborn as sns
# from bokeh.plotting import figure
# from bokeh.io import output_notebook, show, output_file
# from bokeh.models import ColumnDataSource, HoverTool, Panel
# from bokeh.models.widgets import Tabs
# # import albumentations as albu
#
# # Setup the paths to train and test images
# TRAIN_DIR = 'data/cizhuan/defect_Images/'
# TEST_DIR = 'data/cizhuan/testA_imgs/'
# TRAIN_CSV_PATH = 'data/cizhuan/Annotations/train_annos.json'
#
# # Glob the directories and get the lists of train and test images
# train_fns = glob(TRAIN_DIR + '*')
# test_fns = glob(TEST_DIR + '*')
# print('Number of train images is {}'.format(len(train_fns)))
# print('Number of test images is {}'.format(len(test_fns)))
#
#
# ######读取json文件转换成dataframe
# import json
# train = []
# with open(TRAIN_CSV_PATH, 'r', encoding='utf-8') as f:
#     data = json.load(f)
# train = pd.DataFrame(data)
#
# print(train.head(5))
#
# #####按照name进行合并，观察bbox
# all_train_images = pd.DataFrame([fns.split('/')[-1] for fns in train_fns])
# all_train_images.columns=['name']
# # merge image with json info
# all_train_images = all_train_images.merge(train, on='name', how='left')
# # replace nan values with zeros
# fil = pd.Series([0, 0, 0, 0])
# all_train_images['bbox'] = all_train_images.bbox.fillna('[0,0,0,0]')
# print(all_train_images.head(5))
#
# #####拆分bbox坐标方便后续观察
# # [xmin，ymin，xmax，ymax]
# bbox_items = all_train_images.bbox
#
# all_train_images['bbox_xmin'] = bbox_items.apply(lambda x: x[0])
# all_train_images['bbox_ymin'] = bbox_items.apply(lambda x: x[1])
# all_train_images['bbox_width'] = bbox_items.apply(lambda x: x[2]-x[0])
# all_train_images['bbox_height'] = bbox_items.apply(lambda x: x[3]-x[1])
# # print(all_train_images)
# print('{} images without bbox.'.format(len(all_train_images) - len(train)))
#
#
# def get_all_bboxes(df, name):
#     image_bboxes = df[df.name == name]
#
#     bboxes = []
#     for _, row in image_bboxes.iterrows():
#         bboxes.append((row.bbox_xmin, row.bbox_ymin, row.bbox_width, row.bbox_height))
#
#     return bboxes
#
#
# def plot_image_examples(df, rows=3, cols=3, title='Image examples'):
#     fig, axs = plt.subplots(rows, cols, figsize=(10, 10))
#     for row in range(rows):
#         for col in range(cols):
#             idx = np.random.randint(len(df), size=1)[0]
#             name = df.iloc[idx]["name"]
#             img = Image.open(TRAIN_DIR + str(name))
#             axs[row, col].imshow(img)
#
#             bboxes = get_all_bboxes(df, name)
#
#             for bbox in bboxes:
#                 rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r',
#                                          facecolor='none')
#                 axs[row, col].add_patch(rect)
#
#             axs[row, col].axis('off')
#
#     plt.suptitle(title)
#
#
# plot_image_examples(all_train_images)