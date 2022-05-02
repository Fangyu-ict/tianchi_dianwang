import json
import os
from glob import glob
from tqdm import tqdm
import cv2
from PIL import Image
import pandas as pd

def save(images, annotations, name):
    ann = {}
    ann['type'] = 'instances'
    ann['images'] = images
    ann['annotations'] = annotations

    category = [
            # {'id': 0, 'name': '_bkg', 'supercategory': 'china'},
            {'id': 1, 'name': 'badge'},
            {'id': 2, 'name': 'person'},
            {'id': 3, 'name': 'glove'},
            {'id': 4, 'name': 'wrongglove'},
            {'id': 5, 'name': 'operatingbar'},
            {'id': 6, 'name': 'powerchecker'},
    ]
    ann['categories'] = category
    json.dump(ann, open('../data/data1/annotations/dianwang_{}.json'.format(name), 'w'))


def test_dataset(im_dir, df):
    im_list = glob(im_dir + '/*.jpg')
    idx = 0
    image_id = -1
    images = []
    annotations = []
    #h, w, = 1696, 4096
    # for im_path in tqdm(im_list):
    for i, path in enumerate(df[0]):
        #image_id += 1
        if 'image_url' in path:
            continue
        #im = cv2.imread(im_path)
        im_path = '../data/data1/' + path
        im = Image.open(im_path)
        #h, w = im.[:2]
        w, h = im.size
        image_id += 1
        image = {'file_name': #os.path.split(im_path)[-1].split(".")[0] + "/" +
                 os.path.split(im_path)[-1], 'width': w, 'height': h, 'id': image_id}
        images.append(image)
        labels = [[10, 10, 20, 20]]
        for label in labels:
            bbox = [label[0], label[1], label[2] - label[0], label[3] - label[1]]
            seg = []
            ann = {'segmentation': [seg], 'area': bbox[2] * bbox[3], 'iscrowd': 0, 'image_id': image_id,
                   'bbox': bbox, 'category_id': 1, 'id': image_id, 'ignore': 0}
            idx += 1
            annotations.append(ann)
    save(images, annotations, 'testb_round1')


if __name__ == '__main__':
    test_dir = '../data/data1/' +  '1_test_imagesa'#'/tcdata/guangdong1_round2_testB_20191024'
    print("generate test json label file.")
    df = pd.read_csv("../data/data1/1_testa_user.csv", header=None)


    test_dataset(test_dir, df)
