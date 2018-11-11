
import os
from datasets.coco import CocoDataset
from mrcnn import model as modellib, utils
import skimage
import numpy as np
import sys
import matplotlib.pyplot as plt

desired_size = 224

dataset_train = CocoDataset()
cat_nms = ['book', 'apple', 'keyboard']
cat_clrs = [[0., 0., 128.], [128., 0., 0.], [0., 128., 0.]]

dataset_train.load_coco('/Volumes/SercanHDD/coco', "train", year='2017',
                        auto_download=False, cat_nms=cat_nms)
dataset_train.prepare()

paths = ['./data/coco/train', './data/coco/train_labels']
mapping_file_path = './data/coco/train_mapping.csv'


def load_image(image_id, coco_dataset, sq_size=224):
    image = coco_dataset.load_image(image_id)
    mask = coco_dataset.load_mask_one_hot(image_id)
    
    delta_h = sq_size - image.shape[0]
    delta_w = sq_size - image.shape[1]
    ratio = image.shape[0] / image.shape[1]
    if abs(delta_w) > abs(delta_h):
        size = (int(sq_size), int(sq_size / ratio), image.shape[2])
    else:
        size = (int(sq_size * ratio), int(sq_size), image.shape[2])
    
    image = skimage.transform.resize(image, size, anti_aliasing=True)
    mask = skimage.transform.resize(mask, (size[0], size[1], mask.shape[2]), order=0)

    # random crop
    if image.shape[0] != sq_size:
        pad_needed = image.shape[0] - sq_size
        pad_l = np.random.randint(0, pad_needed)
        pad_r = pad_needed - pad_l

        image = image[pad_l:-pad_r, :, :]
        mask = mask[pad_l:-pad_r, :, :]
    elif image.shape[1] != sq_size:
        pad_needed = image.shape[1] - sq_size
        pad_l = np.random.randint(0, pad_needed)
        pad_r = pad_needed - pad_l

        image = image[:, pad_l:-pad_r, :]
        mask = mask[:,  pad_l:-pad_r, :]

    assert image.shape[0] == sq_size and image.shape[1] == sq_size
    assert mask.shape[0] == sq_size and mask.shape[1] == sq_size

    return image, mask

# make the paths
for path in paths:
    if not os.path.exists(path):
        os.makedirs(path)

# clean the contents of the file
open(mapping_file_path, 'w').close()

lent = len(dataset_train.image_info)
for i in range(lent):
    image, mask = load_image(i, dataset_train, 224)
    # image = dataset_train.load_image(i)
    # mask = dataset_train.load_mask_one_hot(i)
    
    # delta_h = desired_size - image.shape[0]
    # delta_w = desired_size - image.shape[1]
    # ratio = image.shape[0] / image.shape[1]
    # # 100x200 , 0.5, 
    # if abs(delta_w) > abs(delta_h):
    #     size = (int(desired_size), int(desired_size / ratio), image.shape[2])
    # else:
    #     size = (int(desired_size * ratio), int(desired_size), image.shape[2])
    
    # image = skimage.transform.resize(image, size, anti_aliasing=True)
    # mask = skimage.transform.resize(mask, (size[0], size[1], mask.shape[2]), order=0)

    # plt.imshow(image)
    # plt.show()

    # plt.imshow(mask[:,:,1:])
    # plt.show()

    skimage.io.imsave('./data/coco/train/{}.jpg'.format(i), image)
    np.save('./data/coco/train_labels/{}'.format(i), mask)
    with open(mapping_file_path, 'a') as the_file:
        the_file.write('{}, {}\n'.format(i, dataset_train.image_info[i]['id']))

    if i % 10 == 0:
        print('Progress:\t{}/{}'.format(i, lent))
