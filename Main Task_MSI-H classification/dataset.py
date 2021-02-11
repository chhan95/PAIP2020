import csv
import glob
import random
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
from torchvision import transforms
from torchvision.utils import make_grid
from imgaug import augmenters as iaa

import numpy as np


class DatasetSerial(data.Dataset):

    def __init__(self, pair_list,tile_size,num_tile,train_mode=True):
        self.pair_list=pair_list

        self.tile_size=tile_size
        self.num_tile=num_tile
        self.train_mode=train_mode

    def __getitem__(self, idx):

        pair_list = self.pair_list[idx]


        input_img = np.zeros((self.tile_size * self.num_tile, self.tile_size * self.num_tile, 3), dtype=np.uint8)

        tile_list = pair_list[0]

        if len(tile_list) > self.num_tile * self.num_tile:
            temp_tile_list = np.random.choice(tile_list, self.num_tile * self.num_tile, replace=False)
        else:
            temp_tile_list = np.random.choice(tile_list, self.num_tile * self.num_tile)

        if self.train_mode:

            seq = iaa.Sequential(
                [
                    iaa.CropToFixedSize(self.tile_size, self.tile_size),

                    iaa.PadToFixedSize(self.tile_size*2,self.tile_size*2, pad_mode='reflect'),

                    iaa.Affine(
                        # scale images to 80-120% of their size, individually per axis
                        scale={"x": (0.8, 1.2),
                               "y": (0.8, 1.2)},
                        # translate by -A to +A percent (per axis)
                        translate_percent={"x": (-0.01, 0.01),
                                           "y": (-0.01, 0.01)},
                        rotate=(-179, 179),  # rotate by -179 to +179 degrees
                        shear=(-5, 5),  # shear by -5 to +5 degrees
                        order=[0],  # use nearest neighbour
                        backend='cv2'  # opencv for fast processing
                    ),
                    # iaa.ElasticTransformation(alpha=self.tile_size, sigma=self.tile_size * 0.085, cval=255),

                    iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                    iaa.Flipud(0.5),  # vertically flip 50% of all images

                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)),  # gaussian blur with random sigma
                        iaa.MedianBlur(k=(3, 5)),  # median with random kernel sizes
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                    ]),
                    iaa.Sequential([
                        iaa.Add((-26, 26)),
                        iaa.AddToHueAndSaturation((-10, 10)),
                        iaa.LinearContrast((0.8, 1.2), per_channel=1.0),
                    ], random_order=True),
                    iaa.CropToFixedSize(self.tile_size , self.tile_size, position="center"),
                ]
            )
        else:
            seq = iaa.Sequential(
                [iaa.CropToFixedSize(self.tile_size, self.tile_size)]
            )
        temp_img_idx = 0

        for y in range(self.num_tile):
            for x in range(self.num_tile):
                temp_tile = cv2.imread(temp_tile_list[temp_img_idx])
                temp_tile = seq.augment_image(temp_tile)

                temp_img_idx += 1
                temp_tile = cv2.cvtColor(temp_tile, cv2.COLOR_BGR2RGB)
                input_img[self.tile_size * y: self.tile_size * y + self.tile_size,
                self.tile_size * x: self.tile_size * x + self.tile_size, :] = temp_tile

        img_label = pair_list[1]  # normal is 0
        return input_img, img_label

    def __len__(self):
        return len(self.pair_list)

def prepare_PAIP2020_PANDA(valid_fold):
    def load_data_info(big_pathname, parse_label=True, label_value=0):

        big_tile_list = glob.glob(big_pathname+"*.png")

        label_list = [label_value for file_path in big_tile_list]
        print(big_pathname)
        print(Counter(label_list))
        return list(zip([big_tile_list], label_list))

    root_path = '/media/data1/han/PAIP2020_DATA/PAIP2020_train/wsi_5x'

    train_set = []
    valid_set = []
    test_set = []

    import os
    for i in range(1,48):
        if os.path.exists('{0}/tumor/{1:02d}/'.format(root_path, i)):
            train_set += load_data_info('{0}/tumor/{1:02d}/'.format(root_path, i), parse_label=False, label_value=1)
    for i in range(1, 48):
        if os.path.exists('{0}/benign/{1:02d}/'.format(root_path, i)):
            train_set += load_data_info('{0}/benign/{1:02d}/'.format(root_path, i) , parse_label=False, label_value=0)

    root_path = '/media/data1/han/PAIP2020_DATA/PAIP2020_validation/wsi_5x_tiles_128'

    for i in range(1,32):
        if os.path.exists('{0}/tumor/{1:02d}/'.format(root_path, i)):
            valid_set += load_data_info('{0}/tumor/{1:02d}/'.format(root_path, i), parse_label=False, label_value=1)
    for i in range(1, 32):
        if os.path.exists('{0}/benign/{1:02d}/'.format(root_path, i)):
            valid_set += load_data_info('{0}/benign/{1:02d}/'.format(root_path, i) , parse_label=False, label_value=0)


    return train_set[:]*300, valid_set[:]*300


####
def visualize(ds, batch_size, nr_steps=3):
    data_idx = 0
    cmap = plt.get_cmap('jet')
    for i in range(0, nr_steps):
        if data_idx >= len(ds):
            data_idx = 0
        for j in range(1, batch_size + 1):
            sample = ds[data_idx + j]
            if len(sample) == 2:
                img = sample[0]
            else:
                img = sample[0]
                # TODO: case with multiple channels
                aux = np.squeeze(sample[-1])
                aux = cmap(aux)[..., :3]  # gray to RGB heatmap
                aux = (aux * 255).astype('uint8')
                img = np.concatenate([img, aux], axis=0)
                img = cv2.resize(img, (40, 80), interpolation=cv2.INTER_CUBIC)
            plt.subplot(1, batch_size, j)
            plt.title(str(sample[1]))
            plt.imshow(img)
        plt.show()
        data_idx += batch_size
