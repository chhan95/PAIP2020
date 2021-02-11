import os
import glob
import cv2

import shutil

import tifffile
import openslide
import numpy as np
from tqdm import trange
for idx in trange(1,48):
    if not os.path.exists("data/mendatory_task/benign/{0}".format(idx)):
        os.mkdir("data/mendatory_task/benign/{0}".format(idx))
    if not os.path.exists("data/mendatory_task/tumor/{0}".format(idx)):
        os.mkdir("data/mendatory_task/tumor/{0}".format(idx))
    if not os.path.exists("data/mendatory_task/ambiguous/{0}".format(idx)):
        os.mkdir("data/mendatory_task/ambiguous/{0}".format(idx))

    wsi_tissue=tifffile.imread("data/resize_wsi/tissue_label/{0:02}.tif".format(idx))
    wsi_label=tifffile.imread("data/mask_img_l2/training_data_{0:02}_l2_annotation_tumor.tif".format(idx))
    wsi_path = '/media/PAIP2020/wsi_folder/' + "training_data_{0:02d}.svs".format(idx)

    wsi = openslide.OpenSlide(wsi_path)
    wsi_w,wsi_h=wsi.dimensions



    window_size=1024
    for i in range(0, wsi_h - window_size, window_size):
        for j in range(0, wsi_w - window_size, window_size):
            patch = wsi.read_region((j, i), 0, (window_size, window_size)).convert("RGB")
            patch = np.array(patch)

            if wsi_tissue[i:i+window_size,j:j+window_size].sum()<400000:
                continue
            patch=cv2.cvtColor(patch,cv2.COLOR_RGB2BGR)
            if wsi_label[i:i+window_size,j:j+window_size].sum()<window_size*window_size*0.2:
                cv2.imwrite("/media/PAIP2020/training_set/benign/{0}/{1}_{2}.png".format(idx,i,j),patch)
            elif wsi_label[i:i+window_size,j:j+window_size].sum()>window_size*window_size*0.8:
                cv2.imwrite("/media/PAIP2020/training_set/tumor/{0}/{1}_{2}.png".format(idx,i,j),patch)
            else:
                cv2.imwrite("/media/PAIP2020/training_set/ambiguous/{0}/{1}_{2}.png".format(idx,i,j),patch)


