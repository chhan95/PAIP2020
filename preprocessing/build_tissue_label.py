import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, draw
import openslide
import tifffile
import openslide


import argparse



for i in range(1, 32):
    print(i)

    file="data/validation_wsi/validation_data_{0:02d}.svs".format(i)
    filename = file.split("/")[-1].split(".")[0].split("_")[-1]
    wsi = openslide.OpenSlide(file)
    wsi_w, wsi_h = wsi.dimensions

    wsi_1x = wsi.get_thumbnail((wsi_w//40,wsi_h//40)).convert("RGB")
    wsi_1x=np.array(wsi_1x)


    h, w, _ = wsi_1x.shape

    wsi_1x = cv2.fastNlMeansDenoisingColored(wsi_1x, None, 10, 7, 21)
    wsi_1x = cv2.medianBlur(wsi_1x, 31)

    R, G, B = cv2.split(wsi_1x)
    _, R1 = cv2.threshold(R, 220, 1, cv2.THRESH_BINARY)
    _, G1 = cv2.threshold(G, 200, 1, cv2.THRESH_BINARY)
    _, B1 = cv2.threshold(B, 220, 1, cv2.THRESH_BINARY)
    background_label_img = R1 * B1 * G1

    forground_label = np.ones((h, w)) - background_label_img
    kernel = np.ones((10, 10), np.uint8)
    # dilation=cv2.dilate(forground_label,kernel,iterations=1)
    opening = cv2.morphologyEx(forground_label, cv2.MORPH_OPEN, kernel)

    opening=opening.astype(np.uint8)
    connectivity = 4
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opening, connectivity, cv2.CV_32S)
    tissue_set=set([])
    for idx in range(1,num_labels):
        if stats[idx][4]<10000:
            continue
        tissue_set.add(idx)

    removed_small_component=np.zeros(opening.shape,dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if labels[y,x] in tissue_set:
                removed_small_component[y,x]=1

    kernel=np.ones((10,10),np.uint8)

    removed_small_component=cv2.dilate(removed_small_component,kernel,iterations=1)

    plt.imsave("data/validation_wsi/tissue_img/{0}.png".format(i),removed_small_component)

    wsi = openslide.OpenSlide("data/validation_wsi/validation_data_{0:02d}.svs".format(i))

    out = cv2.resize(removed_small_component,wsi.dimensions)
    tifffile.imsave("data/validation_wsi/tissue/{0}.tif".format(i),out,compress=9)


