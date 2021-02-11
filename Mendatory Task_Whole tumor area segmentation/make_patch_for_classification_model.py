import os
import glob
import cv2

import shutil

import tifffile
import openslide
import numpy as np
from tqdm import trange
import torch
from efficientnet_pytorch import EfficientNet
import torch.nn as nn

def load_weight(net, load_path):
    state = torch.load(load_path)

    new_state = {}
    for key, value in state.items():
        new_state[key[7:]] = value
    net.load_state_dict(new_state)

    # if Parallel:
    #     net = torch.nn.DataParallel(net).tonew_state('cuda')

    return net
####
with torch.no_grad():
    device="cuda"

    net=EfficientNet.from_pretrained("efficientnet-b2",num_classes=2)

    net = load_weight(net, "/media/data1/han/PAIP2020_code/log/PAIP2020/efficientnet-b2_find_tissue/model_net_28524.pth")

    net = torch.nn.DataParallel(net).to(device)
    net.eval()
    patch_size = 1024
    # net = self.load_weight(net, "/home/han/바탕화면/PAIP2020/densenet/densenet121-a639ec97.pth")
    aug_time = 1

    for idx in trange(1,48):
        if not os.path.exists("/media/data1/han/PAIP2020_DATA/training_set/benign/{0}".format(idx)):
            os.mkdir("/media/data1/han/PAIP2020_DATA/training_set/benign/{0}".format(idx))
        if not os.path.exists("/media/data1/han/PAIP2020_DATA/training_set/tumor/{0}".format(idx)):
            os.mkdir("/media/data1/han/PAIP2020_DATA/training_set/tumor/{0}".format(idx))


        wsi_tissue=tifffile.imread("/media/data1/han/PAIP2020_DATA/resize_wsi/tissue_label/{0:02}.tif".format(idx))

        wsi_path = '/media/data1/han/PAIP2020_DATA/wsi_folder/' + "training_data_{0:02d}.svs".format(idx)

        wsi = openslide.OpenSlide(wsi_path)
        wsi_w,wsi_h=wsi.dimensions
        for i in range(0, wsi_h - 1024, 1024):
            for j in range(0, wsi_w - 1024, 1024):
                patch = wsi.read_region((j, i), 0, (1024, 1024)).convert("RGB")
                patch = np.array(patch)

                if wsi_tissue[i:i+1024,j:j+1024].sum()<400000:
                    continue
                patch=cv2.cvtColor(patch,cv2.COLOR_BGR2RGB)
                imgs=torch.from_numpy(patch)


                imgs=torch.unsqueeze(imgs,dim=0)

                imgs = imgs.permute(0, 3, 1, 2)  # to NCHW

                # push data to GPUs and convert to float32
                imgs = imgs.to(device).float()


                logit, feature_vector = net(imgs)
                prob = nn.functional.softmax(logit, dim=-1)

                pred = torch.argmax(prob, dim=-1)


                patch=cv2.cvtColor(patch,cv2.COLOR_RGB2BGR)
                if pred==0:
                    cv2.imwrite("/media/data1/han/PAIP2020_DATA/training_set/benign/{0}/{1}_{2}.png".format(idx,i,j),patch)
                else:
                    cv2.imwrite("/media/data1/han/PAIP2020_DATA/training_set/tumor/{0}/{1}_{2}.png".format(idx,i,j),patch)


