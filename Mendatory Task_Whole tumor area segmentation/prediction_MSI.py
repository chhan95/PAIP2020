import torch
from config import Config
import glob
import matplotlib.pyplot as plt
import os
from collections import Counter
import torch.utils.data as data
from imgaug import augmenters as iaa
import dataset

import numpy as np
import efficientnet_pytorch
from torch import nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
import openslide
from efficientnet_pytorch import EfficientNet
from tifffile import imsave, imread
import cv2
from tqdm import trange


class Prediction(Config):
    def __init__(self):
        super(Config, self).__init__()

        self.data_root_dir = '/media/han/hard_2/PAIP2020_validation/'


    def load_weight(self, net, load_path, Parallel=False):
        state = torch.load(load_path)

        new_state = {}
        for key, value in state.items():
            new_state[key[7:]] = value
        net.load_state_dict(new_state)

        if Parallel:
            net = torch.nn.DataParallel(net).tonew_state('cuda')
        return net



    def predict_patch(self, starting_point,window_size,wsi, tissue, net, pred_result, wsi_h, wsi_w):
        for i in range(starting_point, wsi_h - window_size, window_size):
            for j in range(starting_point, wsi_w - window_size, window_size):
                img = wsi.read_region((j, i), 0, (window_size, window_size)).convert("RGB")

                img = np.array(img)
                if tissue[i:i + window_size, j:j + window_size].sum() < window_size*window_size*0.8:
                    continue

                patch = torch.from_numpy(img).to("cuda").float()
                patch = torch.unsqueeze(patch, dim=0)
                patch = patch.permute(0, 3, 1, 2)  # to NCHW
                logit = net(patch)  # forward

                #  logit, aux_logit = net(imgs) # forward
                prob = F.softmax(logit, dim=-1)
                pred = torch.argmax(prob, dim=-1)

                if pred == 1:
                    pred_result[i:i + window_size, j:j + window_size] = 1
        return pred_result

    def predict(self):
        net = EfficientNet.from_pretrained('efficientnet-b2', num_classes=2)
        with torch.no_grad():
            net = self.load_weight(net, "/home/han/바탕화면/PAIP2020/log/model_net_28524.pth")
            net = net.to("cuda")
            print("Prediction Start!")
            for idx in range(1, 32):
                wsi_path = self.data_root_dir + "validation_data_{0:02d}.svs".format(idx)
                wsi = openslide.OpenSlide(wsi_path)

                wsi_w, wsi_h = wsi.dimensions

                tissue = imread("data/PAIP2020_validation/tissue/{0}.tif".format(idx))
                wsi_pred = np.zeros((wsi_h, wsi_w), dtype=np.uint8)

                window_size=1024
                for i in range(0, wsi_w, window_size):
                    img = wsi.read_region((i, wsi_h - window_size), 0, (window_size, window_size)).convert("RGB")

                    img = np.array(img)

                    if tissue[wsi_h - window_size:wsi_h, i:i + window_size].sum() < window_size*window_size*0.8:
                        continue

                    patch = torch.from_numpy(img).to("cuda").float()
                    patch = torch.unsqueeze(patch, dim=0)
                    patch = patch.permute(0, 3, 1, 2)  # to NCHW
                    logit = net(patch)  # forward

                    #  logit, aux_logit = net(imgs) # forward
                    prob = F.softmax(logit, dim=-1)
                    pred = torch.argmax(prob, dim=-1)

                    if pred == 1:
                        wsi_pred[wsi_h - window_size:wsi_h, i:i + window_size] += prob.cpu().numpy()
                for i in range(0, wsi_h, window_size):
                    img = wsi.read_region((wsi_w - window_size, i), 0, (window_size, window_size)).convert("RGB")
                    img = np.array(img)

                    if tissue[i:i + window_size, wsi_w - window_size:wsi_w].sum() < window_size*window_size*0.8:
                        continue

                    patch = torch.from_numpy(img).to("cuda").float()
                    patch = torch.unsqueeze(patch, dim=0)
                    patch = patch.permute(0, 3, 1, 2)  # to NCHW
                    logit = net(patch)  # forward

                    #  logit, aux_logit = net(imgs) # forward
                    prob = F.softmax(logit, dim=-1)
                    pred = torch.argmax(prob, dim=-1)

                    if pred == 1:
                        wsi_pred[i:i + window_size, wsi_w - window_size:wsi_w] += prob.cpu().numpy()
                for i in range(0,1024,256):
                    wsi_pred = self.predict_patch(i,window_size,wsi, tissue, net, wsi_pred, wsi_h, wsi_w)
                print("저장")
                small_wsi_pred = cv2.resize(wsi_pred, (0, 0), fx=0.01, fy=0.01, interpolation=cv2.INTER_NEAREST)
                plt.imsave('data/PAIP2020_validation/tumor/{0}.png'.format(idx), small_wsi_pred)
                wsi_pred = wsi_pred.astype('uint8')
                imsave('data/PAIP2020_validation/tumor/{0}_t.tif'.format(idx), wsi_pred, compress=9)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    predict = Prediction()
    predict.predict()
