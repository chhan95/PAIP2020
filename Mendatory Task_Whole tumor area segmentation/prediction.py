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
from tifffile import imsave
import cv2
from tqdm import trange

class Prediction(Config):
    def __init__(self):
        super(Config, self).__init__()

        self.data_root_dir = '/media/data1/han/PAIP2020_DATA/wsi_folder/'
        self.model1 = None
        self.model2=None

        self.test_loader = None

    def load_weight(self, net, load_path, Parallel=False):
        state = torch.load(load_path)

        new_state = {}
        for key, value in state.items():
            new_state[key[7:]] = value
        net.load_state_dict(new_state)

        if Parallel:
            net = torch.nn.DataParallel(net).tonew_state('cuda')
        return net


    def predict_patch(self,wsi,net, pred_result, wsi_h,wsi_w):
        for i in range(wsi_w - 1024, wsi_w, 1024):
            img = wsi.read_region((i, wsi_h - 1024), 0, (1024, 1024)).convert("RGB")

            img = np.array(img)

            R, G, B = cv2.split(img)
            _, R1 = cv2.threshold(R, 230, 1, cv2.THRESH_BINARY)
            _, G1 = cv2.threshold(G, 210, 1, cv2.THRESH_BINARY)
            _, B1 = cv2.threshold(B, 230, 1, cv2.THRESH_BINARY)
            background_img = R1 * B1 * G1

            if background_img.sum() > 1024 * 1024 * 0.3:
                continue

            patch = torch.from_numpy(img).to("cuda").float()
            patch = torch.unsqueeze(patch, dim=0)
            patch = patch.permute(0, 3, 1, 2)  # to NCHW
            logit = net(patch)  # forward

            #  logit, aux_logit = net(imgs) # forward
            prob = F.softmax(logit, dim=-1)
            pred = torch.argmax(prob, dim=-1)

            if pred == 1:
                pred_result[i:i + 1024, j:j + 1024] = 1
        for i in range(wsi_h - 1024, wsi_h, 1024):
            img = wsi.read_region((wsi_w - 1024, i), 0, (1024, 1024)).convert("RGB")

            img = np.array(img)

            R, G, B = cv2.split(img)
            _, R1 = cv2.threshold(R, 230, 1, cv2.THRESH_BINARY)
            _, G1 = cv2.threshold(G, 210, 1, cv2.THRESH_BINARY)
            _, B1 = cv2.threshold(B, 230, 1, cv2.THRESH_BINARY)
            background_img = R1 * B1 * G1

            if background_img.sum() > 1024 * 1024 * 0.3:
                continue

            patch = torch.from_numpy(img).to("cuda").float()
            patch = torch.unsqueeze(patch, dim=0)
            patch = patch.permute(0, 3, 1, 2)  # to NCHW
            logit = net(patch)  # forward

            #  logit, aux_logit = net(imgs) # forward
            prob = F.softmax(logit, dim=-1)
            pred = torch.argmax(prob, dim=-1)

            if pred == 1:
                pred_result[i:i + 1024, j:j + 1024] = 1
        for i in trange(0, wsi_h - 1024, 1024):

            for j in range(0, wsi_w - 1024, 1024):
                img = wsi.read_region((j, i), 0, (1024, 1024)).convert("RGB")

                img = np.array(img)

                R, G, B = cv2.split(img)
                _, R1 = cv2.threshold(R, 230, 1, cv2.THRESH_BINARY)
                _, G1 = cv2.threshold(G, 210, 1, cv2.THRESH_BINARY)
                _, B1 = cv2.threshold(B, 230, 1, cv2.THRESH_BINARY)
                background_img = R1 * B1 * G1

                if background_img.sum() > 1024 * 1024 * 0.3:
                    continue

                patch = torch.from_numpy(img).to("cuda").float()
                patch = torch.unsqueeze(patch, dim=0)
                patch = patch.permute(0, 3, 1, 2)  # to NCHW
                logit = net(patch)  # forward

                #  logit, aux_logit = net(imgs) # forward
                prob = F.softmax(logit, dim=-1)
                pred = torch.argmax(prob, dim=-1)


                if pred == 1:
                    pred_result[i:i + 1024, j:j + 1024] =1
        print("예측끝")
        return pred_result


    def predict(self):
        net = EfficientNet.from_pretrained('efficientnet-b2', num_classes=2)
        with torch.no_grad():
            net = self.load_weight(net, "log/model_net_28524.pth")
            net=net.to("cuda")
            print("Prediction Start!")
            for idx in range(41,48):
                wsi_path=self.data_root_dir+"training_data_{0}.svs".format(idx)
                wsi=openslide.OpenSlide(wsi_path)

                wsi_w,wsi_h=wsi.dimensions
                wsi_pred=np.zeros((wsi_h,wsi_w),dtype=np.uint8)
                wsi_pred=self.predict_patch(wsi,net,wsi_pred,wsi_h,wsi_w)


                print("저장")
                small_wsi_pred = cv2.resize(wsi_pred, (0, 0), fx=0.01, fy=0.01, interpolation=cv2.INTER_NEAREST)
                plt.imsave('/media/data1/han/PAIP2020_DATA/train_tumor_result/{0}.png'.format(idx), small_wsi_pred)
                wsi_pred = wsi_pred.astype('uint8')
                imsave('/media/data1/han/PAIP2020_DATA/train_tumor_result/{0}_t.tif'.format(idx), wsi_pred,compress=9)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    predict = Prediction()
    predict.predict()
