import torch
import cv2
from efficientnet_pytorch.model import EfficientNet
import glob
from torch import nn
import csv

from tqdm import trange
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

def load_weight( net, load_path, Parallel=True):
    state = torch.load(load_path)

    new_state = {}
    for key, value in state.items():
        new_state[key[7:]] = value
    net.load_state_dict(new_state)

    if Parallel:
        net = torch.nn.DataParallel(net).to('cuda')

    return net

net = EfficientNet.from_pretrained("efficientnet-b0", num_classes=2)

net = load_weight(net,"log/PAIP2020_MSI/efficientnet-b0_MSI_1_all_data/model_net_{0}.pth".format(712*1),
                      False).cuda()
net.eval()
with open("validation_MSI.csv", 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile,["WSI_ID", "MSI-H"])
    writer.writeheader()
    for i in range(1,32):
        print(i)
        file="data/wsi_5x_patchs/{0:02d}".format(i)


        num_positive=0
        num_negative=0
        for j in trange(300):

            with torch.no_grad():


                patch = cv2.imread(file+"/{0}.png".format(j))
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

                patch = torch.from_numpy(patch)
                patch = torch.unsqueeze(patch, dim=0).to("cuda")
                patch = patch.permute(0, 3, 1, 2)  # to NCHW

                patch = patch.to("cuda").float()

                logit,feature_vector = net(patch)
                prob = nn.functional.softmax(logit, dim=-1)
                pred = torch.argmax(prob, dim=-1)

                if pred==1:
                    num_positive+=1
                else:
                    num_negative+=1
        print("Negative",num_negative)
        print("Positive",num_positive)
        if num_negative>num_positive:
            print(file,i,": Negative")
            writer.writerow({"WSI_ID":"validation_data_{0:02d}".format(i),"MSI-H":0})
        else:
            print(file,i,": Positive")
            writer.writerow({"WSI_ID":"validation_data_{0:02d}".format(i),"MSI-H":1})


