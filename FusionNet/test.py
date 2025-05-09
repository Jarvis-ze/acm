# GPL License
# Copyright (C) 2023 , UESTC
# All Rights Reserved
#
# @Time    : 2023/10/1 20:29
# @Author  : Qi Cao
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in [2]])
import h5py
import torch
import time
import numpy as np
from Toolbox.model import FusionNet
from Toolbox.others import dir_name
from Toolbox.data import load_h5py
import scipy.io as sio

# ================== Pre-Define =================== #

ckpt = f'weights/1001.pth'
origin_path = r"/Data2/Datasets/PanCollection/test_data/test_wv3_OrigScale_multiExm1.h5"
device = "cuda"
save_mat = True

model = FusionNet()
weight = torch.load(ckpt)
model.load_state_dict(weight)


###################################################################
# ------------------- Main Test (Run second)----------------------------------
###################################################################


def test():
    print("Test...")

    includes_gt = "OrigScale" not in origin_path
    if includes_gt:
        ms, lms, pan, gt = load_h5py(origin_path, includes_gt)
    else:
        ms, lms, pan = load_h5py(origin_path, includes_gt)

    t1 = time.time()

    batch_size = ms.shape[0]
    sr = torch.zeros([batch_size, ms.shape[1], pan.shape[2], pan.shape[3]])

    for i in range(batch_size):
        res = model(lms[i].unsqueeze(0), pan[i].unsqueeze(0))
        out = res + lms[i].unsqueeze(0)
        sr[i,:,:,:] = out

    if "OrigScale" in origin_path:
        I_ms = ms.cpu().detach().numpy() * 2047.0
        I_lms = lms.cpu().detach().numpy() * 2047.0
        I_pan = pan.cpu().detach().numpy() * 2047.0
        I_sr = sr.cpu().detach().numpy() * 2047.0
    else:
        I_sr = sr.cpu().detach().numpy() * 2047.0

    t2 = time.time()
    print(f"Average time: {(t2 - t1)/batch_size: .4f}")

    if save_mat:
        if "OrigScale" in origin_path:
            sio.savemat('result/' + dir_name(origin_path, ckpt) + '.mat', {'sr': I_sr, 'ms': I_ms, 'lms': I_lms, 'pan': I_pan})
        else:
            gt = gt.numpy() * 2047.
            sio.savemat('result/' + dir_name(origin_path, ckpt) + '.mat', {'sr': I_sr, 'gt': gt})

###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################


if __name__ == "__main__":
    test()