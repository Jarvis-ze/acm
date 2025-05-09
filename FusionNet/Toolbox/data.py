import torch
import torch.utils.data as data
import scipy.io as sio
import numpy as np
import h5py


class Dataset(data.Dataset):
    def __init__(self, file_path):
        super(Dataset, self).__init__()

        dataset = h5py.File(file_path, 'r')

        ms = np.array(dataset['ms'], dtype=np.float32) / 2047.0
        lms = np.array(dataset['lms'], dtype=np.float32) / 2047.0
        pan = np.array(dataset['pan'], dtype=np.float32) / 2047.0
        gt = np.array(dataset['gt'], dtype=np.float32) / 2047.0

        self.ms = torch.from_numpy(ms).float()
        self.lms = torch.from_numpy(lms).float()
        self.pan = torch.from_numpy(pan).float()
        self.gt = torch.from_numpy(gt).float()

    def __getitem__(self, item):
        return self.ms[item], self.lms[item], self.pan[item], self.gt[item]

    def __len__(self):
        return self.ms.shape[0]

def load_h5py(file_path, includes_gt):
    data = h5py.File(file_path)
    ms = data["ms"][...]  # W H C N
    ms = np.array(ms, dtype=np.float32) / 2047.
    ms = torch.from_numpy(ms)
    lms = data["lms"][...]  # W H C N
    lms = np.array(lms, dtype=np.float32) / 2047.
    lms = torch.from_numpy(lms)
    pan = data["pan"][...]  # W H C N
    pan = np.array(pan, dtype=np.float32) / 2047.
    pan = torch.from_numpy(pan)
    
    if includes_gt:
        gt = data["gt"][...]  # W H C N
        gt = np.array(gt, dtype=np.float32) / 2047.
        gt = torch.from_numpy(gt)
        return ms, lms, pan, gt
    else:
        return ms, lms, pan