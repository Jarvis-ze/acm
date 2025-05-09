# GPL License
# Copyright (C) 2023 , UESTC
# All Rights Reserved
#
# @Time    : 2023/10/1 20:29
# @Author  : Qi Cao
import argparse
import time
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Toolbox.data import Dataset
from Toolbox.model import FusionNet
from Toolbox.wald_utils import wald_protocol_v1


# ================== Pre-Define =================== #
SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# cudnn.benchmark = True  ###自动寻找最优算法
cudnn.deterministic = True
# ============= HYPER PARAMS(Pre-Defined) ========= #
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
parser.add_argument("--epochs", type=int, default=10000, help="Number of epochs")
parser.add_argument("--ckpt", type=int, default=1000, help="Checkpoint interval")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--device", type=str, default='cuda', help="Device to use")
parser.add_argument("--satellite", type=str, default='wv3/', help="Satellite type")
parser.add_argument("--origin_path", type=str, default='D:\\Datasets\\wv3\\train.h5', help="Test picture path")
args = parser.parse_args()

lr = args.lr
epochs = args.epochs
batch_size = args.batch_size
device = torch.device(args.device)
satellite = args.satellite
origin_path = args.origin_path

if "wv2" in satellite:
    sensor = "WV2"
elif "wv3" in satellite:
    sensor = "WV3"

model = FusionNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))  # optimizer 1
betas = [8, 1]

###################################################################
# ------------------- Main Train (Run second)----------------------------------
###################################################################


def train(training_data_loader):
    for epoch in range(epochs):
        epoch += 1
        epoch_loss = []
        # ============Epoch Train=============== #
        model.train()
        for batch in training_data_loader:
            ms  = Variable(batch[0]).to(device) # [B, 8, 16, 16]
            lms = Variable(batch[1]).to(device) # [B, 8, 64, 64]
            pan = Variable(batch[2]).to(device) # [B, 1, 64, 64]
            gt  = Variable(batch[3]).to(device) # [B, 8, 64, 64]
            
            res = model(lms, pan) # [B, 8, 64, 64]
            out = res + lms
            
            optimizer.zero_grad()  # fixed
            loss = criterion(out, gt)
            loss.backward()
            optimizer.step()  # fixed

            epoch_loss.append(loss.item())

        print(f"Epoch [{epoch}]/[{epochs}]: {sum(epoch_loss)/len(epoch_loss): .8f}")

        if (epoch) % 10 == 0:
            torch.save(model.state_dict(), f"./weights/{epoch + 1}.pth")

###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################


if __name__ == "__main__":
    train_set = Dataset(origin_path)
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True)  # put training data to DataLoader for batches
    train(training_data_loader)