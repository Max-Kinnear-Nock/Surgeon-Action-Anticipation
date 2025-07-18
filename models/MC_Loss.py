import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch

from models.my_pooling import my_MaxPool2d

criterion = nn.BCEWithLogitsLoss()


def Mask(nb_batch, channels, classes_num):

    foo = [1] * int(channels/2) + [0] *  int(channels/2)
    bar = []
    for i in range(classes_num):
        random.shuffle(foo)
        bar += foo
    bar = [bar for i in range(nb_batch)]
    bar = np.array(bar).astype("float32")
    bar = bar.reshape(nb_batch,classes_num*channels,1,1)
    bar = torch.from_numpy(bar)
    bar = bar.cuda()
    bar = Variable(bar)
    return bar


def supervisor(x,targets,height,cnum):
        # classes=6, cnum=340, channels=2040

        # L_div
        branch = x  # x: 10, 2040, 7, 7
        branch = branch.reshape(branch.size(0),branch.size(1), branch.size(2) * branch.size(3))  # 10, 2040, 49
        branch = F.softmax(branch,2)  # Softmax
        branch = branch.reshape(branch.size(0),branch.size(1), x.size(2), x.size(2))  # 10, 2040, 7, 7
        branch = my_MaxPool2d(kernel_size=(1, cnum), stride=(1, cnum))(branch)  # (10, 6, 7, 7) CCMP
        branch = branch.reshape(branch.size(0),branch.size(1), branch.size(2) * branch.size(3))  # 10, 6, 49
        loss_2 = 1.0 - 1.0*torch.mean(torch.sum(branch, 2))/cnum  #



        # L_dis
        mask = Mask(x.size(0), cnum, targets.shape[1])
        x, mask = Variable(x.cuda()), Variable(mask.cuda())
        branch_1 = x * mask  # CWA (10,2040,7,7)

        branch_1 = my_MaxPool2d(kernel_size=(1,cnum), stride=(1,cnum))(branch_1) # (10, 6, 7, 7) CCMP
        branch_1 = nn.AvgPool2d(kernel_size=(height, height))(branch_1)  # GAP  10,6,1,1
        branch_1 = branch_1.view(branch_1.size(0), -1)  # 10,6

        targets = targets.data.float()
        loss_1 = criterion(branch_1, targets)  # Softmax
        
        return [loss_1, loss_2]     