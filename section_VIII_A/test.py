from re import S
import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import os
import argparse
import cupy as cp
import cusignal
from tqdm import tqdm
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from dl import *
from network import *


dataset_path = ''
train_dataloader = torch.utils.data.DataLoader(dataset=reverb_dataset(dataset_path),\
                                              batch_size=32,\
                                              shuffle=True,\
                                              num_workers=0,
                                              drop_last=False)

model = ImageTransformNet_Spectrogram().to("cuda:0")
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

model.train()


pbar = tqdm(train_dataloader)
for batch in pbar:
    image_input = batch[0]
    image_output = batch[1]
    # image_input, image_output = image_input.cuda(), image_output.cuda()
    # image_input, image_output = Variable(image_input), Variable(image_output)
    # y = model(image_input)

    break

print(image_input.shape)
print(image_output.shape)