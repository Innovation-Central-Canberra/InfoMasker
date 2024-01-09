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
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from dl import *
from network import *



def parse_args():
    desc = 'audio denoising using encoder decoder'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--batch_size', type=int, default=32)
    
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--continue_train', type=bool, default=False)
    
    parser.add_argument('--load_dir', type=str, default='model_store/')
    parser.add_argument('--save_dir', type=str, default='model_store/')
    
    return parser.parse_args()

def train(args):
    dataset_path = 'train/'

    train_dataloader = torch.utils.data.DataLoader(dataset=reverb_dataset(dataset_path),\
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=0,
                                                drop_last=False)


    model = ImageTransformNet_Spectrogram()
    model = model.cuda()
    # optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)


    restart_epoch = 0
    # Restart train or continue train
    if(args.continue_train==True):
        model_path = os.path.join(args.load_dir,'best.pth.tar')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if('optimizer_state_dict' in checkpoint ):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if('epoch' in checkpoint):
            restart_epoch = checkpoint['epoch']

    model.train()

    best_mse = 1e8

    for epoch in range(restart_epoch, args.epochs):
        avg_loss = []
        pbar = tqdm(train_dataloader)
        for batch in pbar:
            image_input = batch[0]
            image_output = batch[1]
            image_input, image_output = image_input.cuda(), image_output.cuda()
            image_input, image_output = Variable(image_input), Variable(image_output)
            print(image_input.shape)
            y = model(image_input)

            # Use l2 loss instead of l1 loss to avoid lots of zeros in result
            l2 = torch.nn.MSELoss()
            # l1 = torch.nn.L1Loss()
            loss = l2(y, image_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss.append(loss.item())

            pbar.set_postfix({"epoch": epoch, "loss":sum(avg_loss) / len(avg_loss)})
            # pbar.set_postfix({"loss":loss.item()})
            
        temp_loss = sum(avg_loss) / len(avg_loss)

        state = {'epoch': epoch,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict()} 
        if ((epoch+1)%10 ==0): 
            torch.save(state,os.path.join(args.save_dir, str(epoch+1)+'pth.tar'))

        if (temp_loss < best_mse):
            torch.save(state, os.path.join(args.save_dir, 'best.pth.tar'))
            best_mse = temp_loss
        

if __name__ == '__main__':
    args = parse_args()
    
    model = ImageTransformNet_Spectrogram()

    if args.mode == 'train':
        train(args)
