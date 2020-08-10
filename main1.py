from model import Generator,Discriminator
from dataset import GAN_Dataset
from torch.utils.data import DataLoader
import torch
import torch.optim as Optim
import torch.nn as nn
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('-d','--dataset',help='your dataset path')
parser.add_argument('-z','--nz',default=100,help='your noisy dim')
parser.add_argument('-b','--batch_size',default=128,help='your batch size')
parser.add_argument('-c','--if_cuda',default=False,help='if use cuda')
parser.add_argument('--g_lr',default=0.0002,help='the generator lr')
parser.add_argument('--d_lr',default=0.0001,help='the discritor lr')
parser.add_argument('if_train',default=True,help='if train your dataset or generator your imgs')
parser.add_argument('many_imgs',default=1,help='if you dont train your model,you can input how many imgs you want')
parser.add_argument('model_file',default='./generator.pt',help='your generator model file')
args=parser.parse_args()

path=args.d
BATCH_SIZE=args.b
decive=torch.device('cuda' if torch.cuda.is_available() and args.c else 'cpu')
encoder=Generator(100,96,3).to(decive)
decoder=Discriminator(3,96).to(decive)
dataset=GAN_Dataset(path,os.listdir(path))
dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)
positive_label=1
negetive_label=0

loss_fn=nn.BCELoss().to(decive)
optim_encoder=Optim.Adam(encoder.parameters(),lr=args.g_lr,betas=(0.9,0.999))
optim_decoder=Optim.Adam(decoder.parameters(),lr=args.d_lr,betas=(0.9,0.999))
epoches=10
step=0

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

encoder.apply(weights_init)
decoder.apply(weights_init)

def train(dataloader,epoches):
    for epoch in range(epoches):
        for x in dataloader:
            optim_decoder.zero_grad()

            x=x.to(device)

            noise=torch.randn(len(x), args.nz, 1, 1,).to(device)
            noise_img = encoder(noise)
            label = torch.ones(len(x)).to(device)
            positive_label = label * 1
            negetive_label = label * 0
            pred_y=decoder(x)
            loss_1=loss_fn(pred_y,positive_label)

            pred_y=decoder(noise_img)
            loss_2=loss_fn(pred_y,negetive_label)
            loss=loss_1+loss_2
            loss.backward(retain_graph=True)
            optim_decoder.step()

            optim_encoder.zero_grad()
            pred_y_noise = decoder(noise_img)

            loss_negetive_f1=loss_fn(pred_y_noise,positive_label)

            loss_negetive_f1.backward()
            optim_encoder.step()
            print(loss,loss_negetive_f1)


        image=Image.fromarray(np.asanyarray(255*noise_img[0].detach().cpu().numpy(),dtype=np.uint8).transpose(1,2,0))
        plt.imshow(image)
        plt.show()
        torch.save(encoder.state_dict(),'generator.pt')
        torch.save(decoder.state_dict(),'discritor.pt')

def generator():
    noise = torch.randn(1, args.nz, 1, 1, ).to(decive)
    noise_img = encoder(noise)
    return noise_img

if args.if_train:
    train(dataloader,epoches)

else:
    encoder.load_state_dict(torch.load(args.model_file))
    for i in range(args.many_imgs):
        noise_img=generator()
        image = Image.fromarray(np.asanyarray(255 * noise_img[0].detach().cpu().numpy(), dtype=np.uint8).transpose(1, 2, 0))
        plt.imshow(image)
        plt.show()
