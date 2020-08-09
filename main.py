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

path='./faces/'
BATCH_SIZE=128
decive=torch.device('cuda' if torch.cuda.is_available()  else 'cpu')
encoder=Generator(100,96,3).to(decive)
decoder=Discriminator(3,96).to(decive)
dataset=GAN_Dataset(path,os.listdir(path))
dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)
positive_label=1
negetive_label=0

loss_fn=nn.BCELoss().to(decive)
optim_encoder=Optim.Adam(encoder.parameters(),lr=0.0002,betas=(0.9,0.999))
optim_decoder=Optim.Adam(decoder.parameters(),lr=0.0001,betas=(0.9,0.999))

epoches=100
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

for epoch in range(epoches):
    for x in dataloader:
        optim_decoder.zero_grad()

        x=x.cuda()

        noise=torch.randn(len(x),100, 1, 1,).cuda()
        noise_img = encoder(noise)
        label = torch.ones(len(x)).cuda()
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



    torch.save(encoder.state_dict(),'generator.pkl')
    torch.save(decoder.state_dict(),'discritor.pkl')
image=Image.fromarray(np.asanyarray(255*noise_img[0].detach().cpu().numpy(),dtype=np.uint8).transpose(1,2,0))
plt.imshow(image)
plt.show()
