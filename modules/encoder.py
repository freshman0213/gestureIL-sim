import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import pdb
from layers import CausalConv1D
from layers import Flatten
from layers import conv2d
from layers import View
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms
from PIL import Image



class MyEncoder(nn.Module):
    def __init__(self, action_dim, mode,continuous=False):
        """
        Image encoder taken from selfsupervised code
        """
        super().__init__()
        self.out_dim = 3 #TODO
        self.action_dim = action_dim
        self.modality = mode
        self.encoded_img_dim = 8 #TODO
        
        
        self.affine1 = nn.Linear(1000, 100)
        self.affine2 = nn.Linear(100, 10)

        self.action_mean = nn.Linear(6, self.action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, self.action_dim), requires_grad = True)


        # img
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, self.encoded_img_dim)  #TODO
        
        self.resnet2 = models.resnet18(pretrained=True)
        self.resnet2.fc = nn.Linear(512, self.encoded_img_dim)  #TODO
        
        self.newmodel1 = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.newmodel2 = torch.nn.Sequential(*(list(self.resnet2.children())[:-1]))

        self.mod3linear = nn.Linear(1024, self.encoded_img_dim) 
        
        
        self.transform = transforms.Compose([            #[1]
            transforms.Resize(224),                    #[2]
            transforms.ToTensor(),                     #[4]
            transforms.Normalize(                      #[5]
            mean=[0.485, 0.456, 0.406],                #[6]
            std=[0.229, 0.224, 0.225]                  #[7]
            )])

        self.transform2 = transforms.Compose([            #[1]
            transforms.Resize(224),                    #[2]
            transforms.ToTensor(),                     #[4]
            transforms.Normalize(                      #[5]
            mean=[0.485, 0.456, 0.406],                #[6]
            std=[0.229, 0.224, 0.225]                  #[7]
            )])


    def forward(self, front_image, top_image, state, status="train"):
        if status=="eval":
            self.resnet.eval()
            self.resnet2.eval()
            self.newmodel1.eval()
            self.newmodel2.eval()
        else:
            self.resnet.train()
            self.resnet2.train()
            self.newmodel1.eval()
            self.newmodel2.train()
        images = []
        images2 = []
        if self.mode==1: # only front image
            for i in range(front_image.shape[0]):
                im = Image.fromarray(front_image.cpu().numpy().astype(np.uint8)[i])
                img_t = self.transform(im)
                img_t = torch.unsqueeze(img_t, 0).to("cuda:0")
                images.append(img_t)
            batch_t = torch.cat(images).to("cuda:0")
            cur_z = self.resnet(batch_t).reshape((-1,self.encoded_img_dim))

        if self.mode==2: # only top image
            for i in range(top_image.shape[0]):
                im = Image.fromarray(top_image.cpu().numpy().astype(np.uint8)[i])
                img_t = self.transform(im)
                img_t = torch.unsqueeze(img_t, 0).to("cuda:0")
                images2.append(img_t)
            batch_t = torch.cat(images2).to("cuda:0")
            cur_z = self.resnet(batch_t).reshape((-1,self.encoded_img_dim))
        
        if self.mode == 3: #both
            for i in range(front_image.shape[0]):
                im = Image.fromarray(front_image.cpu().numpy().astype(np.uint8)[i])
                img_t = self.transform(im)
                img_t = torch.unsqueeze(img_t, 0).to("cuda:0")
                images.append(img_t)
            batch_t = torch.cat(images).to("cuda:0")
            feature1 = self.newmodel1(batch_t)
            feature1 = torch.flatten(feature1, 1)
            
            for i in range(top_image.shape[0]):
                im = Image.fromarray(top_image.cpu().numpy().astype(np.uint8)[i])
                img_t = self.transform(im)
                img_t = torch.unsqueeze(img_t, 0).to("cuda:0")
                images2.append(img_t)
            batch_t = torch.cat(images2).to("cuda:0")
            feature2 = self.newmodel2(batch_t)
            feature2 = torch.flatten(feature2, 1)
            
            cur_z = torch.cat((feature1,feature2),axis=1)
            cur_z = self.mod3linear(cur_z).reshape((-1,self.encoded_img_dim))
        
        cur_z = torch.cat((state, cur_z))

        x = torch.tanh(self.affine1(cur_z))
        action_mean = torch.tanh(self.affine2(x))
        action_std = torch.exp(self.action_log_std)
    
        return action_mean, self.action_log_std, action_std