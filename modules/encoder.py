import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pdb
import torchvision.models as models
from torchvision import transforms
from PIL import Image


class MyEncoder(nn.Module):
    def __init__(self, action_dim):
        """
        Image encoder taken from selfsupervised code
        """
        super().__init__()
        self.out_dim = 4 #TODO
        self.action_dim = action_dim
        self.encoded_img_dim = 8 #TODO
        
        
        self.affine1 = nn.Linear(1000, 100)
        self.affine2 = nn.Linear(100, 10)

        self.action_mean = nn.Linear(6, self.action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, self.action_dim), requires_grad = True)


        # img
        self.resnet_encoder_observation_front = nn.Sequential(*(list(models.resnet18(pretrained=True).children())[:-1]))
        self.resnet_encoder_observation_side = nn.Sequential(*(list(models.resnet18(pretrained=True).children())[:-1]))
        self.resnet_encoder_gesture_front = nn.Sequential(*(list(models.resnet18(pretrained=True).children())[:-1]))
        self.resnet_encoder_gesture_side = nn.Sequential(*(list(models.resnet18(pretrained=True).children())[:-1]))
#         for child in self.resnet_encoder.children():
#             for param in child.parameters():
#                 param.requires_grad = False

        self.front_img_linear = nn.Linear(512, self.encoded_img_dim)
        self.side_img_linear = nn.Linear(512, self.encoded_img_dim) 
        self.front_gesture1_linear = nn.Linear(512, self.encoded_img_dim)  
        self.front_gesture2_linear = nn.Linear(512, self.encoded_img_dim)  
        self.side_gesture1_linear = nn.Linear(512, self.encoded_img_dim)  
        self.side_gesture2_linear = nn.Linear(512, self.encoded_img_dim)  

#         self.linears = nn.Sequential(
#                                     nn.Linear(6*self.encoded_img_dim, self.out_dim*16),
#                                     nn.Linear(self.out_dim*16, self.out_dim*4), 
#                                     nn.Linear(self.out_dim*4, self.out_dim),
#                                     )
        self.x_output_head = nn.Linear(6*self.encoded_img_dim, 5)
        self.y_output_head = nn.Linear(6*self.encoded_img_dim, 5)
        self.z_output_head = nn.Linear(6*self.encoded_img_dim, 5)
        self.gripper_output_head = nn.Linear(6*self.encoded_img_dim, 2)

    def forward(self, front_image, side_image, front_gesture_1, side_gesture_1, front_gesture_2, side_gesture_2,status="train"):
        if status=="eval":
            self.resnet_encoder_observation_front.eval()
            self.resnet_encoder_observation_side.eval()
            self.resnet_encoder_gesture_front.eval()
            self.resnet_encoder_gesture_side.eval()
        else:
            self.resnet_encoder_observation_front.train()
            self.resnet_encoder_observation_side.train()
            self.resnet_encoder_gesture_front.train()
            self.resnet_encoder_gesture_side.train()
        front_image_feature = self.front_img_linear(torch.flatten(self.resnet_encoder_observation_front(front_image),1))
        side_image_feature = self.side_img_linear(torch.flatten(self.resnet_encoder_observation_side(side_image),1))
        front_gesture_feature = self.front_gesture1_linear(torch.flatten(self.resnet_encoder_gesture_front(front_gesture_1),1))
        side_gesture_feature = self.side_gesture1_linear(torch.flatten(self.resnet_encoder_gesture_side(side_gesture_1),1))
        front_gesture_feature2 = self.front_gesture2_linear(torch.flatten(self.resnet_encoder_gesture_front(front_gesture_2),1))
        side_gesture_feature2 = self.side_gesture2_linear(torch.flatten(self.resnet_encoder_gesture_side(side_gesture_2),1))
        
        cat_feature = torch.cat((front_image_feature,side_image_feature, front_gesture_feature, side_gesture_feature, front_gesture_feature2, side_gesture_feature2),axis=1)
        x1 = self.x_output_head(cat_feature).reshape((-1,5))
        x2 = self.y_output_head(cat_feature).reshape((-1,5))
        x3 = self.z_output_head(cat_feature).reshape((-1,5))
        x4 = self.gripper_output_head(cat_feature).reshape((-1,2))
        
#         action_mean = torch.tanh(x)
#         action_std = torch.exp(self.action_log_std)
    
        return x1,x2,x3,x4