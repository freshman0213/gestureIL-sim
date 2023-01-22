import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image

class LocalityPreservedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(512, 16, 1),
            nn.Flatten(),
            nn.Linear(16*7*7, 512),
            nn.ReLU()
        )

    def forward(self, features_with_locality):
        return self.module(features_with_locality)


class AttentionMaskPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet_encoder = nn.Sequential(*(list(models.resnet18(weights="DEFAULT").children())[:-2]))
        for child in self.resnet_encoder.children():
            for param in child.parameters():
                param.requires_grad = False

        self.front_additional_encoder = LocalityPreservedEncoder()
        self.side_addtional_encoder = LocalityPreservedEncoder()

        self.front_output_head = nn.Linear(1024, 7*7)
        self.side_output_head = nn.Linear(1024, 7*7) 

    def forward(self, front_gesture, side_gesture):
        front_features = self.front_additional_encoder(self.resnet_encoder(front_gesture))
        side_features = self.side_addtional_encoder(self.resnet_encoder(side_gesture))
        concat_features = torch.cat((front_features, side_features), dim=1)

        front_mask = torch.reshape(F.softmax(self.front_output_head(concat_features), dim=1), (-1, 1, 7, 7))
        side_mask = torch.reshape(F.softmax(self.side_output_head(concat_features), dim=1), (-1, 1, 7, 7))
        return front_mask, side_mask


class AttentionActionPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet_encoder = nn.Sequential(*(list(models.resnet18(weights="DEFAULT").children())[:-1]), nn.Flatten())
        for child in self.resnet_encoder.children():
            for param in child.parameters():
                param.requires_grad = False
        self.intermediate_features = None
        def hook(model, input, output):
            self.intermediate_features = input
        self.hook_handle = list(self.resnet_encoder.children())[-2].register_forward_hook(hook)
        
        self.score_mlp = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self.mask_predictor = AttentionMaskPredictor()

        self.front_additional_encoder = LocalityPreservedEncoder()
        self.side_addtional_encoder = LocalityPreservedEncoder()

        self.x_output_head = nn.Linear(1024, 5)
        self.y_output_head = nn.Linear(1024, 5)
        self.z_output_head = nn.Linear(1024, 5)
        self.gripper_output_head = nn.Linear(1024, 2)

    def forward(self, front_image, side_image, front_gesture_1, side_gesture_1, front_gesture_2, side_gesture_2):
        front_features = self.resnet_encoder(front_image)
        front_features_with_locality = self.intermediate_features[0]
        side_features = self.resnet_encoder(side_image)
        side_features_with_locality = self.intermediate_features[0]

        mask_score = F.softmax(self.score_mlp(torch.cat((front_features, side_features), dim=1)), dim=1)
        front_mask_1, side_mask_1 = self.mask_predictor(front_gesture_1, side_gesture_1)
        front_mask_2, side_mask_2 = self.mask_predictor(front_gesture_2, side_gesture_2)
        front_mask = mask_score[:, 0, None, None, None] * front_mask_1 + mask_score[:, 1, None, None, None] * front_mask_2
        side_mask = mask_score[:, 0, None, None, None] * side_mask_1 + mask_score[:, 1, None, None, None] * side_mask_2

        attention_front_features_with_locality = front_mask * front_features_with_locality
        attention_side_features_with_locality = side_mask * side_features_with_locality
        attention_front_features = self.front_additional_encoder(attention_front_features_with_locality)
        attention_side_features = self.side_addtional_encoder(attention_side_features_with_locality)
        attention_concat_features = torch.cat((attention_front_features, attention_side_features), dim=1)
        return F.softmax(self.x_output_head(attention_concat_features), dim=1), \
            F.softmax(self.y_output_head(attention_concat_features), dim=1), \
            F.softmax(self.z_output_head(attention_concat_features), dim=1), \
            F.softmax(self.gripper_output_head(attention_concat_features), dim=1)