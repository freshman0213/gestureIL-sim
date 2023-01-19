import torch
import pdb
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from PIL import Image
import os, os.path
from torchvision import transforms

class Datasettrain(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self):
        'Initialization'
        data_path = "../datasets/pick-and-place/"
        train_demo_start = 0
        train_demo_end = 10 # adjust
        self.front_images = []
        self.side_images = []
        self.front_gestures_1 = [] #start
        self.side_gestures_1 = []
        self.front_gestures_2 = [] #end
        self.side_gestures_2 = []
        self.actions = []
        for i in range(train_demo_start, train_demo_end):
            print("loading "+ f'{i:03d}'+" demo")
            pathname_demo_0 = data_path+f'{i:03d}'+"/demonstration/0/"
            pathname_demo_1 = data_path+f'{i:03d}'+"/demonstration/1/"
            pathname_gesture_0 = data_path+f'{i:03d}'+"/gesture/0/"
            pathname_gesture_1 = data_path+f'{i:03d}'+"/gesture/1/"
            self.actions.append(np.load(data_path+f'{i:03d}'+"/demonstration/actions.npy"))
            step = 0
            for path in os.listdir(pathname_demo_0):
                front_image =  Image.open(pathname_demo_0 + path).convert('RGB')
                self.front_images.append(front_image)
                step += 1
            for path in os.listdir(pathname_demo_1):
                side_image = Image.open(pathname_demo_1 + path).convert('RGB')
                self.side_images.append(side_image)
            for path in os.listdir(pathname_gesture_0):
                if path == '0000.jpg':
                    front_gesture_1 =  Image.open(pathname_gesture_0 + path).convert('RGB')
                    for i in range(step):
                        self.front_gestures_1.append(front_gesture_1)
                else:
                    front_gesture_2 =  Image.open(pathname_gesture_0 + path).convert('RGB')
                    for i in range(step):
                        self.front_gestures_2.append(front_gesture_2)
            for path in os.listdir(pathname_gesture_1):
                if path == '0000.jpg':
                    side_gesture_1 =  Image.open(pathname_gesture_1 + path).convert('RGB')
                    for i in range(step):
                        self.side_gestures_1.append(side_gesture_1)
                else:
                    side_gesture_2 =  Image.open(pathname_gesture_1 + path).convert('RGB')
                    for i in range(step):
                        self.side_gestures_2.append(side_gesture_2)
            print(len(self.front_images), len(self.actions))
        
        self.actions = np.concatenate(self.actions, axis = 0)
        assert len(self.front_gestures_1) == len(self.front_gestures_2)
        assert len(self.side_gestures_1) == len(self.side_gestures_2)
        # pdb.set_trace()
                    
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.front_images)

  def __getitem__(self, index):
        'Generates one sample of data'
        front_images = self.front_images[index]
        side_images = self.side_images[index]
        front_gestures_1 = self.front_gestures_1[index]
        side_gestures_1 = self.side_gestures_1[index]
        front_gestures_2 = self.front_gestures_2[index]
        side_gestures_2 = self.side_gestures_2[index]
        action = self.actions[index]
        convert_tensor = transforms.ToTensor() # can do any preprocessing here
        return convert_tensor(front_images),convert_tensor(side_images),convert_tensor(front_gestures_1), convert_tensor(side_gestures_1),convert_tensor(front_gestures_2), convert_tensor(side_gestures_2), torch.from_numpy(action)

class Datasettest(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self):
        'Initialization'
        data_path = "../datasets/pick-and-place/"
        test_demo_start = 990
        test_demo_end = 1000 # adjust
        self.front_images = []
        self.side_images = []
        self.front_gestures_1 = [] #start
        self.side_gestures_1 = []
        self.front_gestures_2 = [] #end
        self.side_gestures_2 = []
        self.actions = []
        for i in range(test_demo_start, test_demo_end):
            print("loading "+ f'{i:03d}'+" demo")
            pathname_demo_0 = data_path+f'{i:03d}'+"/demonstration/0/"
            pathname_demo_1 = data_path+f'{i:03d}'+"/demonstration/1/"
            pathname_gesture_0 = data_path+f'{i:03d}'+"/gesture/0/"
            pathname_gesture_1 = data_path+f'{i:03d}'+"/gesture/1/"
            self.actions.append(np.load(data_path+f'{i:03d}'+"/demonstration/actions.npy"))
            step = 0
            for path in os.listdir(pathname_demo_0):
                front_image =  Image.open(pathname_demo_0 + path).convert('RGB')
                self.front_images.append(front_image)
                step += 1
            for path in os.listdir(pathname_demo_1):
                side_image = Image.open(pathname_demo_1 + path).convert('RGB')
                self.side_images.append(side_image)
            for path in os.listdir(pathname_gesture_0):
                if path == '0000.jpg':
                    front_gesture_1 =  Image.open(pathname_gesture_0 + path).convert('RGB')
                    for i in range(step):
                        self.front_gestures_1.append(front_gesture_1)
                else:
                    front_gesture_2 =  Image.open(pathname_gesture_0 + path).convert('RGB')
                    for i in range(step):
                        self.front_gestures_2.append(front_gesture_2)
            for path in os.listdir(pathname_gesture_1):
                if path == '0000.jpg':
                    side_gesture_1 =  Image.open(pathname_gesture_1 + path).convert('RGB')
                    for i in range(step):
                        self.side_gestures_1.append(side_gesture_1)
                else:
                    side_gesture_2 =  Image.open(pathname_gesture_1 + path).convert('RGB')
                    for i in range(step):
                        self.side_gestures_2.append(side_gesture_2)
            print(len(self.front_images), len(self.actions))
        
        self.actions = np.concatenate(self.actions, axis = 0)
        assert len(self.front_gestures_1) == len(self.front_gestures_2)
        assert len(self.side_gestures_1) == len(self.side_gestures_2)
        # pdb.set_trace()
                    
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.front_images)

  def __getitem__(self, index):
        'Generates one sample of data'
        front_images = self.front_images[index]
        side_images = self.side_images[index]
        front_gestures_1 = self.front_gestures_1[index]
        side_gestures_1 = self.side_gestures_1[index]
        front_gestures_2 = self.front_gestures_2[index]
        side_gestures_2 = self.side_gestures_2[index]
        action = self.actions[index]
        convert_tensor = transforms.ToTensor()
        return convert_tensor(front_images),convert_tensor(side_images),convert_tensor(front_gestures_1), convert_tensor(side_gestures_1),convert_tensor(front_gestures_2), convert_tensor(side_gestures_2), torch.from_numpy(action)
