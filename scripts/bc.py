import numpy as np
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import pdb
import gym
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from modules.encoder import MyEncoder
from modules.datasets import DemonstrationDataset
import resource
import torch.distributions as D
from torchvision import transforms
import time

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save_interval', type=int, default=1, metavar='N',
                    help='interval between testing status logs (default: 5)')
parser.add_argument('--num_epochs', type=int, default=200, help='the numer of epochs')
parser.add_argument('--output_model_path', type=str, default="outputs/resnetcat/")
parser.add_argument('--valid_loss', default="l2", metavar='G',
                    help='name of validation loss function')
parser.add_argument('--goal', default="middle", metavar='G',
                    help='name of validation loss function')
parser.add_argument('--random', default=0,
                    help='name of validation loss function')
parser.add_argument('--continuous', default=0,
                    help='name of validation loss function')
parser.add_argument("--data_path", type=str, default="datasets/pick-and-place")

if __name__ == "__main__":
    args = parser.parse_args()

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    batch_size = args.batch_size
    # Generators
    transform = transforms.Compose([            #[1]
                transforms.Resize(224),                    #[2]
                transforms.ToTensor(),                     #[4]
                transforms.Normalize(                      #[5]
                mean=[0.485, 0.456, 0.406],                #[6]
                std=[0.229, 0.224, 0.225]                  #[7]
                )])
    print("loading training data")
    training_set = DemonstrationDataset(args.data_path, transform=transform)
    print("loading validation data")
    testing_set = DemonstrationDataset(args.data_path, split="test", transform=transform)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(testing_set, batch_size=1, shuffle=False, num_workers=4)


    num_actions = 4 #TODO
    max_epoch = args.num_epochs

    encoder = MyEncoder(action_dim = num_actions).to(device)#.float()
    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(encoder.parameters(), lr=0.0003, weight_decay=0.0005)

    valid_loss = []
    train_loss = []
    bestloss=np.inf
    for epoch in range(max_epoch):
        if epoch % args.save_interval == 0:
            with torch.no_grad():
                loss = 0
                for iter_, data in enumerate(test_loader):
                    front_image, side_image, front_gesture_1, side_gesture_1, front_gesture_2, side_gesture_2, action = data ##
                    front_image, side_image, front_gesture_1, side_gesture_1, front_gesture_2, side_gesture_2, action  = front_image.to(device),side_image.to(device), front_gesture_1.to(device),side_gesture_1.to(device),front_gesture_2.to(device), side_gesture_2.to(device),action.to(device) 
                    predicted_x_action, predicted_y_action, predicted_z_action, predicted_gripper_action = encoder(front_image, side_image, front_gesture_1, side_gesture_1, front_gesture_2, side_gesture_2,status="eval")

#                     l2loss = nn.MSELoss()
#                     loss += l2loss(sim_action_mean, action)
                    loss += loss_fn(predicted_x_action, action[:, 0]) + \
                        loss_fn(predicted_y_action, action[:, 1]) + \
                        loss_fn(predicted_z_action, action[:, 2]) + \
                        loss_fn(predicted_gripper_action, action[:, 3])
#                     print(torch.argmax(nn.Softmax()(predicted_x_action)),torch.argmax(nn.Softmax()(predicted_y_action)),torch.argmax(nn.Softmax()(predicted_z_action)),torch.argmax(nn.Softmax()(predicted_gripper_action)), action)
                print("validation loss:", loss.item())
                valid_loss.append(loss.item())
                if loss<bestloss:
                    bestloss = loss
                print("bestloss", bestloss)

        #training
        lossall=0
        for iter_, data in enumerate(train_loader):
            front_image, side_image, front_gesture_1, side_gesture_1, front_gesture_2, side_gesture_2, action = data ##
            front_image, side_image, front_gesture_1, side_gesture_1, front_gesture_2, side_gesture_2, action  = front_image.to(device),side_image.to(device), front_gesture_1.to(device),side_gesture_1.to(device),front_gesture_2.to(device), side_gesture_2.to(device),action.to(device) 
            optimizer.zero_grad()
            predicted_x_action, predicted_y_action, predicted_z_action, predicted_gripper_action = encoder(front_image, side_image, front_gesture_1, side_gesture_1, front_gesture_2, side_gesture_2,status="train")

#             l2loss = nn.MSELoss()
#             loss = l2loss(sim_action_mean, action)
            loss = loss_fn(predicted_x_action, action[:, 0]) + \
                        loss_fn(predicted_y_action, action[:, 1]) + \
                        loss_fn(predicted_z_action, action[:, 2]) + \
                        loss_fn(predicted_gripper_action, action[:, 3])
            loss.backward()
            optimizer.step()
            lossall+=loss.item()
        print('epoch {:03d}, training loss {}'.format(epoch, lossall))
        train_loss.append(lossall)
        if (epoch+1) % 100 ==0:
            torch.save(encoder.state_dict(), os.path.join(args.output_model_path,'seed{}_epoch{}.pth'.format(args.seed, epoch)))
            
        
        
# np.save(os.path.join(args.output_model_path, 'valid_loss.npy'), np.array(valid_loss))
# np.save(os.path.join(args.output_model_path, 'train_loss.npy'), np.array(train_loss))
           

###Sample calling arguments###
###python bc.py --demo_files '/mnt/c/Users/Yilun/Desktop/Robot/robomimic/datasets/lift/ph/image.hdf5' --output_model_path ../../models/ --num_modality 3 --proprio 1 --image 1 --selfimage 1
