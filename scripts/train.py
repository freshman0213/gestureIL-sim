import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from modules.datasets import DemonstrationDataset
from modules.attention_predictor import AttentionActionPredictor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main(args):
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    training_set = DemonstrationDataset(args.data_path, transform=transform)
    testing_set = DemonstrationDataset(args.data_path, split="test", transform=transform)
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(testing_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

    agent = AttentionActionPredictor().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(agent.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(args.num_epochs):
        agent.train()
        num_iter, avg_loss = 0, 0.
        for data in train_loader:
            front_image, side_image, front_gesture_1, side_gesture_1, front_gesture_2, side_gesture_2, action = data
            front_image = front_image.to(device)
            side_image = side_image.to(device)
            front_gesture_1 = front_gesture_1.to(device)
            side_gesture_1 = side_gesture_1.to(device)
            front_gesture_2 = front_gesture_2.to(device)
            side_gesture_2 = side_gesture_2.to(device)
            action = action.to(device)

            optimizer.zero_grad()
            predicted_x_action, predicted_y_action, predicted_z_action, predicted_gripper_action = \
                agent(front_image, side_image, front_gesture_1, side_gesture_1, front_gesture_2, side_gesture_2)
            loss = loss_fn(predicted_x_action, action[:, 0]) + \
                loss_fn(predicted_y_action, action[:, 1]) + \
                loss_fn(predicted_z_action, action[:, 2]) + \
                loss_fn(predicted_gripper_action, action[:, 3])
            loss.backward()
            optimizer.step()

            num_iter += 1
            avg_loss += loss.item()

        avg_loss /= num_iter
        print(f"[{epoch:03d}] Training Loss: {avg_loss:.4f}")

        if (epoch+1) % args.validation_cycle == 0:
            with torch.no_grad():
                agent.eval()
                num_iter, avg_loss = 0, 0.
                for data in test_loader:
                    front_image, side_image, front_gesture_1, side_gesture_1, front_gesture_2, side_gesture_2, action = data
                    front_image = front_image.to(device)
                    side_image = side_image.to(device)
                    front_gesture_1 = front_gesture_1.to(device)
                    side_gesture_1 = side_gesture_1.to(device)
                    front_gesture_2 = front_gesture_2.to(device)
                    side_gesture_2 = side_gesture_2.to(device)
                    action = action.to(device)

                    predicted_x_action, predicted_y_action, predicted_z_action, predicted_gripper_action = \
                        agent(front_image, side_image, front_gesture_1, side_gesture_1, front_gesture_2, side_gesture_2)
                    loss = loss_fn(predicted_x_action, action[:, 0]) + \
                        loss_fn(predicted_y_action, action[:, 1]) + \
                        loss_fn(predicted_z_action, action[:, 2]) + \
                        loss_fn(predicted_gripper_action, action[:, 3])
                    
                    num_iter += 1
                    avg_loss += loss.item()
                
                avg_loss /= num_iter
                print(f"[{epoch:03d}] Validation Loss: {avg_loss:.4f}")

        # TODO: Save checkpoints



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="datasets/pick-and-place")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--validation_cycle", type=int, default=5,
                        help="Run validation every the specified number epochs")
    args = parser.parse_args()

    main(args)