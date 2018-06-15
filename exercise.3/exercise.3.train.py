import argparse
import pickle
import sys

import lmdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data


class udacityData(data.Dataset):
        def __init__(self, path):
                self.env = lmdb.open(path,
                                     max_readers=1,
                                     readonly=True,
                                     lock=False,
                                     readahead=False,
                                     meminit=False)

                with self.env.begin(write=False) as txn:
                        self.length = txn.stat()['entries']
                assert self.length

        def __getitem__(self, index):
                key = index  # TODO
                with self.env.begin(write=False) as txn:
                        value = pickle.loads(txn.get(key))
                        image = value['image']
                        angle = value['angle']
                return image, angle

        def __len__(self):
                return self.length


class CG23(nn.Module):
        def __init__(self):
                super(CG23, self).__init__()
                self.features = nn.Sequential(
                        nn.Conv2d(32, 64, kernel_size=3, padding=1).Shape,
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Dropout(0.25),

                        nn.Conv2d(64, 128, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Dropout(0.25),

                        nn.Conv2d(128, 1024, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Dropout(0.5),
                )
                self.classifier = nn.Sequential(
                        nn.Linear(1024, 1),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.5),
                        nn.Linear(1, 1)
                )

        def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x


def train(device, loader, model, optimizer, loss_fn, epoch):
        model.train()

        for batch_idx, (data, target) in enumerate(loader):
                # Move the input and target data on the GPU
                data, target = data.to(device), target.to(device)
                # Zero out gradients from previous step
                optimizer.zero_grad()
                # Forward pass of the neural net
                output = model(data)
                # Calculation of the loss function
                loss = loss_fn(output, target)
                # Backward pass (gradient computation)
                loss.backward()
                # Adjusting the parameters according to the loss function
                optimizer.step()
                # For plotting training curve
                if batch_idx % 10 == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                epoch, batch_idx * len(data), len(loader.dataset),
                                       100. * batch_idx / len(loader), loss.item()))
                        sys.stdout.flush()


def validate(device, loader, model):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
                for data, target in loader:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        # sum up batch loss
                        test_loss += F.cross_entropy(output,
                                                     target,
                                                     size_average=False).item()
                        # get the index of the max log-probability
                        pred = output.max(1, keepdim=True)[1]

                        # If you just want so see # the failing examples
                        # if not pred.eq(target.view_as(pred)):

                        # cv_mat = data.cpu().data.squeeze().numpy()
                        # cv_mat = cv2.resize(cv_mat, (400, 400))
                        # cv2.imshow("test image", cv_mat)
                        # print("Target label is : %d" % target.cpu().item())
                        # print("Predicted label is : %d" % (pred.cpu().data.item()))
                        # cv2.waitKey()

                        correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(loader)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(loader),
                100. * correct / len(loader)))
        sys.stdout.flush()


def main():
        parser = argparse.ArgumentParser(description='Train the cg23 baseline')
        parser.add_argument(
                '-d',
                '--datadir',
                type=str,
                nargs='?',
                default='training.data',
                help='Path to training data')
        parser.add_argument(
                '-e',
                '--epochs',
                type=int,
                nargs='?',
                default=10,
                help='Number of epochs to train')
        args = parser.parse_args()

        training_data = udacityData(args.datadir)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CG23().to(device)

        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        loss = F.mse_loss

        for epoch in range(1, args.epochs + 1):
                train(device, train_loader, model, optimizer, loss, epoch)
                validate(test_loader)


if __name__ == '__main__':
        main()
