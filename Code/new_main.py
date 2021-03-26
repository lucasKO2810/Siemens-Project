import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from NNModel import Net
from utils.Dataset import NewDataset


def train(epoch):
    for batch, (data, target) in enumerate(train_data):
        # data = Variable(data)
        # target = Variable(target)
        optimizer.zero_grad()

        out = model(data)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)] \tLoss {:.6f} '.format(epoch,
                                                                        batch * len(data), len(train_data),
                                                                        100. * batch / len(train_data), loss.data))


def main():
    print('Start Coding, Have fun!')

    for epoch in range(0, 10):
        train(epoch)


if __name__ == "__main__":
    # parameter initialization
    model = Net()

    optimizer = optim.RMSprop(model.parameters(), lr=0.01)

    set = NewDataset('../Dataset/trainingdata_a.xls')
    train_data = torch.utils.data.DataLoader(set, batch_size=4, shuffle=True)

    loss_fn = F.mse_loss

    # run main
    main()
