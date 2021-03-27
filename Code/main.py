import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

import torch.optim as optim
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from utils.Dataset import NewDataset
from Model.NNModel import Net

import pickle
from argparse import ArgumentParser


def loadData():

    datasets = {
        "a": [],
        "b": [],
        "c": []
    }

    for i in ["a", "b", "c"]:
        datasets[i] = pd.read_excel("../Dataset/trainingdata_{}.xls".format(i))

    return datasets


def dataPlot(data):
    X = []
    Y = []
    Lables = []
    data_num = 0
    for i in ["a", "b", "c"]:
        Classes = {
            "red": [],
            "blue": []
        }

        dataset = data[i]
        X.append(dataset["x_i1"])
        Y.append(dataset["x_i2"])
        Lables.append(dataset["l_i"])

        index = 0
        for label in Lables[data_num]:
            if label == 1:
                Classes["red"].append([X[data_num][index], Y[data_num][index]])
            else:
                Classes["blue"].append([X[data_num][index], Y[data_num][index]])
            index = index + 1

        fig = plt.figure()

        x_red = []
        y_red = []
        for x, y in Classes["red"]:
            x_red.append(x)
            y_red.append(y)
        plt.plot(x_red,y_red, 'or')

        x_blue = []
        y_blue = []
        for x, y in Classes["blue"]:
            x_blue.append(x)
            y_blue.append(y)
        plt.plot(x_blue, y_blue, 'ob')

        fig.savefig("../Dataset/data_{}".format(i), dpi=300)
        data_num = data_num + 1


def train_nn(train_data, model, i):
    optimizer = optim.RMSprop(model.parameters(), lr=0.01)
    loss_fn = F.mse_loss

    for epoch in range(0,10):
        for batch, (data, target) in enumerate(train_data):
            optimizer.zero_grad()

            out = model(data)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()

            print('Train Epoch: {} [{}/{} ({:.0f}%)] \tLoss {:.6f} '.format(epoch,
                                                                            batch * len(data), len(train_data),
                                                                            100. * batch / len(train_data), loss.data))

    PATH = './Model/model_{}_net.pth'.format(i)
    torch.save(model.state_dict(), PATH)
    print("Training {} Finished".format(i))
    print(" ")


def test_nn(test_data, model, i):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch, (data, target) in enumerate(test_data):
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print("Test {}".format(i))
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


def confidence(test_data, model, svm, i):
    distance = []
    for batch, (data, target) in enumerate(test_data):
        outputs_nn = model(data)
        nn_value = outputs_nn.item()
        output_svm = svm.predict(data)[0]

        ##### Distance
        distance.append(abs(output_svm - nn_value))

    return distance


def train_svm(input_train, label_train, i):
    # ----------------------------------------------------------------------
    # Train SVM
    # ----------------------------------------------------------------------
    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(input_train, label_train)
    # ----------------------------------------------------------------------
    # Save SVM
    # ----------------------------------------------------------------------
    filename = './Model/svm_model_{}.sav'.format(i)
    pickle.dump(svclassifier, open(filename, 'wb'))

    return svclassifier


def main(args):
    print('Start Coding, Have fun!')
    # ----------------------------------------------------------------------
    # Save parameters
    # ----------------------------------------------------------------------
    loadSVM = args.loadSVM
    loadNN = args.loadNN

    datasets = loadData()
    test_split = 0.1
    for i in ["a", "b", "c"]:
        # ----------------------------------------------------------------------
        # Get inputs and targets & split data
        # ----------------------------------------------------------------------
        input = datasets[i][["x_i1", "x_i2"]]
        label = datasets[i]['l_i']
        input_train, input_test, label_train, label_test = train_test_split(input, label, test_size=test_split)
        # ----------------------------------------------------------------------
        # Load or train SVM model
        # ----------------------------------------------------------------------
        if loadSVM:
            svclassifier = pickle.load(open('./Model/svm_model_{}.sav'.format(i), 'rb'))
            print("----- SVM model {} succesfully loaded -----".format(i))
        else:
            svclassifier = train_svm(input_train, label_train, i)
        # ----------------------------------------------------------------------
        # Test SVM model
        # ----------------------------------------------------------------------
        label_pred = svclassifier.predict(input_test)
        print("SVM results  for Dataset {}".format(i))
        print(confusion_matrix(label_test, label_pred))
        print(classification_report(label_test, label_pred))
        # ----------------------------------------------------------------------
        # Load and split data for NN
        # ----------------------------------------------------------------------
        set = NewDataset('../Dataset/trainingdata_{}.xls'.format(i))

        train_range = list(range(0, int(len(set) - test_split * len(set))))
        test_range = list(range(int(len(set) - test_split * len(set)), len(set)))

        train_set = torch.utils.data.Subset(set, train_range)
        tes_set = torch.utils.data.Subset(set, test_range)

        train_data = torch.utils.data.DataLoader(train_set, batch_size=4,
                                                  shuffle=True)

        test_data = torch.utils.data.DataLoader(tes_set, batch_size=1,
                                                shuffle=True)
        # ----------------------------------------------------------------------
        # Load or train Neural Network
        # ----------------------------------------------------------------------
        if loadNN:
            PATH = './Model/model_{}_net.pth'.format(i)
            model = Net()
            model.load_state_dict(torch.load(PATH))
            print("----- Neural Network {} succesfully loaded -----".format(i))
        else:
            model = Net()
            train_nn(train_data, model, i)
        # ----------------------------------------------------------------------
        # Test Neural Network
        # ----------------------------------------------------------------------
        test_nn(test_data, model, i)
        # ----------------------------------------------------------------------
        # Compute confidence/distrust
        # ----------------------------------------------------------------------
        distrust = confidence(test_data, model, svclassifier, i)
        print(distrust)
        fig_distrust = plt.figure()
        data_number = np.linspace(0, len(distrust), num=len(distrust))
        plt.xlabel("Sample")
        plt.ylabel("Distrust")
        plt.plot(data_number, distrust, '--r')
        plt.show()
        fig_distrust.savefig("../Distrust_Data_{}".format(i), dpi=300)


def options():
    parser = ArgumentParser()
    parser.add_argument('--loadSVM',
                        type=int,
                        default=1,  # True
                        choices=[0, 1],
                        help="Load SVM model")
    parser.add_argument('--loadNN',
                        type=int,
                        default=1,  # True
                        choices=[0, 1],
                        help="Load Neural Network")
    return parser.parse_args()


if __name__ =="__main__":
    main(options())
