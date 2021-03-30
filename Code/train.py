import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

import torch.optim as optim
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from utils.Dataset import NewDataset
from Model.NNModel import Net
from Model.NNModel import ConfNet
from utils.utils import nn_train_test_split, predction_plots, test_confnet, loadData, test_nn, confidence, create_labelvec

import pickle


def train_nn(train_data, model, i):
    optimizer = optim.RMSprop(model.parameters(), lr=0.01)
    loss_fn = F.mse_loss

    for epoch in range(0, 10):
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
    return model


def train_confnet(train_data, svm, pre_trained_model, confnet, i):
    optimizer = optim.RMSprop(confnet.parameters(), lr=0.01)
    loss_fn = F.mse_loss

    for epoch in range(0, 10):
        for batch, (data, target) in enumerate(train_data):
            optimizer.zero_grad()
            label_dic = create_labelvec(target)
            labelvec = torch.Tensor([[label_dic[1], label_dic[0]]])
            out_net = pre_trained_model(data)
            output_svm = svm.predict(data)[0]
            output_svm = torch.Tensor([[output_svm]])
            all_data = torch.cat([data, out_net, output_svm], 1)
            out = confnet(all_data)
            loss = loss_fn(out, labelvec)
            loss.backward()
            optimizer.step()

            print('Train Epoch: {} [{}/{} ({:.0f}%)] \tLoss {:.6f} '.format(epoch,
                                                                            batch * len(data), len(train_data),
                                                                            100. * batch / len(train_data), loss.data))

    PATH = './Model/model_{}_confnet.pth'.format(i)
    torch.save(confnet.state_dict(), PATH)
    return confnet


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


def main():
    print('Start Coding, Have fun!')
    # ----------------------------------------------------------------------
    # Save parameters
    # ----------------------------------------------------------------------

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
        # Train SVM model
        # ----------------------------------------------------------------------
        svclassifier = train_svm(input_train, label_train, i)
        # ----------------------------------------------------------------------
        # Test SVM model
        # ----------------------------------------------------------------------
        label_pred = svclassifier.predict(input_test)
        print("SVM results for Dataset {}".format(i))
        print(confusion_matrix(label_test, label_pred))
        print(classification_report(label_test, label_pred))

        # ----------------------------------------------------------------------
        # Load and split data for NN
        # ----------------------------------------------------------------------
        set = NewDataset('../Dataset/trainingdata_{}.xls'.format(i))
        train_data, test_data = nn_train_test_split(set, test_split)

        # ----------------------------------------------------------------------
        # Train Neural Network
        # ----------------------------------------------------------------------
        model = Net()
        trained_nn_model = train_nn(train_data, model, i)
        # ----------------------------------------------------------------------
        # Test Neural Network
        # ----------------------------------------------------------------------
        test_nn(test_data, trained_nn_model, i)

        # ----------------------------------------------------------------------
        # Compute confidence/distrust
        # ----------------------------------------------------------------------
        distrust = confidence(test_data, trained_nn_model, svclassifier, i)
        fig_distrust = plt.figure()
        data_number = np.linspace(0, len(distrust), num=len(distrust))
        plt.xlabel("Sample")
        plt.ylabel("Distrust")
        plt.plot(data_number, distrust, '--r')

        fig_distrust.savefig("../Results/Distrust_Data_{}".format(i), dpi=300)

        predction_plots(test_data, trained_nn_model, svclassifier, i)

        # ----------------------------------------------------------------------
        # Load or train Neural Confidence Network
        # ----------------------------------------------------------------------
        confnet_model = ConfNet()
        trained_confnet_model = train_confnet(train_data, svclassifier, trained_nn_model, confnet_model, i)

        # ----------------------------------------------------------------------
        # Test Neural Confidence Network
        # ----------------------------------------------------------------------
        test_confnet(test_data, svclassifier, trained_nn_model, trained_confnet_model, i)


if __name__ == "__main__":
    main()
