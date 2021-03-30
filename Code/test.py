import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch

from utils.Dataset import NewDataset
from Model.NNModel import Net
from Model.NNModel import ConfNet
from utils.utils import nn_train_test_split, predction_plots, test_confnet, loadData, test_nn, confidence, create_labelvec

import pickle
from argparse import ArgumentParser

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
        _, input_test, _, label_test = train_test_split(input, label, test_size=test_split)
        # ----------------------------------------------------------------------
        # Load or train SVM model
        # ----------------------------------------------------------------------
        svclassifier = pickle.load(open('./Model/svm_model_{}.sav'.format(i), 'rb'))
        print("----- SVM model {} succesfully loaded -----".format(i))
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
        _, test_data = nn_train_test_split(set, test_split)

        # ----------------------------------------------------------------------
        # Load or train Neural Network
        # ----------------------------------------------------------------------
        PATH = './Model/model_{}_net.pth'.format(i)
        nn_model = Net()
        nn_model.load_state_dict(torch.load(PATH))
        print("----- Neural Network {} succesfully loaded -----".format(i))

        # ----------------------------------------------------------------------
        # Test Neural Network
        # ----------------------------------------------------------------------
        test_nn(test_data, nn_model, i)
        # ----------------------------------------------------------------------
        # Compute confidence/distrust
        # ----------------------------------------------------------------------
        distrust = confidence(test_data, nn_model, svclassifier, i)
        fig_distrust = plt.figure()
        data_number = np.linspace(0, len(distrust), num=len(distrust))
        plt.xlabel("Sample")
        plt.ylabel("Distrust")
        plt.plot(data_number, distrust, '--r')

        fig_distrust.savefig("../Results/Distrust_Data_{}".format(i), dpi=300)

        predction_plots(test_data, nn_model, svclassifier, i)

        # ----------------------------------------------------------------------
        # Load or train Neural Confidence Network
        # ----------------------------------------------------------------------
        PATH = './Model/model_{}_confnet.pth'.format(i)
        confnet_model = ConfNet()
        confnet_model.load_state_dict(torch.load(PATH))
        print("----- Neural Confidence Network {} succesfully loaded -----".format(i))
        # ----------------------------------------------------------------------
        # Test Neural Confidence Network
        # ----------------------------------------------------------------------
        test_confnet(test_data, svclassifier, nn_model, confnet_model, i)


if __name__ == "__main__":
    main()
