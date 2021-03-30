import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def loadData():
    datasets = {
        "a": [],
        "b": [],
        "c": []
    }

    for i in ["a", "b", "c"]:
        datasets[i] = pd.read_excel("../Dataset/trainingdata_{}.xls".format(i))

    return datasets


def nn_train_test_split(dataset, test_size):
    train_range = list(range(0, int(len(dataset) - test_size * len(dataset))))
    test_range = list(range(int(len(dataset) - test_size * len(dataset)), len(dataset)))

    train_set = torch.utils.data.Subset(dataset, train_range)
    tes_set = torch.utils.data.Subset(dataset, test_range)

    train_data = torch.utils.data.DataLoader(train_set, batch_size=1,
                                             shuffle=True)

    test_data = torch.utils.data.DataLoader(tes_set, batch_size=1,
                                            shuffle=True)
    return train_data, test_data


def predction_plots(test_data, model, svm, i):
    prediction_nn = {
        "red": [],
        "blue": []
    }
    prediction_svm = {
        "red": [],
        "blue": []
    }
    ground_trouth = {
        "red": [],
        "blue": []
    }
    for batch, (data, target) in enumerate(test_data):
        outputs_nn = model(data)
        nn_value = outputs_nn.item()
        output_svm = svm.predict(data)[0]
        x = data[0][0].item()
        y = data[0][1].item()

        if target == 1:
            ground_trouth["red"].append([x, y])
        else:
            ground_trouth["blue"].append([x, y])

        if nn_value >= 0.5:
            prediction_nn["red"].append([x, y])
        else:
            prediction_nn["blue"].append([x, y])

        if output_svm == 1:
            prediction_svm["red"].append([x, y])
        else:
            prediction_svm["blue"].append([x, y])

    # ----------------------------------------------------------------------
    #### Ground Trouth
    # ----------------------------------------------------------------------
    fig_gt = plt.figure()
    x_red = []
    y_red = []
    for x, y in ground_trouth["red"]:
        x_red.append(x)
        y_red.append(y)
    plt.plot(x_red, y_red, 'or')

    x_blue = []
    y_blue = []
    for x, y in ground_trouth["blue"]:
        x_blue.append(x)
        y_blue.append(y)
    plt.plot(x_blue, y_blue, 'ob')
    fig_gt.savefig("../Results/testdata_{}".format(i), dpi=300)
    # ----------------------------------------------------------------------
    #### NN Predictions
    # ----------------------------------------------------------------------
    fig_nn = plt.figure()
    x_red_nn = []
    y_red_nn = []
    for x, y in prediction_nn["red"]:
        x_red_nn.append(x)
        y_red_nn.append(y)
    plt.plot(x_red_nn, y_red_nn, 'or')

    x_blue_nn = []
    y_blue_nn = []
    for x, y in prediction_nn["blue"]:
        x_blue_nn.append(x)
        y_blue_nn.append(y)
    plt.plot(x_blue_nn, y_blue_nn, 'ob')
    fig_nn.savefig("../Results/nn_pred_{}".format(i), dpi=300)
    # ----------------------------------------------------------------------
    ### SVM Predictions
    # ----------------------------------------------------------------------
    fig_svm = plt.figure()
    x_red_svm = []
    y_red_svm = []
    for x, y in prediction_svm["red"]:
        x_red_svm.append(x)
        y_red_svm.append(y)
    plt.plot(x_red_svm, y_red_svm, 'or')

    x_blue_svm = []
    y_blue_svm = []
    for x, y in prediction_svm["blue"]:
        x_blue_svm.append(x)
        y_blue_svm.append(y)
    plt.plot(x_blue_svm, y_blue_svm, 'ob')
    fig_svm.savefig("../Results/svm_pred_{}".format(i), dpi=300)


def test_confnet(test_data, svm, pre_trained_model, confnet, i):
    correct = 0
    total = 0
    output_collection = []
    with torch.no_grad():
        for batch, (data, target) in enumerate(test_data):
            out_net = pre_trained_model(data)
            output_svm = svm.predict(data)[0]
            output_svm = torch.Tensor([[output_svm]])
            all_data = torch.cat([data, out_net, output_svm], 1)
            outputs = confnet(all_data)
            output_collection.append(max(outputs.data[0]))
            test = outputs.data[0][0]
            predicted = 1 if outputs.data[0][0] >= 0.5 else 0
            total += target.size(0)
            correct += 1 if predicted == target else 0

    print("Test {}".format(i))
    print('Accuracy of the confidence network on the test data: %d %%' % (
            100 * correct / total))

    # ----------------------------------------------------------------------
    # Plot Confidence over Samples
    # ----------------------------------------------------------------------
    figconf = plt.figure()
    samples = np.linspace(0, len(output_collection), len(output_collection))
    plt.plot(samples, output_collection)
    plt.xlabel("Samples")
    plt.ylabel("Confidence")
    figconf.savefig("../Results/Confidence_{}".format(i), dpi=300)


def test_nn(test_data, model, i):
    correct = 0
    total = 0
    with torch.no_grad():
        for (data, target) in test_data:
            outputs = model(data)

            predicted = 1 if outputs.data >= 0.5 else 0
            total += target.size(0)
            correct += 1 if predicted == target else 0
            # _, predicted = torch.max(outputs.data, 1)
            # total += target.size(0)
            # correct += (predicted == target).sum().item()

    print("Test {}".format(i))
    print('Accuracy of the network on the test data: %d %%' % (
            100 * correct / total))


def confidence(test_data, model, svm, i):
    distance = []
    for batch, (data, target) in enumerate(test_data):
        outputs_nn = model(data)
        nn_value = outputs_nn.item()
        output_svm = svm.predict(data)[0]
        # ----------------------------------------------------------------------
        ##### Distance
        # ----------------------------------------------------------------------
        distance.append(abs(output_svm - nn_value))

    return distance


def create_labelvec(target):
    labelvec = {1: 0, 0: 0}
    if target == 1:
        labelvec[1] = 1
    else:
        labelvec[0] = 1
    return labelvec
