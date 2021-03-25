import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import NNModel
import torch

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]

        return X, y

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



def main():
    print('Start Coding, Have fun!')
    datasets = loadData()
    #dataPlot(datasets)
    for i in ["a", "b", "c"]:

        input = datasets[i][["x_i1", "x_i2"]]
        label = datasets[i]['l_i']

        ###### SVM Training
        input_train, input_test, label_train, label_test = train_test_split(input, label, test_size=0.10)
        svclassifier = SVC(kernel='rbf')
        svclassifier.fit(input_train, label_train)

        label_pred = svclassifier.predict(input_test)

        print("SVM results  for Dataset {}".format(i))
        print(confusion_matrix(label_test, label_pred))
        print(classification_report(label_test, label_pred))

        ######### Neural Network
        #input = torch.tensor(input.values)
        #label = torch.tensor(label)
        trainset = Dataset(input, label)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                    shuffle=True, num_workers=2)

        model = NNModel.Net()
        loss_fn = torch.nn.MSELoss(reduction='sum')
        learning_rate = 1e-3
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)


        for epoch in range(0,10):
            running_loss = 0.0
            index = 0
            for k in range(0, len(trainloader.dataset)):
            #for k, data in enumerate(trainloader, 0):
                #inputs, labels = data
                inputs = torch.tensor(trainloader.dataset.list_IDs.iloc[k].astype('float32'))
                labels = torch.tensor(trainloader.dataset.labels.iloc[k])
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                if index % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, index + 1, running_loss / 2000))
                    running_loss = 0.0

            index = index + 1

        print('Finished Training')











if __name__ =="__main__":
    main()