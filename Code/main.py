import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd


def loadData():

    datasets = {
        "a": [],
        "b": [],
        "c": []
    }

    for i in ["a", "b", "c"]:
        datasets[i] = pd.read_excel("../Dataset/trainingdata_{}.xls".format(i))
        print("Dataset {}".format(i))
        print(datasets[i])

    return datasets

def main():
    print('Start Coding, Have fun!')
    datasets = loadData()









if __name__ =="__main__":
    main()