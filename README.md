# Siemens-Project

Members: 
* Bker Sawalha 
* Allaa El-Khodr
* Lucas Kneffel Otal


### AI safety application

The aim is to develop an algorithm to increase the confidence of classification results. Especially when it comes to safety critical situations, we would like to have clearly results. Also look at the picture with the data, we don't want that if a missclassifaction occurs a blue one is classified as a red one, because this is a safety critical situation. The oppiste way is allowed cause it's not safety critical.


![Data a](https://github.com/lucasKO2810/Siemens-Project/blob/main/Dataset/data_a.png?raw=true)


### Approach

We try to improve the result of an simple neural network by adding the additional information from a support vector machine. What we mainly do is to use the result of the firtst nn and the svm result to compute a classification result by an other neural network, called confidence network. In addition it uses a modified BCELoss to make sure a blue is not wrongly classified as a red one.

