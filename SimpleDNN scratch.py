import time
import numpy as np
import scipy.special


class SimpleDNN:
    """
    This class gives an insight of math employed behind a simple neural network with 1 input layer, 1 hidden layer
    and 1 output layer. Here we will use this class for classifying our target objective into 2 classes,
    i.e. a binary-classifier with binary cross entropy as the loss function.
    """
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        """
        Initializing all the hyper parameters and weights and biases

        :param input_nodes: This number corresponds to the number of features each input has
        :param hidden_nodes: Number of hidden nodes
        :param output_nodes: Number of output nodes
        """
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.activation_sigmoid = lambda x: scipy.special.expit(x)  # sigmoid activation function
        self.activation_relu = lambda x: np.maximum(0, x)  # ReLu activation function
        self.log = {'loss': [], 'accuracy': []}  # keeping a track of loss and acc to avoid potential threat of overfitting

        self.wih = np.random.randn(self.inodes, self.hnodes) * 0.1
        self.bh = np.zeros((self.hnodes,))

        self.who = np.random.randn(self.hnodes, self.onodes) * 0.1
        self.bo = np.zeros((self.onodes,))

    def forward_propagation(self, inputs):
        """
        Forward passing the features in a neural network
        :param inputs: data to fit in a neural network
        """

        inputs = inputs.reshape(-1, self.inodes)
        self.inputs = inputs

        self.hidden_inp = inputs.dot(self.wih) + self.bh  #  y = mx + c
        self.hidden_out = self.activation_relu(self.hidden_inp)

        self.final_inp = self.hidden_out.dot(self.who) + self.bo #  y = mx + c
        self.final_out = self.activation_sigmoid(self.final_inp)

    @classmethod
    def ETA(cls, var, val=0.00001):
        """
        This function is used to clip the 0 values to a very small value to avoid occurrence of bad values in
        computation of log function and thereby increasing the performance of the neural network
        :param var: Array of values to clip 0s
        :param val: A float value to clip 0s with
        :return np.maximum(var, val): Clipped array where 0s are replaced with {val}
        """
        return np.maximum(var, val)

    def binary_cross_entropy(self, y_actual):
        """
        Calculates binary cross entropy a.k.a log-loss for every sample
        :param y_actual: Actual label values
        :return loss: return log-loss for each samples
        """
        n = len(y_actual)
        y_hat = SimpleDNN.ETA(self.final_out)
        y_hat_inv = 1.0 - y_hat
        y_hat_inv = SimpleDNN.ETA(y_hat_inv)
        y_actual_inv = 1.0 - y_actual

        loss = -1 / n * (np.sum(np.multiply(np.log(y_hat), y_actual) + np.multiply(np.log(y_hat_inv), y_actual_inv)))
        return loss

    def accuracy(self, y_actual):
        """
        Calculates accuracy values by comparing predicted and actual values
        :param y_actual: Actual label values
        :return acc_score: Accuracy score in decimal (multiplying this value by 100 will give the percentage accuracy)
        """
        preds = self.final_out.round()
        acc_score = np.sum(np.where(preds == y_actual, 1, 0)) / len(y_actual)
        return acc_score

    def backpropagation(self, y_actual):
        """

        :param y_actual: Actual label values
        :return:
        """

        y_actual_inv = 1.0 - y_actual
        y_hat = self.final_out
        y_hat_inv = 1.0 - self.final_out

        dl_wrt_a2 = -np.divide(y_actual, SimpleDNN.ETA(y_hat)) + np.divide(y_actual_inv, SimpleDNN.ETA(y_hat_inv))
        da2_wrt_z2 = y_hat * (1.0 - y_hat)
        dl_wrt_z2 = dl_wrt_a2 * da2_wrt_z2
        dz2_wrt_who = self.hidden_out
        dl_wrt_who = dz2_wrt_who.transpose().dot(dl_wrt_z2)
        dl_wrt_bo = dl_wrt_z2

        dl_wrt_a1 = dl_wrt_z2.dot(self.who.transpose())
        da1_wrt_z1 = np.where(self.hidden_out > 0, 1, 0)
        dl_wrt_z1 = dl_wrt_a1 * da1_wrt_z1
        dz1_wrt_wih = self.inputs
        dl_wrt_wih = dz1_wrt_wih.transpose().dot(dl_wrt_z1)
        dl_wrt_bh = dl_wrt_z1

        self.who = self.who - self.lr * dl_wrt_who
        self.bo = self.bo - self.lr * dl_wrt_bo

        self.wih = self.wih - self.lr * dl_wrt_wih
        self.bh = self.bh - self.lr * dl_wrt_bh

    def fit(self, X, y, learning_rate=0.001, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs

        for epoch in range(self.epochs):
            start = time.time()
            self.forward_propagation(X)
            loss = self.binary_cross_entropy(y)
            acc = self.accuracy(y)
            self.backpropagation(y)

            self.log['loss'].append(loss)
            self.log['accuracy'].append(acc)

            print(f'''Epoch {epoch + 1} / {self.epochs}\n[==============================] - {time.time() - start} - loss: {loss} - accuracy: {acc}''')

        return self.log

    def predict(self, y_actual):
        pass



import pandas as pd

# add header names
headers =  ['age', 'sex','chest_pain','resting_blood_pressure',
        'serum_cholestoral', 'fasting_blood_sugar', 'resting_ecg_results',
        'max_heart_rate_achieved', 'exercise_induced_angina', 'oldpeak',"slope of the peak",
        'num_of_major_vessels','thal', 'heart_disease']

heart_df = pd.read_csv('heart.csv', sep=' ', names=headers)
print(heart_df.head())


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#convert imput to numpy arrays
X = heart_df.drop(columns=['heart_disease'])

#replace target class with 0 and 1
#1 means "have heart disease" and 0 means "do not have heart disease"
heart_df['heart_disease'] = heart_df['heart_disease'].replace(1, 0)
heart_df['heart_disease'] = heart_df['heart_disease'].replace(2, 1)

y_label = heart_df['heart_disease'].values.reshape(X.shape[0], 1)

#split data into train and test set
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y_label, test_size=0.2, random_state=2)

#standardize the dataset
sc = StandardScaler()
sc.fit(Xtrain)
Xtrain = sc.transform(Xtrain)
Xtest = sc.transform(Xtest)

print(f"Shape of train set is {Xtrain.shape}")
print(f"Shape of test set is {Xtest.shape}")
print(f"Shape of train label is {ytrain.shape}")
print(f"Shape of test labels is {ytest.shape}")


nn = SimpleDNN(13, 8, 1)
history = nn.fit(Xtrain, ytrain)
