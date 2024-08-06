import numpy as np
import pandas as pd
from sklearn import preprocessing

def relu(x):
    '''
    Rectilinear unit function. Better than sigmoid.
    '''
    return np.maximum(0, x)

def reluprime(x):
    return (x > 0).astype(float)

class Network():
    def __init__(self, neurons):
        self.hidden_shape = tuple(neurons)
        self.networkshape = None
        self.network = []
        pass

    def build(self, shape):
        '''
        Function to build a neural network

        Inputs
        ------

        shape: array
            The desired shape of the network. e.g. A network of 3 layers with 2 input neurons, 5 hidden neurons, and 2 outputs would have shape [2, 5, 2]

        Outputs
        --------

        Network: list
            Returns a list of Layer objects with the sizes of each layer
        '''
        for i, size in enumerate(shape):
            j = i - 1
            if i == 0:
                self.network.append(Layer(size,size))
            else:
                self.network.append(Layer(size, shape[j]))
            pass
        return self.network

    def binariser(self, y):
        '''
        Function to convert labels into a binary output. Single bit labels are still converted into a 2 bit output.

        Inputs
        -------

        y: array
            The array of labels that you wish to be binarised

        Outputs
        --------

        binarised_labels: array
            An array that has converted all the labels into binary.
        '''
        y = np.array(y).squeeze()
        lb = preprocessing.LabelBinarizer()
        lb.fit(y)
        binarised_labels = lb.transform(y).squeeze()
        self.labels = lb.classes_
        if len(self.labels) == 2:
            binarised_labels = np.array([binarised_labels, 1 - binarised_labels]).T
        return binarised_labels

    def forwardprop(self, x):
        '''
        Propagating a network input x to the final layer

        Inputs
        -------

        x: array
            An array of the shape of the input layer

        Outputs
        --------

        Output: array
            The network output after propagating the data through all layers
        '''
        for layer in self.network:
            x = layer.forward(x)     # recursively assigns the input of the next loop as the output of the current loop
            pass
        return x

    def backwardprop(self, x):
        '''
        Backpropagating a network output error to find wieght and bias errors

        Inputs
        -------

        x: array
            The output error of the network

        Outputs
        --------

        network.a: array
            The output of the network used to begin back propagation
        '''
        for layer in reversed(self.network):
            x = layer.backward(x)
            pass
        return self.network[-1].a

    def evaluate(self, xtrain, ytrain):
        '''
        A function to asses how well the neural network is doing

        Inputs
        -------

        xtrain: array
            The data set to ingest and propagate through the network

        ytrain: array
            The binarised labels to test the network output against

        Outputs
        --------

        score: int
            The raw score of the network on the data set
        '''
        predictions = np.array([np.argmax(self.forwardprop(x)) for x in xtrain ])
        truths = np.array([np.argmax(y) for y in ytrain ])
        self.predictions = np.array([self.labels[i] for i in predictions])
        self.truths = np.array([self.labels[i] for i in truths])
        return np.sum([truths == predictions])

    def cost(self, Xtest, ytest):
        '''
        Finding the cost function of the network as a whole

        Inputs
        -------

        xtest: array
            The unseen data to propagate through the network

        ytest: array
            The binarised target to compare the network output to

        Outputs
        --------

        C: float
            The cost of training the network
        '''
        outputs = np.array([self.forwardprop(x) for x in Xtest]).squeeze()
        n = len(ytest)
        return np.sum((ytest - outputs) ** 2) / 2 / n

    def __repr__(self):
        return f'Neural network, shape = {self.networkshape}.'

    def SGD(self, xtrain, ytrain, xtest=[], ytest=None, epochs=10, eta=0.1, prntout=False):
        '''
        Stochastic Gradient Descent. A method by which the neural network is fed data and then retroactively adjusts itself to better predict data.

        Inputs
        ------

        xtrain: array
            Training data

        ytrain: array
            Training targets

        xtest: array, default=None
            Testing data

        ytest: array, default=None
            Associated testing targets

        epochs: int, default=40
            The number of training loops to execute

        eta: float
            The learning step. Highly recommended to keep at 0.1.

        prntout: bool, default=False
            Whether to print out training progress every 10 epochs.

        Outputs
        --------

        Score: str
            The final training score of the last training epoch.
        '''
        traintargets = self.binariser(ytrain)
        testtargets = self.binariser(ytest)
        networkshape = xtrain[0].shape + self.hidden_shape + np.unique(ytrain).shape
        self.networkshape = networkshape
        self.build(networkshape)

        self.trainscores = []
        self.testscores = []

        for j in range(epochs):
            for x,y in zip(xtrain, traintargets):
                z = self.forwardprop(x) - y
                self.error = z
                self.backwardprop(z)
                '''
                if i % (len(xtrain)//4) == 0:
                    print(f'error: {z}\nTarget: {y}\n\n')
                '''
                for layer in self.network:
                    layer.weights -= eta * layer.nabla_w
                    layer.biases -= eta * layer.nabla_b
                    pass
                pass

            #print(f'Output:{self.network[-1].a}\nTruth:{y}')
            self.trainscores.append(self.evaluate(xtrain, traintargets))
            if len(xtest) == 0:
                pass
            else:
                self.testscores.append(self.evaluate(xtest, testtargets))
                pass
            if prntout == True:
                if j % 10 == 0 or j+1 == epochs:
                    print(f'Epoch #{j}:\nTraining Perfomance: {self.trainscores[j]}/{len(ytrain)}\nTest Perfomance: {self.testscores[j]}/{len(ytest)}\n')
                    pass
                pass
            pass
        if len(xtest) == 0:
            return f'Training score of: {self.trainscores[-1]/len(ytrain)}'
        else:
            return f'Training score of: {self.trainscores[-1]/len(ytrain): .2f}, Test score of: {self.testscores[-1]/len(ytest): .2f}'
        pass
    pass


class Layer():
    def __init__(self, neurons, inputshape):
        self.inshape = inputshape
        weightshape = (neurons, inputshape)
        self.weights = np.random.normal(size=weightshape) * np.sqrt(2 / inputshape) #He initialisation for ReLu function
        self.z = None
        self.a = None
        self.biases =  np.random.normal(size=neurons) * np.sqrt(2 / inputshape)
        self.nabla_a = None
        self.nabla_b = None
        self.nabla_w = None
        pass

    def __repr__(self):
        return f'Layer of size {len(self.a)}'

    def forward(self, x):
        '''
        Forward propagation functions for a network layer.

        Inputs
        -------

        x: array
            An array of inputs to the layer. Must match inputshape.

        Outputs
        --------

        self.a: array
            The layer outputs.
        '''
        self.start = x
        self.z = self.weights @ x + self.biases
        self.a = relu(self.z)
        return self.a

    def backward(self, nabla_a):
        '''
        Backward propagation of error given an error in the layer ahead

        Inputs
        -------

        nabla_a: array
            The output error of this layer

        Outputs
        -------

        nabla_a: array
            The output error of the previous layer
        '''
        self.nabla_a = nabla_a
        self.nabla_b = nabla_a * reluprime(self.z)
        self.nabla_w = np.tensordot(self.nabla_b, self.start, axes=0)
        return self.weights.T@self.nabla_b
    pass