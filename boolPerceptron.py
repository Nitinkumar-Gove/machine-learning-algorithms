# @author : nitinkumargove
# @date : 03/03/2018
# implementation of perceptron learning algorithm
from numpy import random, dot, array
from random import choice

class Perceptron:
    eta = 0.2 # controls the learning rate
    iterations = 200 # default no of learning iterations
    weights = random.rand(3) # initial random weights
    errors = [] #  store errors during learning
    training_dataset = []

    # set no of learning interations
    def setIterations(self, iter):
        self.iterations = iter

    # get iterations
    def getIterations(self):
        return self.iterations

    # initialize training given_dataset
    def setTrainingDataset(self, td):
        self.training_dataset = td

    # get training Dataset
    def getTrainingDataset(self):
        if not self.training_dataset:
            print('Initialize training dataset')
            return
        return self.training_dataset

    # step function
    def unitStep(self, x):
        return 0 if x < 0 else 1

    # train perceptron with given dataset
    def trainPerceptron(self):
        # check if the training dataset is initialized
        if not self.training_dataset:
            print('Initialize training dataset')
            return

        # predict for n iterations and adjust weights as per errors
        for step in xrange(self.iterations):
            data, expected = choice(self.training_dataset)
            result = dot(self.weights, data)
            error = expected - self.unitStep(result) # calculate error
            self.errors.append(error)
            self.weights += self.eta * error * data # adjust weights

    # test model for given input
    def testPerceptron(self, input):
        # check if the training dataset is initialized
        if not self.training_dataset:
            print('Initialize training dataset')
            return

        result = dot(input, self.weights)
        return self.unitStep(result)

    # print weights
    def printWeights(self):
        print(self.weights)

    # get weights
    def getWeights(self):
        return self.weights

    # def errors
    def printErrors(self):
        print(self.errors)

p = Perceptron()

# OR FUNCTION Data Table
training_dataset_OR = [ (array([0,0,1]),0), (array([0,1,1]),1), (array([1,0,1]),1), (array([1,1,1]),1)] #training dataset

# AND FUNCTION Data Table
training_dataset_AND = [ (array([0,0,1]),0), (array([0,1,1]),0), (array([1,0,1]),0), (array([1,1,1]),1)] #training dataset

# train - test PLA.
p.setTrainingDataset(training_dataset_OR)
p.setIterations(100)
p.trainPerceptron()

# test input
inp = array([0,1,1])
print("\n\n Dataset - {} \n\n No. of iterations - {} \n\n Weights - {} \n\n Input - {} \n\n Output - {}".format(p.getTrainingDataset(), p.getIterations(), p.getWeights(), inp[:2], p.testPerceptron(inp)))

'''
 *** Program Output ***

 Dataset - [(array([0, 0, 1]), 0), (array([0, 1, 1]), 1), (array([1, 0, 1]), 1), (array([1, 1, 1]), 1)]

 No. of iterations - 100

 Weights - [ 0.8640843   0.6966609  -0.16174062]

 Input - [0 1]

 Output - 1
'''
