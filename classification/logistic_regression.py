from util import sigmoid
import numpy as np
import random

class logistic_regression:
    def __init__(self, data, label, n_iter, eta):
        self.n_iter = n_iter
        self.eta = eta
        self.data = data
        self.label = label
        self.n_feature = len(data[0])
        # inteception
        for d in self.data:
            d = np.append(d, 1.0)

    def train(self):
        self.W = np.random.rand(self.n_feature + 1)
        for iter in range(self.n_iter):
            g = [0.0 for i in range(len(self.W))]
            for index, d in enumerate(self.data):
                mu = sigmoid(self.W, d)
                g = [ x + y for x, y in zip([(mu - label[index]) * z for z in data[index]], g) ]
            self.W = [x - self.eta * y for x, y in zip(self.W, g)]

    def test(self, test_data, label):
        n_pos = 0.0
        for i in range(len(test_data)):
            if sigmoid(self.W, test_data[i]) >= 0.5 and label[i] == 1:
                n_pos += 1
            elif sigmoid(self.W, test_data[i]) < 0.5 and label[i] == 0:
                n_pos += 1
        
        print n_pos, len(test_data), n_pos / len(test_data)
        print self.W
        
if __name__=='__main__':
    label = np.array([])
    cov = [[0.1, 0], [0, 0.1]]
    num = 2000
    data1 = np.random.multivariate_normal([1, 2], cov, num / 2)
    data2 = np.random.multivariate_normal([2, 1], cov, num / 2)
    data = np.vstack((data1, data2))
    for i in range(num / 2):
        label = np.append(label, 1)
    for i in range(num / 2):
        label = np.append(label, 0)
    rate_train = 0.7
    data_train = np.zeros(shape=(int(num * rate_train), len(data[0])))
    data_test = np.zeros(shape=(int(num * (1 - rate_train)), len(data[0])))
    label_train = np.zeros(shape=(int(num * rate_train), 1))
    label_test = np.zeros(shape=(int(num * (1 - rate_train)), 1))
    random.seed()
    index = set(random.sample(range(num), int(num * rate_train)))
    index_train = 0
    index_test = 0
    for i in range(len(data)):
        if i in index:
            data_train[index_train] = data[i]
            label_train[index_train] = label[i]
            index_train += 1
        else:
            data_test[index_test] = data[i]
            label_test[index_test] = label[i]
            index_test += 1
    
    lr = logistic_regression(data_train, label_train, 50, 0.01)
    lr.train()
    lr.test(data_test, label_train)
    lr.test(data_test, label_test)
