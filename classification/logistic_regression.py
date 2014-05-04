from util import sigmoid
import numpy as np

class logistic_regression:
    def __init__(self, data, label, n_iter, eta):
        self.n_iter = n_iter
        self.eta = eta
        self.data = data
        self.label = label
        self.n_feature = len(data[0])
        # inteception
        for d in self.data:
            d.append(1.0)

    def train(self):
        self.W = np.random.rand(self.n_feature + 1)
        for iter in range(self.n_iter):
            g = [0.0 for i in range(len(self.W))]
            for index, d in enumerate(self.data):
                mu = sigmoid(self.W, d)
                g = [ x + y for x, y in zip([(mu - label[index]) * z for z in data[index]], g) ]
            self.W = [x - self.eta * y for x, y in zip(self.W, g)]
        n_pos = 0.0
        for i in range(len(self.data)):
            if sigmoid(self.W, self.data[i]) >= 0.5 and label[i] == 1:
                n_pos += 1
            elif sigmoid(self.W, self.data[i]) < 0.5 and label[i] == 0:
                n_pos += 1
        print n_pos / len(self.data)
        print self.W
def test():
    W = [0.1, 0.2, 0.2]
    X = [0.3, 0.2, 0.1]
    print sigmoid(W, X)
    #print function.sigmoid(W, X)

if __name__=='__main__':
    #data = [[1, 3], [2, 3], [1, 4], [2, 4], [4, 2], [4, 1], [3, 2], [3, 1]]
    #label = [1, 1, 1, 1, 0, 0, 0, 0]
    data = []
    label = []
    cov = [[0.1, 0], [0, 0.1]]
    data1 = np.random.multivariate_normal([1, 2], cov, 1000)
    for d in data1:
        data.append([d[0], d[1]])
        label.append(1)
    data2 = np.random.multivariate_normal([2, 1], cov, 1000)
    for d in data2:
        data.append([d[0], d[1]])
        label.append(0)
    lr = logistic_regression(data, label, 50, 0.01)
    lr.train()
