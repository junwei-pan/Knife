import math

def sigmoid(W, X):
    res = 1 / (1 + math.exp(-1 * sum([w * x for w, x in zip(W, X)])))
    return res
