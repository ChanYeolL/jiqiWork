import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

##Linear regression with multiple variables

path = 'data/housing_scale.txt'
X,y = load_svmlight_file(path)
print(X,'  ',y,'\n X.shape[0]:',X.shape)

X_train, X_dev, y_train, y_dev = train_test_split(X, y,test_size=0.3, random_state = 20, shuffle=True)

theta = np.random.randn(X.shape[1])

def costFunction(X, y, theta):
    inner = np.power(X * theta.T - y, 2)
    m = X.shape[0]
    return np.sum(inner)/(2*m)

def gradientDescent(X, y, theta, alpha, epoch):

    cost = np.zeros(epoch)
    m = X.shape[0]

    for i in range(epoch):
        term = theta - (alpha/m) * (X*theta.T  - y).T * X
        theta = term
        cost[i] = costFunction(X, y, theta)

    return theta, cost

alpha = 0.01
epoch = 1500

final_theta, cost = gradientDescent(X, y, theta, alpha, epoch)

print(costFunction(X, y, theta), ' \n ', final_theta)

plt.plot(cost,final_theta,linewidth=5)
plt.show()