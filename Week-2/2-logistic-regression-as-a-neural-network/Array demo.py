# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 20:33:17 2025

@author: ali
"""

import numpy as np

# a = np.array([[[[1,1,1],[2,2,2]]]])
# print(a)
# print(a.shape)
# a.reshape(a.shape[0],-1)
# a.reshape(a.shape[0],-1).T

# a = np.array([[[[1,1,1],[2,2,2]],[[1,1,1],[2,2,2]]]])
# print(a)
# print(a.shape)


# a = np.array([
#     [
#         [[1,1,1],[2,2,2]],
#         [[3,3,3],[4,4,4]]
#     ],
#     [
#         [[1,1,1],[2,2,2]],
#         [[3,3,3],[4,4,4]]
#     ]
# ])


a = np.array([[[[1,1,1],[2,2,2]],[[3,3,3],[4,4,4]]],[[[1,1,1],[2,2,2]],[[3,3,3],[4,4,4]]]])
print(a)
#print(a.shape)
a_flat=a.reshape(a.shape[0],-1)#.T
print(a_flat)
a_flat=a_flat.T
print(a_flat)


# 1. Create random dataset
m = 50                           # number of samples
X_org = np.random.randint(0, 256, size=(m, 2, 2, 3))  # shape (50,2,2,3), values 0–255

# 2. Create random binary labels
y = np.random.randint(0, 2, size=(1, m))          # shape (50,1)

num_px = X_org.shape[1]


X_flatten=X_org.reshape(X_org.shape[0],-1).T

X=X_flatten/255

def sigmoid(z):
    return 1/(1+np.exp(-z))

def initialize_with_zero(dim):
    w=np.zeros([dim,1])
    b=0
    assert(w.shape == (dim,1))
    assert(isinstance(b,float) or isinstance(b,int))
    return w,b

w,b=initialize_with_zero(num_px*num_px*3)


def propagate(w,b,X,Y):
    m=X.shape[1]
    A=sigmoid(np.dot(w.T,X)+b)
    cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m     # compute cost
    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

grads, cost = propagate(w, b, X, y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))


def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost=False):
    
    costs=[]
    
    for i in range(num_iterations):
        grads,cost=propagate(w, b, X, Y)
        
        dw=grads['dw']
        db=grads['db']
        
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        if i % 5 == 0:
            costs.append(cost)
            
        if print_cost and i % 5 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        
        
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs
    
    
params, grads, costs = optimize(w, b, X, y, num_iterations= 10, learning_rate = 0.009, print_cost = True)

print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))


def predict(w,b,X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        Y_prediction[0][i] = 1 if A[0][i] > 0.5 else 0
        
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

print ("predictions = " + str(predict(w, b, X)))