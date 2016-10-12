import numpy as np
import random

data = np.load('training_data.npy')

def loss_unit(ans, data, w, b):
    #ans: 1*471
    #data: 162*471
    #w: 1*162
    #b: 1*471
    loss = ans - b - np.dot(w,data)
    loss = np.square(loss)
    return np.sum(loss)

def diff_Wunit(ans, data, w, b):
    ans = (-2)*(ans-b-np.dot(w,data))
    xT = np.transpose(data)
    return np.dot(ans,xT)

def diff_Bunit(ans, data, w, b):
    ans = (-2)*(ans-b-np.dot(w,data))
    return np.sum(ans)

#weight = np.random.rand(1,162)/1000000000
#weight = np.zeros((1,162))
weight = np.load('weight.npy')
#bias = random.randint(0,5)/100000000
#bias = 0
bias = np.load('bias.npy')
learning_rate = 0.0000000001
iteration = 1000

for i in range(iteration):
    print "iteration  " + str(i)
    loss = 0
    diff_w = 0
    diff_b = 0
    for m in range(12):
        inputdata = np.empty((162,0))
        bi = np.zeros((1,471))
        bi.fill(bias)
        for d in range(471):
            temp = data[m][0:18,d:d+9].reshape(162,1)
            inputdata = np.append(inputdata,temp,1)
        loss = loss+loss_unit(data[m][9,9:480],inputdata,weight,bi)
        diff_w = diff_w+diff_Wunit(data[m][9,9:480],inputdata,weight,bi)
        diff_b = diff_b+diff_Bunit(data[m][9,9:480],inputdata,weight,bi)
    weight = weight - learning_rate*diff_w
    bias = bias -learning_rate*diff_b
    print loss

np.save("weight",weight)
np.save("bias",bias)
