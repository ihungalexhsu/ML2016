import numpy as np
import random

data = np.load('training_data.npy')

def loss_unit(loss):
    #ans: 471*1
    #data: 471*162
    #w: 1*162
    #b: 471*1
    ans = np.square(loss)
    return np.sum(ans)

def diff_Wunit(loss, data):
    ans = (-2)*loss
    ans = ans*data
    return np.sum(ans, axis=0)

def diff_Bunit(loss):
    ans = (-2)*loss
    return np.sum(ans)

#weight = np.random.rand(1,162)/1000000000
#weight = np.zeros((1,162))
weight = np.load('weight.npy')
#bias = random.randint(0,5)/100000000
#bias = 0
bias = np.load('bias.npy')
learning_rate = 0.0000000001
iteration = 1000


for m in range(12):
    for d in range(471):
        temp = data[m][0:18,d:d+9].reshape(162,1)
        inputdata = np.append(inputdata,temp,1)

for i in range(iteration):
    print "iteration  " + str(i)
    loss = 0
    diff_w = 0
    diff_b = 0
    for m in range(12):
        inputdata = np.empty((162,0))
        bi = np.zeros((471,1))
        bi.fill(bias)
        for d in range(471):
            temp = data[m][0:18,d:d+9].reshape(162,1)
            inputdata = np.append(inputdata,temp,1)
        temp = np.dot(weight,inputdata)
        ans = np.transpose(data[m][9,9:480].reshape(1,471))
        lo = ans - bi - np.transpose(temp)
        loss = loss+loss_unit(lo)
        diff_w = diff_w+diff_Wunit(lo,np.transpose(inputdata))
        diff_b = diff_b+diff_Bunit(lo)
    weight = weight - learning_rate*diff_w
    bias = bias -learning_rate*diff_b
    print loss

np.save("weight",weight)
np.save("bias",bias)
