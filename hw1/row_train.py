import numpy as np
import random

data = np.load('training_data.npy')

def loss_unit(loss):
    ans = np.square(loss)
    return np.sum(ans)

def diff_Wunit(loss, data):
    xT = np.transpose(data)
    return (-2)*(np.dot(loss,xT))

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
iteration = 100000

#arrange input data
inputdata = []
for m in range (12):
    drawer = np.empty((162,0))
    for d in range(471):
        temp = data[m][0:18,d:d+9].reshape(162,1)
        drawer = np.append(drawer,temp,1)
    inputdata.append(drawer)

for i in range(iteration):
    print "iteration  " + str(i)
    loss = 0
    diff_w = 0
    diff_b = 0
    for m in range(12):
        bi = np.zeros((1,471))
        bi.fill(bias)
        lo = data[m][9,9:480].reshape(1,471) - bi - np.dot(weight,inputdata[m])
        loss = loss+loss_unit(lo)
        diff_w = diff_w+diff_Wunit(lo,inputdata[m])
        diff_b = diff_b+diff_Bunit(lo)
    weight = weight - learning_rate*diff_w
    bias = bias - learning_rate*diff_b
    print loss

np.save("weight",weight)
np.save("bias",bias)
