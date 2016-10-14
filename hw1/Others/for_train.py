import numpy as np
import random

data = np.load('training_data.npy')

def loss_unit(ans, data, w, b):
    #ans: scalar
    #data: 162*1
    #w: 1*162
    #b: scalar
    loss = ans - b - float(np.dot(w,data))
    #print loss
    return loss**2

def diff_Wunit(ans, data, w, b):
    return (-2*(ans-b-float(np.dot(w,data))))*data
def diff_Bunit(ans, data, w, b):
    return (-2)*(ans-b-float(np.dot(w,data)))

#weight = np.random.rand(1,162)/1000000000
#weight = np.zeros((1,162))
weight = np.load('weight.npy')
#bias = random.randint(0,5)/100000000
#bias = 0
bias = np.load('bias.npy')
learning_rate = 0.00000000003
iteration = 50000

for i in range(iteration):
    print "iteration  " + str(i)
    loss = 0
    diff_w = 0
    diff_b = 0
    for m in range(12):
        for d in range(471):
            inputdata = data[m][0:18,d:d+9]
            loss = loss+loss_unit(data[m][9,d+9],inputdata.flatten(),weight,bias)
            diff_w = diff_w+diff_Wunit(data[m][9,d+9],inputdata.flatten(),weight,bias)
            diff_b = diff_b+diff_Bunit(data[m][9,d+9],inputdata.flatten(),weight,bias)
    weight = weight - learning_rate*diff_w
    bias = bias -learning_rate*diff_b
    print loss

np.save("weight",weight)
np.save("bias",bias)
