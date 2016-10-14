import numpy as np
import random
import math

data = np.load('training_data.npy')

#weight = np.zeros((1,162))
weight = (np.random.rand(1,162))/100.0 

#bias = 0 
bias = random.uniform(0,0.5)

#utility function for calcuating
def loss_unit(loss):
    ans = np.square(loss)
    return np.sum(ans)

def diff_Wunit(loss, data):
    xT = np.transpose(data)
    return (-2)*(np.dot(loss,xT))

def diff_Bunit(loss):
    ans = (-2)*loss
    return np.sum(ans)

learning_r = 0.00000000005
#adagrad
rate = 0.01
learn_rate = np.zeros((1,162))
learn_rate_b = rate
learn_rate.fill(rate)
denominator = np.zeros((1,162))
denominator_b = 0.0
learning_rate = np.zeros((1,162))
learning_rate_b = 0.0


iteration = 450000
test_num = 240
lamda = 0

#arrange input data
inputdata = []
for m in range (12):
    drawer = np.empty((162,0))
    for d in range(471):
        temp = data[m][0:18,d:d+9].reshape(162,1)
        drawer = np.append(drawer,temp,1)
    inputdata.append(drawer)

#updating
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
    '''
    #regularization
    loss =loss+lamda*(np.sum(np.square(weight)))
    diff_w =diff_w + 2*lamda*(weight)
    '''
    if (i==250000):
        learning_r = learning_r / 5.0
    if (i==350000):
        learning_r = learning_r / 5.0
    if (i<72000):
        #adagrad
        denominator = np.sqrt(np.square(denominator)+np.square(diff_w))
        learning_rate = np.divide(learn_rate,denominator)
        denominator_b = math.sqrt((denominator_b**2)+(diff_b**2))
        learning_rate = np.divide(learn_rate,denominator)
        learning_rate_b = learn_rate_b / denominator_b
        weight = weight - learning_rate*diff_w
        bias = bias - learning_rate_b*diff_b
    else:
        weight = weight - learning_r*diff_w
        bias = bias - learning_r*diff_b
    print "training loss : " + str(math.sqrt(loss/5652) )


np.save("weight_kaggle",weight)
np.save("bias_kaggle",bias)
