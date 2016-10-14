import numpy as np
import random
import math

#load data and validation data
data = np.load('training_data.npy')
test_data = np.load('test_data.npy')
test_data_ans = np.load('test_data_answer.npy')

#weight = np.zeros((1,162))
weight = (np.random.rand(1,162))/100.0 
#weight = np.load("weight_reporthope.npy")
bias = random.uniform(0,0.5)
#bias = np.load("bias_reporthope.npy")

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


learning_r = 0.000000000055
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
#validation
min_loss = 100000000
min_weight = weight
min_bias = bias

#arrange input data
inputdata = []
for m in range (12):
    drawer = np.empty((162,0))
    for d in range(471):
        temp = data[m][0:18,d:d+9].reshape(162,1)
        drawer = np.append(drawer,temp,1)
    inputdata.append(drawer)

testdata = []
for i in range(test_num):
    temp = test_data[i].reshape(162,1)
    testdata.append(temp)

#updating
for i in range(iteration):
    print "iteration  " + str(i)
    loss = 0
    diff_w = 0
    diff_b = 0
    loss_val = 0
    for m in range(12):
        bi = np.zeros((1,471))
        bi.fill(bias)
        lo = data[m][9,9:480].reshape(1,471) - bi - np.dot(weight,inputdata[m])
        loss = loss+loss_unit(lo)
        diff_w = diff_w+diff_Wunit(lo,inputdata[m])
        #diff_w = 1*162
        diff_b = diff_b+diff_Bunit(lo)
    
    for jj in range(test_num):
        predict_ans = bias + float(np.dot(weight,testdata[jj]))
        loss_val = loss_val+(test_data_ans[jj]-predict_ans)**2
    
    
    #regularization
    loss =loss+lamda*(np.sum(np.square(weight)))
    diff_w =diff_w + 2*lamda*(weight)
    if (i==250000):
        learning_r = learning_r / 5.0
    if (i==350000):
        learning_r = learning_r / 5.0
    if (i < 73000):
        #adagrad
        print "ada"
        denominator = np.sqrt(np.square(denominator)+np.square(diff_w))
        learning_rate = np.divide(learn_rate,denominator)
        denominator_b = math.sqrt((denominator_b**2)+(diff_b**2))
        learning_rate = np.divide(learn_rate,denominator)
        learning_rate_b = learn_rate_b / denominator_b
        weight = weight - learning_rate*diff_w
        bias = bias - learning_rate_b*diff_b
    else:
        print "learning_r : "+str(learning_r)
        weight = weight - learning_r*diff_w
        bias = bias - learning_r*diff_b
    
    if (loss_val < min_loss):
        min_weight = weight
        min_bias = bias
        min_loss = loss_val
        print "update~~~"
    print "training loss : " + str(math.sqrt(loss/5652) )
    print "validation loss : " + str(math.sqrt(loss_val/test_num))

np.save("weight_reporthope1",min_weight)
np.save("bias_reporthope1",min_bias)
np.save("denominator_reporthope1",denominator)
np.save("denominator_b_reporthope1",denominator_b)
