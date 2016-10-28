import numpy as np
import random
import sys
import math

dataset = np.loadtxt(sys.argv[1],delimiter=',',usecols=range(1,59))

np.random.seed(3)
np.random.shuffle(dataset)

#slice to validation set 
valid_data = dataset[0:400,:]
validAns = valid_data[:,57].reshape((1,400))
valid_data = np.transpose(valid_data[:,0:57])
dataset = dataset[400:4001,:]
inputdata = dataset[:,0:57]
inputdata = np.transpose(inputdata)
dataAns = dataset[:,57].reshape((1,3601))

little_num = math.exp(-30)

#utility function for calcuating
def diff_cross_entropy(x,y):
    temp = np.zeros(x.shape)
    for ind,value in enumerate(y[0][:]):
        if value == 1 :
            if x[0][ind] < 0.0000001:
                temp[0][ind] = -1/math.exp(-8)
            else:
                temp[0][ind] = -1/x[0][ind]
        if value == 0 :
            if (1-x[0][ind]) < 0.0000001:
                temp[0][ind] = -1/math.exp(-8)
            else:
                temp[0][ind] = -1/(1-x[0][ind])
    return temp

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def diff_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def relu(x):
    return np.maximum(x,0)

def diff_relu(x):
    temp = np.zeros(x.shape)
    booltemp = x > temp
    booltemp = booltemp.astype(float)
    return booltemp

def estimate_relu(wei , bi, feature):
    temp = np.dot(wei,feature)
    temp = temp + bi
    temp = relu(temp)
    return temp

def estimate_sigmoid(wei,bi,feature):
    temp = np.dot(wei,feature)
    temp = temp + bi
    temp = sigmoid(temp+little_num)
    return temp

def cross_entropy(x,y):
    loss = 0 
    for ii in range (y.size):
        if (y[0,ii] == 0):
            loss = loss + math.log(1-x[0,ii]+little_num)
        if (y[0,ii] == 1):
            loss = loss + math.log(x[0,ii]+little_num)
    return (-1)*loss

num_hidden = 20
weight1 = np.random.uniform(-1,1,57*num_hidden).reshape(num_hidden,57)
bias1 = np.random.uniform(-1,1,num_hidden).reshape(num_hidden,1)
weight2 = np.random.uniform(-1,1,num_hidden).reshape(1,num_hidden)
bias2 = np.random.uniform(-1,1,1).reshape(1,1)

iteration = 18000
learning_rate = 0.000001

#adagrad
rate = 0.01
learn_rate_w1 = np.zeros((num_hidden,57))
learn_rate_w1.fill(rate)
denominator_w1 = np.ones((num_hidden,57))
learn_rate_b1 = np.zeros((num_hidden,1))
learn_rate_b1.fill(rate)
denominator_b1 = np.ones((num_hidden,1))
learn_rate_w2 = np.zeros((1,num_hidden))
learn_rate_w2.fill(rate)
denominator_w2 = np.ones((1,num_hidden))
learn_rate_b2 = np.zeros((1,1))
learn_rate_b2.fill(rate)
denominator_b2 = np.ones((1,1))

max_acc = 0
weight_max_1 = 0
weight_max_2 = 0
bias_max_1 = 0
bias_max_2 = 0

for i in range(iteration):
    print "iteration " + str(i)
    #layer1 = estimate_relu(weight1, bias1, inputdata)
    layer1 = estimate_sigmoid(weight1, bias1, inputdata)
    prediction = estimate_sigmoid(weight2, bias2, layer1)
    #loss = cross_entropy(prediction, dataAns)
    #updating
    #output layer
    #diff_unit2 = dataAns - prediction
    #diff_w2 = (-1)*np.dot(diff_unit2, np.transpose(layer1))
    #diff_b2 = (-1)*np.sum(diff_unit2)
    
    #sigmoid_diff2 = prediction - np.square(prediction )#1*3601
    #pratial_C = (-1)*(dataAns/prediction)+((1-dataAns)/(1-prediction))#1*3601
    #pratial_C = diff_cross_entropy(prediction, dataAns)
    #pratial_C = dataAns - prediction
    delta2 = dataAns - prediction
    #delta2 = sigmoid_diff2*pratial_C # 1*3601
    diff_w2 = np.dot( delta2 , np.transpose(layer1)) #1*10
    diff_b2 = (np.sum( delta2 )).reshape(1,1)

    #hidden layer
    #relu_diff1 = diff_relu(np.dot(weight1,inputdata)+bias1)
    sigmoid_diff1 = layer1 - np.square( layer1 )
    pratial_delta = np.dot(np.transpose(weight2), delta2)#10*3601
    delta1 = sigmoid_diff1*pratial_delta #num_hidden*3601
    diff_w1 = np.dot( delta1, np.transpose(inputdata) )
    diff_b1 = (np.sum( delta1,1)).reshape((num_hidden,1))
    
    denominator_w1 = np.sqrt(np.square(denominator_w1)+np.square(diff_w1))
    learning_rate_w1 =np.divide(learn_rate_w1,denominator_w1)
    denominator_b1 = np.sqrt(np.square(denominator_b1)+np.square(diff_b1))
    learning_rate_b1 =np.divide(learn_rate_b1,denominator_b1)

    denominator_w2 = np.sqrt(np.square(denominator_w2)+np.square(diff_w2))
    learning_rate_w2 =np.divide(learn_rate_w2,denominator_w2)
    denominator_b2 = np.sqrt(np.square(denominator_b2)+np.square(diff_b2))
    learning_rate_b2 =np.divide(learn_rate_b2,denominator_b2)
    
    weight1 = weight1 + learning_rate_w1*diff_w1
    bias1 = bias1 + learning_rate_b1*diff_b1
    weight2 = weight2 + learning_rate_w2*diff_w2
    bias2 = bias2 + learning_rate_b2*diff_b2
    #validation
    #layer1_val = estimate_relu(weight1, bias1, valid_data)
    layer1_val = estimate_sigmoid(weight1, bias1, valid_data)
    predict_val = estimate_sigmoid(weight2, bias2, layer1_val)
    #loss_val = cross_entropy(predict_val, validAns)
    predict_val = predict_val - 0.5
    temp1 = np.zeros(predict_val.shape)
    temp2 = predict_val > temp1
    temp2 = temp2.astype(int)
    temp3 = temp2 - validAns
    acc = 1-(np.sum(np.square(temp3))/400)

    if ( max_acc < acc):
        max_acc = acc
        weight_max_1 = weight1
        weight_max_2 = weight2
        bias_max_1 = bias1
        bias_max_2 = bias2
    print "acc : " +str(acc)

model = np.array([weight_max_1,bias_max_1,weight_max_2,bias_max_2])
np.save(sys.argv[2],model)


