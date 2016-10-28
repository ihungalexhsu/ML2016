import numpy as np
import random
import math
import sys

dataset = np.loadtxt(sys.argv[1],dtype='float',delimiter=',',usecols=range(1,59))

np.random.seed(1)
np.random.shuffle(dataset)

#slice to validation set
valid_data = dataset[0:400,:]
validAns = valid_data[:,57].reshape((1,400))
valid_data = np.transpose(valid_data[:,0:57])

dataset = dataset[400:4001,:]

weight = np.zeros((1,57))
#weight = np.random.uniform(-0.001,0.001,(1,162)) 
bias = 0 
#bias = random.uniform(-0.1,0.1)

little_num = math.exp(-30)
#utility function for calcuating
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def estimate(wei, bi, feature):
    temp = np.dot(wei, feature)
    temp = temp + bi
    temp = sigmoid(temp)
    return temp

def cross_entropy(x,y):
    loss = 0
    for ii in range(y.size):
        if (y[0,ii] == 0):
            loss = loss + math.log(1-x[0,ii]+little_num)
        if (y[0,ii] == 1):
            loss = loss + math.log(x[0,ii]+little_num)
    return (-1)*loss


iteration = 40000
lamda = 10

#learning_r = 0.000000000058

#adagrad
rate = 0.01 #adagrad rate
learn_rate = np.zeros((1,57)) #learn_rate is the matrix full with constant learning rate
learn_rate.fill(rate) #learn_rate init
learn_rate_b = rate #learn_rate_b is the constant learning rate for bias
denominator = np.zeros((1,57)) #the cumulative gradient
denominator_b = 0.0 #cumulative gradient for bias

learning_rate = np.zeros((1,57)) # temp vessel
learning_rate_b = 0.0 #temp vessel for bias

##########################################################
##------------Starting Implementataion------------------##
##########################################################

#arrange input data
inputdata = dataset[:,0:57]
inputdata = np.transpose(inputdata) #shape of inputdata = 57*3701
dataAns = dataset[:,57].reshape((1,3601)) # shape of dataAns = 1*3701
great = 0
weight_great = 0
bias_great = 0
#updating
for i in range(iteration):
    print "iteration  " + str(i)
    prediction =  estimate(weight,bias,inputdata)
    loss = cross_entropy(prediction, dataAns)
    diff_unit = dataAns - prediction
    diff_w = (-1)*np.dot( diff_unit, np.transpose(inputdata))
    diff_b = (-1)*np.sum(diff_unit)    
    
    #regularization
    loss =loss+lamda*(np.sum(np.square(weight)))
    diff_w =diff_w + 2*lamda*(weight)
    
    #adagrad
    denominator = np.sqrt(np.square(denominator)+np.square(diff_w))
    learning_rate = np.divide(learn_rate,denominator)
    denominator_b = math.sqrt((denominator_b**2)+(diff_b**2))
    learning_rate_b = learn_rate_b / denominator_b
    weight = weight - learning_rate*diff_w
    bias = bias - learning_rate_b*diff_b
	
    #validation
    predict_valid =  estimate(weight,bias,valid_data)
    loss_valid = cross_entropy(predict_valid, validAns)
    answer=[]
    for jj in range(400):
        ans = 0
        if(predict_valid[0,jj] > 0.5):
            ans = 1
        answer.append(ans)
    answer = np.array(answer)
    temp = answer-validAns
    acc = 1-(np.sum(np.square(temp))/400)
    #weight = weight - learning_r*diff_w
    #bias = bias - learning_r*diff_b
    if (great < acc) :
        great = acc
        weight_great = weight
        bias_great = bias
    print "training loss : " + str(loss/3601)
    print "valid loss : " + str(loss_valid/400)
    print "accuracy : "+ str(acc)

model = [weight_great, bias_great]
np.save(sys.argv[2],model)
#np.save("denominator_reg_valid",denominator)
#np.save("denominator_b_reg_valid",denominator_b)
