import numpy as np
import csv
import random
import math

def sigmoid(x):
    return 1.0/(1.0+math.exp(-x))

def estimate(wei, bi, feature):
    #weight as matrix; bias is value ; feature as matrix
    temp = np.dot(wei, feature)
    temp = temp + bi
    for ii in range(temp.size):
        temp[0,ii] = sigmoid(temp[0,ii])
    return temp

weight = np.load('weight_reg_valid_test3.npy')
bias = np.load('bias_reg_valid_test3.npy')
test_data = np.loadtxt('./spam_data/spam_test.csv',dtype='float',delimiter=',',usecols=(range(1,58)))
test_data = np.transpose(test_data)
prediction =  estimate(weight,bias,test_data)

test_num = 600
answer=[]
with open('ans_reg_valid_test2.csv','wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',quotechar='"')
    spamwriter.writerow(['id']+['label'])
    for i in range(test_num):
        ans = 0
        if (prediction[0,i] > 0.5) :
            ans = 1
        spamwriter.writerow([i+1]+[ans])
        answer.append(ans)
