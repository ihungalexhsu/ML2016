import numpy as np
import csv
import random
import math
import sys

little_num = math.exp(-30)
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def estimate_sigmoid(wei,bi,feature):
    temp = np.dot(wei,feature)
    temp = temp+bi
    temp = sigmoid(temp+little_num)
    return temp


model = np.load(sys.argv[1])
weight1 = model[0]
bias1 = model[1]
weight2 = model[2]
bias2 = model[3]
test_data = np.loadtxt(sys.argv[2],delimiter=',',usecols=(range(1,58)))
test_data = np.transpose(test_data)
layer1 = estimate_sigmoid(weight1, bias1, test_data)
predict =estimate_sigmoid(weight2, bias2, layer1)
predict = predict - 0.5
temp1 = np.zeros(predict.shape)
temp2 = predict > temp1
ans = temp2.astype(int)

test_num = 600
with open(sys.argv[3],'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter = ',',quotechar='"')
    spamwriter.writerow(['id']+['label'])
    for i in range(test_num):
        spamwriter.writerow([i+1]+[ans[0][i]])
