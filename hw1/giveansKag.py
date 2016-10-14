import numpy as np
import csv
import random
import math

test_data = np.load('test_data.npy')
weight = np.load('weight_kaggle.npy')
bias = np.load('bias_kaggle.npy')
test_num = 240
answer=[]
with open('kaggle_best.csv','wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',quotechar='"')
    spamwriter.writerow(['id']+['value'])
    for i in range(test_num):
        inputdata = test_data[i].flatten()
        ans = math.ceil( (bias + float(np.dot(weight,inputdata))) *1)/1
        spamwriter.writerow(['id_'+str(i)]+[ans])
        answer.append(ans)

