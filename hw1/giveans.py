import numpy as np
import csv
import random

test_data = np.load('test_data.npy')
weight = np.load('weight.npy')
bias = np.load('bias.npy')
test_num = 240
answer=[]
with open('ans.csv','wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',quotechar='"')
    spamwriter.writerow(['id']+['value'])
    for i in range(test_num):
        inputdata = test_data[i].flatten()
        ans = bias + float(np.dot(weight,inputdata))
        spamwriter.writerow(['id_'+str(i)]+[ans])
        answer.append(ans)

