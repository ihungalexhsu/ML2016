# coding=utf-8
import numpy as np

# parse test data
dataset = np.loadtxt('./data/test_X.csv',dtype='string',skiprows=0,delimiter=',')
dataset = np.delete(dataset,(0,1),1)
test_num = 240
feature_num = 18
data = []
for m in range(test_num):
    for h in range (9):
        if(dataset[10,h] == 'NR'):
            dataset[10,h] = -1
    temp = dataset[0:feature_num,0:9].astype(np.float)
    answer = np.append(answer,dataset[9,9].astype(np.float))
    dataset = np.delete(dataset,np.arange(feature_num),0)
    data.append(temp)
#print np.asarray(data).shape
np.save('test_data',np.asarray(data))


