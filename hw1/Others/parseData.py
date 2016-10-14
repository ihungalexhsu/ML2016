# coding=utf-8
import numpy as np

# parse test data
dataset = np.loadtxt('./data/test_X_total.csv',dtype='string',skiprows=0,delimiter=',')
dataset = np.delete(dataset,(0,1),1)
test_num = 240
feature_num = 18
data = []
answer = np.empty((1,0))
for m in range(test_num):
    for h in range (9):
        if(dataset[10,h] == 'NR'):
            dataset[10,h] = -1
    temp = dataset[0:feature_num,0:9].astype(np.float)
    answer = np.append(answer,dataset[9,9].astype(np.float))
    dataset = np.delete(dataset,np.arange(feature_num),0)
    data.append(temp)
#print np.asarray(data).shape
print data[10][0,:]
print answer
np.save('test_data_answer',answer)
np.save('test_data',np.asarray(data))

'''
#parse training data
dataset = np.loadtxt('./data/train.csv',dtype='string',skiprows=1,delimiter=',')
#delete first three column
dataset = np.delete(dataset,(0,1,2),1)
feature_num = 18
days = 20
#data is modified data
months = 12
data = []

#start classifying
for m in range(months):
    month = np.empty((18,0))
    for d in range(days):
        #parse NR to -1    
        for h in range(24):
            if(dataset[10,h] == 'NR'):
                dataset[10,h] = -1
        month = np.append(month,dataset[0:feature_num,:].astype(np.float),1)
        dataset = np.delete(dataset,np.arange(feature_num),0)
    data.append(month)
#print np.asarray(data).shape
np.save('training_data',np.asarray(data))
'''

