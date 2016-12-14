from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
import pandas as pd
from numpy import genfromtxt
import numpy as np
import sys


def analysisAns(ans_truth, ans_test, top):
	results = []
	for i in range(len(ans_truth)):
		mask = np.in1d(ans_test[i].split(), ans_truth[i].split())
		mask = np.lib.pad(mask, (0, top-len(mask)), 'constant', constant_values=(0, 0))
		results.append(mask)
		# print(i)
	return results

def analysisStopwords(ans_truth, stopwords, top):
	# results = []
	mask = np.in1d(ans_truth.split(), stopwords[i].split())
	# mask = np.lib.pad(mask, (0, top-len(mask)), 'constant', constant_values=(0, 0))
	# results.append(mask)
		# print(i)
	return mask


ans_truth = pd.read_csv(sys.argv[1], quotechar='"', skipinitialspace=True).as_matrix()
# ans_truth = ans_truth[['id', 'tags']]

ans_test = pd.read_csv(sys.argv[2], quotechar='"', skipinitialspace=True).as_matrix()
ans_test.astype(str)
	
# ans_test[:,1] = ['' if element in ans_test[:,1] == np.isnan()]
# ans_test = np.nan_to_num(ans_test)
# x = x[np.logical_not(np.isnan(x))]
# ans_test = ans_test[~(np.isnan(ans_test))]
# ans_test = ans_test[['id', 'tags']]

length = len(ans_test)
corpus = np.concatenate((ans_test[:, 1], ans_truth[:, 1]), axis=0).reshape(2, length).T
nan_position = pd.isnull(corpus)
corpus[ nan_position == True ] = ''

scores = []
features = []
vect = CountVectorizer()
for i in range(length):
	entry = corpus[i]
	feature = vect.fit_transform( entry ).toarray()
	# print(feature)
	scores.append( f1_score(feature[0], feature[1], average='weighted') )
	# features.append(feature)
scores = np.array(scores)
result = scores.mean()
print( result )
with open("result.txt", "a") as myfile:
    myfile.write(str(result)+" ")

top = 10
'''
results = analysisAns(ans_truth[:,1], ans_test[:,1], top)
results = np.array(results).astype(int)
print(results.mean(axis=0))
'''
