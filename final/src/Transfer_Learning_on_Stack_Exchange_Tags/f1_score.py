from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
import pandas as pd
from numpy import genfromtxt
import numpy as np
import sys


ans_truth = pd.read_csv(sys.argv[1], quotechar='"', skipinitialspace=True).as_matrix()
# ans_truth = ans_truth[['id', 'tags']]

ans_test = pd.read_csv(sys.argv[2], quotechar='"', skipinitialspace=True).as_matrix()
# ans_test = ans_test[['id', 'tags']]

length = len(ans_test)
corpus = np.concatenate((ans_test[:, 1], ans_truth[:, 1]), axis=0).reshape(2, length).T

scores = []
vect = CountVectorizer()
for i in range(length):
	entry = corpus[i]
	feature = vect.fit_transform( entry ).toarray()
	scores.append( f1_score(feature[0], feature[1], average='weighted') )
scores = np.array(scores)
print( scores.mean() )
