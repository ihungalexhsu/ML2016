from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from numpy import genfromtxt

import sys

ans_truth = genfromtxt( sys.argv[1], delimiter=',') 
ans_truth = ans_test[1:, 1:]

ans_test = genfromtxt( sys.argv[2], delimiter=',') 
ans_test = ans_test[1:, 1:]
