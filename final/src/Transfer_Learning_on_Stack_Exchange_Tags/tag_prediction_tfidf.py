
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
import bottleneck as bn # sorting
import re

def saveResults(outfileName, id_, result):
	ofile = open(outfileName, "w")
	ofile.write('\"id\",\"tags\"\n')
	for i in range(len(id_)):
		ofile.write( '"' + str(id_[i]) + '"' + "," + '"' + str(" ".join(result[i])) + '"' + '\n' )
	ofile.close()
	return True

def read_words(words_file):
    return [word for line in open(words_file, 'r') for word in line.split()]

path = sys.argv[1]
outfileName = sys.argv[2]

# read from file
# fileName = "biology.csv"
# fileName = "test.csv"

origin_data = pd.read_csv( path, quotechar='"', skipinitialspace=True).as_matrix()

# process data
id_ = origin_data[:, 0]

title  = origin_data[:, 1]
content= origin_data[:, 2]
corpus = origin_data[:, 1:3]
corpus = corpus.astype(object)
corpus = corpus[:, 0] + " " + corpus[:, 1]
corpus = [re.sub(r'\d+', '', word) for word in corpus]

# tags   = origin_data[:, 3]

# define vector
my_words = read_words( "stop_words.txt")
my_stop_words = text.ENGLISH_STOP_WORDS.union(my_words)
vect = TfidfVectorizer(max_df=0.5, min_df=1, analyzer='word', 
   		use_idf=True, stop_words=my_stop_words)

# fit vector
features = vect.fit(corpus)
features_title = vect.transform(title).toarray()
features_content = vect.transform(content).toarray()
print("Finish feature extraction!")
print("Size title = ", features_title.shape, "size content = ", features_content.shape)
# print out the entries with highest weight
feature_arr = []
n_top = int(5)
weights = np.array( vect.idf_ )
featureName = np.array( vect.get_feature_names() )
features_weighted = 2*features_title + features_content

# generate output
print("Start to generate output!")
partion = int(len(corpus)/10)
count = 0
for i in range(len(corpus)):
	selectedFeature = features_weighted[i]
	arg = selectedFeature.argsort()[-1*n_top:][::-1]
	# arg = bottleneck.argpartsort(-a, 10)[:10]
	feature_arr.append( featureName[arg] )
	if i%partion == 0:
		print("Yep: ", count, "/10")
		count = count + 1
print("Finish generating output!")
# save to files
saveResults(outfileName, id_, feature_arr)
print("Finish save to file!")

'''
for i in range(5):
	selectedFeature = features_weighted[i]
	arg = selectedFeature.argsort()[-1*n_top:][::-1]
	selected = np.concatenate( (featureName[arg], weights[arg]), axis=0)
	selected = selected.reshape((2,n_top)).T
	print(selected)
	feature_arr.append(selected)
'''

