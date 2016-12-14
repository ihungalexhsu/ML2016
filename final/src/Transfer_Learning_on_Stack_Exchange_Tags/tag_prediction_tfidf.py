
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
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

def lsa(X, n_components=80):
    print("Performing dimensionality reduction using LSA")
    svd = TruncatedSVD(n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(X)

    return X, svd

def tagsThreshold(threshold, selectedFeature, n_top):
	num = 1
	print(selectedFeature)
	while selectedFeature[num-1]*threshold > selectedFeature[num]:
		num = num + 1
		if num == n_top: break
	return num

def getfeaturesWeighted(vect, title, content, start, end):
	features_title = vect.transform(title[start:end]).toarray()
	# features_title, svd = lsa(features_title, 80)
	features_content = vect.transform(content[start:end]).toarray()
	# features_content, svd = lsa(features_content, 80)
	features_weighted = 2*features_title + features_content
	return features_weighted

def getFeaturearr(feature_arr, corpus, features_weighted, featureName, addThres, threshold, n_top):
	for i in range(len(features_weighted)):
		# selectedFeature = svd.inverse_transform(features_weighted[i].reshape(1,-1))
		# print(selectedFeature.shape)
		selectedFeature = features_weighted[i]
		
		# arg = selectedFeature.argsort()[-1*n_top:][::-1]
		arg = selectedFeature.argsort()[-1*n_top:][::-1]
		if addThres == True:
			tops = tagsThreshold(threshold, selectedFeature[arg], n_top)
		else:
			tops = n_top
		# arg = arg[-1*n_top:]

		# arg = bottleneck.argpartsort(-a, 10)[:10]
		feature_arr.append( featureName[arg[:tops]] )
	return feature_arr

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
# generate output
features = vect.fit(corpus)
feature_arr = []
n_top = int(6)
weights = np.array( vect.idf_ )
featureName = np.array( vect.get_feature_names() )
addThres = False

print("Start to generate output!")
nb_partition = 10
partion = int(len(corpus)/nb_partition)
# threshold = 0.8
threshold = sys.argv[3]
count = 0
for i in range(nb_partition):
	features_weighted = getfeaturesWeighted(vect, title, content, partion*i, partion*(i+1))
	feature_arr = getFeaturearr(feature_arr, corpus[partion*i: partion*(i+1)], features_weighted, 
		featureName, addThres, threshold, n_top)
	if i == nb_partition-1:
		features_weighted = getfeaturesWeighted(vect, title, content, partion*i, len(corpus))
		feature_arr = getFeaturearr(feature_arr, corpus[partion*i: len(corpus)], features_weighted, 
			featureName, addThres, threshold, n_top)
	print("Part: ", i+1, "/", nb_partition)
	# print("features: ", len(feature_arr))
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

