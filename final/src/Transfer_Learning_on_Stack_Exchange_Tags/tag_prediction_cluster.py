import sys
import numpy as np
import pandas as pd
import string
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
#import bottleneck as bn # sorting
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
def process_data():
    origin_data = pd.read_csv( path, quotechar='"', skipinitialspace=True).as_matrix()

    # process data
    id_ = origin_data[:, 0]

    corpus = origin_data[:, 1:3]
    corpus = corpus.astype(object)
    corpus = corpus[:, 0] + corpus[:, 0] + " " + corpus[:, 1]
    lm = WordNetLemmatizer()
    corpus = [re.sub(r'(<[^<]+?>)|(\n)|(\d+)', '', sentence) for sentence in corpus]
    corpus = [sentence.translate(sentence.maketrans({key: None for key in string.punctuation}))
              for sentence in corpus]
    corpus = [" ".join([lm.lemmatize(word) for word in sentence.split(" ")]) for sentence in corpus]
    print(corpus)
    # tags   = origin_data[:, 3]
    return corpus, id_

def tf_idf(corpus):
    my_words = read_words("stop_words.txt")
    my_stop_words = text.ENGLISH_STOP_WORDS.union(my_words)
    vect = TfidfVectorizer(max_df=0.95, min_df=10, analyzer='word', max_features=10000,
                           use_idf=True, stop_words=my_stop_words)

    # fit vector
    features = vect.fit_transform(corpus)
    print("Finish feature extraction!")
    print("Size = ", features.shape)

    return features, vect

def lsa(X, n_components=80):
    print("Performing dimensionality reduction using LSA")
    svd = TruncatedSVD(n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(X)

    return X, svd

def kmeans(X, svd, vectorizer, n_clusters=20):
    km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1,
                verbose=0)

    print("Clustering sparse data with %s" % km)
    km.fit(X)

    return km

def get_tags(km, vectorizer, features, n_clusters):
    original_space_centroids = svd.inverse_transform(features)
    order_centroids = original_space_centroids.argsort()[:, ::-1]

    terms = np.asarray(vectorizer.get_feature_names())

    feature_arr = []
    for i in range(len(features)):
        idx = km.labels_[i]
        tags = np.greater(original_space_centroids[idx, order_centroids[idx, :5]], 0.1).astype(np.int)
        num_tags = len(np.trim_zeros(tags))
        if num_tags == 0:
            num_tags += 1
        feature_arr.append(terms[order_centroids[idx, :num_tags]])
    '''
    # print out the entries with highest weight
    feature_arr = []
    n_top = 3
    weights = np.array( vect.idf_ )
    featureName = np.array( vect.get_feature_names() )
    # generate output
    print("Start to generate output!")
    partion = int(len(features)/10)
    count = 0
    for i in range(len(features)):
        selectedFeature = features[i]
        arg = selectedFeature.argsort()[-1*n_top:][::-1]
        tags = np.greater(selectedFeature[arg], 0.3).astype(np.int)
        num_tags = len(np.trim_zeros(tags))
        if num_tags == 0:
            num_tags += 1
        # arg = bottleneck.argpartsort(-a, 10)[:10]
        feature_arr.append( featureName[arg[:num_tags]] )
        if i%partion == 0:
            print("Yep: ", count, "/10")
            count = count + 1
    print("Finish generating output!")
    '''
    return feature_arr

if __name__ == '__main__':
    corpus, id_ = process_data()
    features, vect = tf_idf(corpus)
    features, svd = lsa(features, 80)
    n_clusters = 800
    km = kmeans(features, svd, vect, n_clusters)
    feature_arr = get_tags(km, vect, features, n_clusters)
    # save to files
    saveResults(outfileName, id_, feature_arr)
    print("Finish save to file!")
