import sys
import numpy as np
import pandas as pd
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import text
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
import re
import os.path
from gensim.models.phrases import Phrases
from collections import OrderedDict
from collections import defaultdict


def saveResults(outfileName, id_, result):
	ofile = open(outfileName, "w")
	ofile.write('\"id\",\"tags\"\n')
	for i in range(len(id_)):
		ofile.write( '"' + str(id_[i]) + '"' + "," + '"' + str(" ".join(result[i])) + '"' + '\n' )
	ofile.close()
	return True

def read_words(words_file):
    return [word for line in open(words_file, 'r') for word in line.split()]

my_stop_words = read_words('long_stop_word.txt')
stop_words = text.ENGLISH_STOP_WORDS.union(my_stop_words)

path = sys.argv[1]
outfileName = sys.argv[2]

def readFromData(filename):
    origin_data = pd.read_csv( filename, quotechar='"', skipinitialspace=True).fillna("none").as_matrix()
    id_ = origin_data[:, 0]
    title  = origin_data[:, 1]
    content= origin_data[:, 2]
    corpus = origin_data[:, 1:3]
    corpus = corpus.astype(object)
    corpus = corpus[:, 0] + " " + corpus[:, 1]
    return id_, title, content, corpus

def tf_idf(corpus):
    vect = CountVectorizer(max_df=0.5, min_df=10, analyzer='word', stop_words=my_stop_words,
                           max_features=10000, token_pattern=r'\b(\w\w+\S+\w\w+)|\w\w+\b')

    # fit vector
    vect.fit(corpus)
    return vect

def getfeaturesWeighted(titleVect, contentVect, title, content):
	features_title = titleVect.transform(title).toarray()
	features_content = contentVect.transform(content).toarray()

	return features_title, features_content

def getFeaturearr(feature_arr, features_title, features_content, titleName, contentName, n_top):
    for i in range(len(features_title)):
        selectedTitle = features_title[i]
        selectedContent = features_content[i]
        argTitle = selectedTitle.argsort()[-1*n_top:][::-1]
        argContent = selectedContent.argsort()[-1*n_top:][::-1]
        tagsTitle = nltk.pos_tag(nltk.word_tokenize(" ".join(titleName[argTitle[:n_top]])))
        tagsContent = nltk.pos_tag(nltk.word_tokenize(" ".join(contentName[argContent[:n_top]])))
        filtered_titleName = [tag[0] for tag in tagsTitle if tag[1].startswith('N')]
        filtered_contentName = [tag[0] for tag in tagsContent if tag[1].startswith('N')]
        filtered_tags = [" ".join(set(filtered_titleName + filtered_contentName))]
        feature_arr.append( filtered_tags )
    return feature_arr

def get_tags(corpus, title, content, titleVect, contentVect):
    feature_arr = []
    n_top = 3
    titleName = np.array( titleVect.get_feature_names() )
    contentName = np.array( contentVect.get_feature_names() )
    print("Start to generate output!")
    nb_part = 100
    part = int(len(corpus)/nb_part)
    for i in range(nb_part):
        if i != nb_part-1:
            features_title, features_content = getfeaturesWeighted(titleVect, contentVect,
                                               title[part*i:part*(i+1)], content[part*i:part*(i+1)])
            feature_arr = getFeaturearr(feature_arr, features_title, features_content,
                                        titleName, contentName, n_top)
        else:
            features_title, features_content = getfeaturesWeighted(titleVect, contentVect,
                                               title[part*i:], content[part*i:])
            feature_arr = getFeaturearr(feature_arr, features_title, features_content,
                                        titleName, contentName, n_top)
        print("Part: ", i+1, "/", nb_part)
    print("Finish generating output!")

    return feature_arr

if __name__ == '__main__':
    id_, title, content, corpus = readFromData(path)
    titleVect = tf_idf(title)
    contentVect = tf_idf(content)
    feature_arr = get_tags(corpus, title, content, titleVect, contentVect)
    # save to files
    saveResults(outfileName, id_, feature_arr)
    print("Finish save to file!")
