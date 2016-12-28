import sys
import numpy as np
import pandas as pd
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
#import bottleneck as bn # sorting
import re
import os.path
from gensim.models import Phrases

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

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def generate_corpus_pos(corpus, name):
    if os.path.isfile('biology_corpus_pos_' + name + '.txt'):
        corpus_tags = []
        with open('biology_corpus_pos_' + name + '.txt', 'r') as f:
            for sentence in f:
                word = sentence.split(' ')
                corpus_tags.append([tuple(word.strip('\n').split(',')) for word in sentence.split(' ')])
    else:
        corpus_tags = [nltk.pos_tag(nltk.word_tokenize(sentence)) for sentence in corpus]
        with open('biology_corpus_pos_' + name + '.txt', 'w') as f:
            for sentence in corpus_tags:
                f.write(' '.join('%s,%s' % word for word in sentence))
                f.write('\n')
    return corpus_tags

def clean_corpus(corpus):
    # '\n', '/', ',', '.', '?', '(', ')', ''' replaced by ' '
    clean_space = re.compile('[\n\/,\.\?()\']')
    # <xxx>, $xxx$, not alphabets and space and '_-', \begin xxx \end replaced by ''
    clean_empty = re.compile('<.*?>|\$+[^$]+\$+|[^a-zA-Z_\- ]|\\+begin[^$]+\\+end')
    corpus = [clean_space.sub(' ', sentence) for sentence in corpus]
    corpus = [clean_empty.sub('', sentence) for sentence in corpus]
    #corpus = [sentence.translate(sentence.maketrans({key: None for key in string.punctuation}))
    #          for sentence in corpus]

    return corpus

def process_data():
    origin_data = pd.read_csv( path, quotechar='"', skipinitialspace=True).as_matrix()

    # process data
    id_ = origin_data[:, 0]
    corpus = origin_data[:, 1:3]

    corpus_title = clean_corpus(corpus[:, 0])
    corpus_content = clean_corpus(corpus[:, 1])

    title_stream = [sentence.split(" ") for sentence in corpus_title]
    content_stream = [sentence.split(" ") for sentence in corpus_content]

    bigram = Phrases(title_stream + content_stream)
    corpus_title = bigram[corpus_title]
    corpus_content = bigram[corpus_content]

    corpus_pos_title = generate_corpus_pos(corpus_title, 'title')
    corpus_pos_content = generate_corpus_pos(corpus_content, 'content')

    lm = WordNetLemmatizer()

    title = [" ".join([lm.lemmatize(word[0], get_wordnet_pos(word[1])) for word in sentence])
            for sentence in corpus_pos_title]
    content = [" ".join([lm.lemmatize(word[0], get_wordnet_pos(word[1])) for word in sentence])
            for sentence in corpus_pos_content]
    corpus = [a + " " + b for a, b in zip(title, content)]
    # tags   = origin_data[:, 3]
    return corpus, title, content, id_

def tf_idf(corpus, title, content):
    my_words = read_words("stop_words.txt")
    my_stop_words = text.ENGLISH_STOP_WORDS.union(my_words)
    vect = TfidfVectorizer(max_df=0.5, min_df=10, analyzer='word',
                           use_idf=True, stop_words=my_stop_words)

    # fit vector
    vect.fit(corpus)
    return vect

def tagsThreshold(threshold, selectedFeature, n_top):
	num = 1
	print(selectedFeature)
	while selectedFeature[num-1]*threshold > selectedFeature[num]:
		num = num + 1
		if num == n_top: break
	return num

def getfeaturesWeighted(vect, title, content, start, end):
	features_title = vect.transform(title[start:end]).toarray()
	features_content = vect.transform(content[start:end]).toarray()
	features_weighted = 2*features_title + features_content
	return features_weighted

def getFeaturearr(feature_arr, corpus, features_weighted, featureName, addThres, threshold, n_top):
    for i in range(len(features_weighted)):
        selectedFeature = features_weighted[i]
        arg = selectedFeature.argsort()[-1*n_top:][::-1]
        if addThres == True:
            tops = tagsThreshold(threshold, selectedFeature[arg], n_top)
        else:
            tops = n_top
        tags = nltk.pos_tag(nltk.word_tokenize(" ".join(featureName[arg[:tops]])))
        filtered_featureName = [tag[0] for tag in tags if tag[1].startswith('N') or tag[1]=='VBG']
        feature_arr.append( filtered_featureName )
    return feature_arr

def get_tags(corpus, title, content, vectorizer):
    feature_arr = []
    n_top = 3
    #weights = np.array( vect.idf_ )
    featureName = np.array( vect.get_feature_names() )
    addThres = False
    print("Start to generate output!")
    nb_partition = 15
    partition = int(len(corpus)/nb_partition)
    threshold = 0.8
    count = 0
    for i in range(nb_partition):
        if i != nb_partition-1:
            features_weighted = getfeaturesWeighted(vect, title, content, partition*i, partition*(i+1))
            feature_arr = getFeaturearr(feature_arr, corpus[partition*i: partition*(i+1)],
                                        features_weighted,featureName, addThres, threshold, n_top)
        else:
            features_weighted = getfeaturesWeighted(vect, title, content, partition*i, len(corpus))
            feature_arr = getFeaturearr(feature_arr, corpus[partition*i: len(corpus)], 
                                        features_weighted,featureName, addThres, threshold, n_top)
        print("Part: ", i+1, "/", nb_partition)
    print("Finish generating output!")

    return feature_arr

if __name__ == '__main__':
    corpus, title, content, id_ = process_data()
    vect = tf_idf(content, title, content)
    feature_arr = get_tags(corpus, title, content, vect)
    # save to files
    saveResults(outfileName, id_, feature_arr)
    print("Finish save to file!")
