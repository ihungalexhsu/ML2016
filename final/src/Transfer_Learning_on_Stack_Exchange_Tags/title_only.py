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

my_stop_words = read_words('stop_words.txt')
stop_words = text.ENGLISH_STOP_WORDS.union(my_stop_words)

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
    '''
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
    '''
    corpus_tags = [nltk.pos_tag(sentence) for sentence in corpus]
    return corpus_tags

def clean_corpus(corpus):
    # '\n', '/', ',', '.', '?', '(', ')', ''' replaced by ' '
    clean_space = re.compile('[\n\/,\.\?()_]')
    # <xxx>, $xxx$, not alphabets and space and '_-', \begin xxx \end replaced by ''
    clean_empty = re.compile('<.*?>|\$+[^$]+\$+|[^a-zA-Z ]|\\+begin[^$]+\\+end')
    corp = []
    for sentence in corpus:
        word = sentence.split(' ')
        #delete words have @
        words = [w for w in word if '@' not in w]
        #delete word have \
        word = []
        for w in words:
            if ('\\' not in w):
                word.append(w)
            elif ((w=='\\begin') or (w=='\\end') or (w=='\\\\begin') or (w=='\\\\end')):
                word.append(w)
            elif ('$' in w):
                word.append(w)
            elif (w=='\\x08'):
                word.append('\\begin')
        corp.append(" ".join(word))
        corpus = corp
    corpus = [clean_space.sub(' ', sentence) for sentence in corpus]
    corpus = [clean_empty.sub('', sentence) for sentence in corpus]
    return corpus
'''
def clean_corpus(corpus):
    # '\n', '/', ',', '.', '?', '(', ')' replaced by ' '
    clean_space = re.compile('[\n\/,\.\?():\-]')
    # <xxx>, $xxx$, not alphabets and space, \begin xxx \end replaced by ''
    clean_empty = re.compile('<.*?>|\$+[^$]+\$+|[^a-zA-Z\' ]|\\+begin[^$]+\\+end|\'s')
    corpus = [clean_space.sub(' ', sentence) for sentence in corpus]
    corpus = [clean_empty.sub('', sentence) for sentence in corpus]

    return corpus
'''
def construct_phrase_dict(corpus_title):
    phrases = [[word for word in sentence if word.find('-') != -1] for sentence in corpus_title]
    mapping = defaultdict(lambda: '0')
    for sentence in phrases:
        for word in sentence:
            print(word)
            idx = word.find('-')
            abbrev = word[0]
            while idx != -1:
                idx += 1
                abbrev += word[idx]
                idx = word.find('-', idx)
            mapping[abbrev] = word

    return mapping

def extend_abbreviation(mapping, corpus_title):
    return [[word if mapping[word]=='0' else mapping[word] for word in sentence] for sentence in corpus_title]

def process_data():
    origin_data = pd.read_csv( path, quotechar='"', skipinitialspace=True).as_matrix()

    # process data
    id_ = origin_data[:, 0]
    #corpus = origin_data[:, 1:3]

    corpus_title = clean_corpus(origin_data[:, 1])
    corpus_content = clean_corpus(origin_data[:, 2])

    title_stream = [nltk.word_tokenize(sentence.lower()) for sentence in corpus_title]
    content_stream = [nltk.word_tokenize(sentence.lower()) for sentence in corpus_content]

    title_stream = [[word for word in sentence if word not in stop_words and len(word) >= 2]
                    for sentence in title_stream]
    content_stream = [[word for word in sentence if word not in stop_words and len(word) >= 2]
                      for sentence in content_stream]

    bigram = Phrases(title_stream, min_count=3, threshold=8, delimiter=b'-')
    trigram = Phrases(bigram[title_stream], min_count=3, threshold=8, delimiter=b'-')
    corpus_title = list(trigram[bigram[title_stream]])
    corpus_content = list(trigram[bigram[content_stream]])

    mapping = construct_phrase_dict(corpus_title + corpus_content)
    corpus_title = extend_abbreviation(mapping, corpus_title)
    corpus_content = extend_abbreviation(mapping, corpus_content)

    title = [" ".join([word for word in sentence if len(word) >= 3]) for sentence in corpus_title]
    content = [" ".join([word for word in sentence if len(word) >= 3]) for sentence in corpus_content]

    corpus = [a + " " + b for a, b in zip(title, content)]
    print("finishing preprocess")
    return corpus, title, content, id_

def tf_idf(corpus):
    vect = CountVectorizer(max_df=0.5, min_df=10, analyzer='word', stop_words=my_stop_words,
                           max_features=5000, token_pattern=r'\b(\w\w+\S\w\w+)|\w\w+\b')

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
    '''
    title_tags = [nltk.pos_tag(nltk.word_tokenize(sentence)) for sentence in title]
    #content_tags = [nltk.pos_tag(nltk.word_tokenize(sentence)) for sentence in content]
    filtered_title_tags = [[" ".join([tag[0] for tag in sentence if tag[1].startswith('N')])]
                           for sentence in title_tags]
    print(filtered_title_tags)
    #filtered_content_tags = [" ".join([tag[0] for tag in sentence if tag[1].startswith('N') or tag[1]=='VBG'])
    #                         for sentence in content_tags]
    #filtered_all_tags = [a + " " + b for a, b in zip(filtered_title_tags, filtered_content_tags)]
    '''
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
    corpus, title, content, id_ = process_data()
    titleVect = tf_idf(title)
    contentVect = tf_idf(content)
    feature_arr = get_tags(corpus, title, content, titleVect, contentVect)
    # save to files
    saveResults(outfileName, id_, feature_arr)
    print("Finish save to file!")
