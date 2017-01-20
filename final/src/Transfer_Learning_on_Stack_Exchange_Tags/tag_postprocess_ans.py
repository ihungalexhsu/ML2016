#coding=utf-8
import sys
import numpy as np
import pandas as pd
import string
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from numpy import genfromtxt
from sklearn.feature_extraction import text
from gensim.models.phrases import Phrases
import itertools
import bottleneck as bn # sorting
import os.path
import collections
import re


class SnowCastleStemmer(nltk.stem.SnowballStemmer):
    """ A wrapper around snowball stemmer with a reverse lookip table """

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self._stem_memory = collections.defaultdict(set)
        # switch stem and memstem
        self._stem=self.stem
        self.stem=self.memstem

    def memstem(self, word):
        """ Wrapper around stem that remembers """
        stemmed_word = self._stem(word)
        self._stem_memory[stemmed_word].add(word)
        return stemmed_word

    def unstem(self, stemmed_word):
        """ Reverse lookup """
        return sorted(self._stem_memory[stemmed_word], key=len)

################ clean word #######################
def clean_html(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

def get_words(text):
    word_split = re.compile('[^a-zA-Z\\+\\-/]')
    return [word.strip().lower() for word in word_split.split(text)]
    # return word_split

def removeWordFromStr(sentence, length):
    string = [ word for word in sentence.split(" ") if len(word) > length ]
    return " ".join(string)

def process_data_ref(corpus, name):
    corpus = [ clean_html(line) for line in corpus ]
    corpus = [ get_words(line) for line in corpus ]
    corpus = [" ".join(word) for word in corpus]
    corpus = [ removeWordFromStr(sentence, 3) for sentence in corpus ]
    lm = WordNetLemmatizer()
    #using pos tag
    corpus_pos= generate_corpus_pos(corpus, name)
    corpus = [" ".join([lm.lemmatize(word[0], get_wordnet_pos(word[1])) for word in sentence])
              for sentence in corpus_pos]
    
    return corpus

def process_data_stem(corpus, stemmer):
    corpus = [re.sub(r'(<[^<]+?>)|(\d+)', '', sentence) for sentence in corpus]
    corpus = [re.sub(r'(\n)',' ',sentence) for sentence in corpus]
    corpus = [sentence.translate(sentence.maketrans({key: None for key in (string.punctuation).replace("-"," ") }))
              for sentence in corpus]
    corpus = [" ".join([stemmer.stem(word) for word in sentence.split(" ")]) for sentence in corpus]
    return corpus, stemmer

def clean_corpus(corpus):
    # '\n', '/', ',', '.', '?', '(', ')', ''' replaced by ' '
    clean_space = re.compile('[\n\/,\.\?()\'_]')
    # <xxx>, $xxx$, not alphabets and space and '_-', \begin xxx \end replaced by ''
    clean_empty = re.compile('<.*?>|\$+[^$]+\$+|[^a-zA-Z\- ]|\\+begin[^$]+\\+end')
    corpus = [clean_space.sub(' ', sentence) for sentence in corpus]
    corpus = [clean_empty.sub('', sentence) for sentence in corpus]
    #corpus = [sentence.translate(sentence.maketrans({key: None for key in string.punctuation}))
    #          for sentence in corpus]
    return corpus

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
    if os.path.isfile('test_corpus_pos_' + name + '.txt'):
        corpus_tags = []
        with open('test_corpus_pos_' + name + '.txt', 'r') as f:
            for sentence in f:
                word = sentence.split(' ')
                corpus_tags.append([tuple(word.strip('\n').split(',')) for word in sentence.split(' ')])
    else:
        corpus_tags = [nltk.pos_tag(nltk.word_tokenize(sentence)) for sentence in corpus]
        with open('test_corpus_pos_' + name + '.txt', 'w') as f:
            for sentence in corpus_tags:
                f.write(' '.join('%s,%s' % word for word in sentence))
                f.write('\n')
    return corpus_tags

def process_data(corpus,name):
    # process data
    corpus = clean_corpus(corpus)
    corpus = [ removeWordFromStr(sentence, 3) for sentence in corpus ]
    lm = WordNetLemmatizer()
    #using pos tag
    corpus_pos= generate_corpus_pos(corpus, name)
    corpus = [" ".join([lm.lemmatize(word[0], get_wordnet_pos(word[1])) for word in sentence])
              for sentence in corpus_pos]
    #corpus = [" ".join([lm.lemmatize(word) for word in sentence.split(" ")]) for sentence in corpus]
    return corpus

#########################get answer###########################
def getResults(result, id_, n_tags=3):
    ans = []
    for i in range(len(id_)):
        if len(result[i]) > n_tags:
            arr = result[i][:n_tags]
        else:
            arr = result[i]
        ans.append(arr)
    return ans

def writeResults(outfileName, id_, ans):
    ofile = open(outfileName + '.csv', "w",encoding='utf-8')
    ofile.write('\"id\",\"tags\"\n')
    for i in range(len(id_)):
        ofile.write( '"' + str(id_[i]) + '"' + "," + '"' + str(" ".join(ans[i])) + '"' + '\n' )

def saveResults(outfileName, id_, result, stemmer, n_tags=3):
    ofile = open(outfileName + '.csv', "w",encoding='utf-8')
    ofile.write('\"id\",\"tags\"\n')
    for i in range(len(id_)):
        if len(result[i]) > n_tags:
            arr = result[i][:n_tags]
        else:
            arr = result[i]
        ofile.write( '"' + str(id_[i]) + '"' + "," + '"' + str(" ".join(arr)) + '"' + '\n' )
    '''
    unstemfile = open('unstemfile'+'.csv','w',encoding='utf-8")
    for j in range(len(id_)):
    unstemfile.write('"'+str(id_[j])+'"'+","+'"')
    if len(result[j]) > n_tags:
    for k in range(n_tags):
    unstemfile.write(str(" ".join(stemmer.unstem(result[j][k]))))
    unstemfile.write(" ")
    else:
    for k in range(len(result[j])):
    unstemfile.write(str(" ".join(stemmer.unstem(result[j][k]))))
    unstemfile.write(" ")
    unstemfile.write('"'+'\n')

    unstemfile.close()
    '''
    ofile.close()
    return True

####################### main function ###########################
def read_words(words_file):
    return [word for line in open(words_file, 'r') for word in line.split()]

def lsa(X, n_components=80):
    print ("Performing dimensionality reduction using LSA")
    svd = TruncatedSVD(n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(X)
    return X, svd

def tagsThreshold(threshold, selectedFeature, n_top):
    num = 1
    # print(selectedFeature)
    while selectedFeature[num-1]*threshold < selectedFeature[num]:
        num = num + 1
        if num == n_top: break
    return num

def getfeaturesWeighted(vect, corpus, title, content, start, end, num):
    if num == 0:
        features_content = vect.transform(content[start:end]).toarray()
        features_title = vect.transform(title[start:end]).toarray()
        # features_title, svd = lsa(features_title, 80)
        # features_content, svd = lsa(features_content, 80)
        features_weighted = 8*features_title + features_content
    elif num == 1:
        features_title = vect.transform(title[start:end]).toarray()
        features_content = vect.transform(content[start:end]).toarray()
        features_weighted = 8*features_title + features_content
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

def filterFromList(results, stop_words):
    for i in range(len(results)):
        results[i] = [ x for x in results[i] if x not in stop_words ]
    return results

def getVect(num):
    my_words = read_words( "stop_words.txt")
    # my_data = pd.read_csv( 'stop_word_list_5000.csv', delimiter=',', skipinitialspace=True).as_matrix()
    # my_data = my_data[:,1].tolist()
    # nb_stopwords = int(len(my_data)*0.5)
    # my_data = my_data[:nb_stopwords]
    # my_stop_words = text.ENGLISH_STOP_WORDS.union(my_words)
    my_stop_words = text.ENGLISH_STOP_WORDS.union(stop_words_2)
    if num == 1:
        vect = TfidfVectorizer(max_df=0.5, min_df=1, analyzer='word', 
                               use_idf=True, stop_words=my_stop_words)
    elif num == 2:
        vect = TfidfVectorizer(max_df=0.5, min_df=1, analyzer='word', 
                               use_idf=False, stop_words=my_stop_words, norm='l2', sublinear_tf=True)
    elif num == 3:
        vect = TfidfVectorizer(max_df=0.5, min_df=1, analyzer='word', 
                               use_idf=True, stop_words=my_stop_words, norm='l2', sublinear_tf=True)
    return vect

def readFromPickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    id_ = data['id_']
    feature_arr  = data['feature_arr']
    corpus = data['corpus']
    return id_, feature_arr, corpus

def readFromAns(filename):
    origin_data = pd.read_csv( filename, quotechar='"', skipinitialspace=True).as_matrix()
    id_ = origin_data[:, 0]
    tags = origin_data[:, 1]
    return id_, tags  

def readFromCsv(filename):
    origin_data = pd.read_csv( filename, quotechar='"', skipinitialspace=True).as_matrix()
    id_ = origin_data[:, 0]
    title  = origin_data[:, 1]
    content= origin_data[:, 2]
    corpus = origin_data[:, 3]
    return id_, title, content, corpus  

def generateOutput(nb_partition, corpus, vect, title, content, featureName):
    feature_arr = []
    partion = int(len(corpus)/nb_partition)
    # threshold = 0.8
    num = 0
    count = 0
    for i in range(nb_partition):
        features_weighted = getfeaturesWeighted(vect, corpus, title, content, 
                                                partion*i, partion*(i+1), num)
        feature_arr = getFeaturearr(feature_arr, corpus[partion*i: partion*(i+1)], 
                                    features_weighted, featureName, addThres, threshold, n_top)
        if i == nb_partition-1:
            features_weighted = getfeaturesWeighted(vect, corpus, title, content, 
                                                    partion*i, len(corpus), num)
            feature_arr = getFeaturearr(feature_arr, corpus[partion*i: len(corpus)],
                                        features_weighted, featureName, addThres, threshold, n_top)
        print ("Part: ", i+1, "/", nb_partition)
    return feature_arr

def printTfidfWeight(features_weighted, n_top, featureName, weights):
    for i in range(5):
        selectedFeature = features_weighted[i]
        arg = selectedFeature.argsort()[-1*n_top:][::-1]
        selected = np.concatenate( (featureName[arg], weights[arg]), axis=0)
        selected = selected.reshape((2,n_top)).T
        print (selected)
        feature_arr.append(selected)
    return True

def preprocessing(corpus, title, content, num):
    stemmer = []
    if num == 0:
        stemmer= SnowCastleStemmer('english')
        content,_  = process_data_stem(content, stemmer)
        stemmer= SnowCastleStemmer('english')
        title,_  = process_data_stem(title, stemmer)
        stemmer= SnowCastleStemmer('english')
        corpus, stemmer = process_data_stem(corpus, stemmer)
    elif num == 1:
        corpus = process_data(corpus,'corpus')
        title  = process_data(title,'title')
        content  = process_data(content,'content')
    elif num == 2:
        corpus = np.array(process_data_ref(corpus,'corpus') )
        title = np.array(process_data_ref(title,'title') )
        content = np.array(process_data_ref(content,'content') )
    return corpus, title, content, stemmer

def getOutputVar(addTop, addThres):
    n_top = int(6)
    if addTop:
        n_top = int(sys_input3)
    if addThres:
        threshold = float(sys_input3)
    else:
        threshold = 1
    return n_top, threshold

def bigramProcess(corpus):
    #tokenize corpus first
    print ("tokenize data")
    corpus = [nltk.word_tokenize(sentences.lower()) for sentences in corpus]
    print ("create bigram")
    sentence=['quantum','mechanics', 'newtonian','mechanics','general','relativity', 'special','relativity', 'classical','mechanics', 'fluid','dynamics', 'particle','physics','visible','light', 'statistical','mechanics','black','holes','newtonian','gravity', 'electromagnetic','radiation', 'condensed','matter', 'experimental','physics','magnetic','fields', 'string','theory', 'lagrangian','formalism','electric','circuits', 'mathematical','physics', 'mass', 'angular','momentum', 'differential','geometry','energy','conservation','nuclear','physics','rotational','dynamics', 'quantum','information','soft','question','resource','recommendations','electrical','resistance', 'quantum','electrodynamics','group','theory','quantum','gravity']    
    for aa in range(11):
        corpus.append(sentence)
    bigram = Phrases(corpus,min_count=3,threshold=10.0,delimiter=b'-')
    print ("bigram corpus")
    #corpus = bigram[corpus]
    return bigram,corpus

def filterPostag(feature_arr):
    features = []
    for i in range(len(feature_arr)):
        tags = nltk.pos_tag(feature_arr[i])
        filtered_featureName = [tag[0] for tag in tags if tag[1].startswith('N') or tag[1]=='VBG']
        features.append( filtered_featureName )
    return features

def wordCount(feature_arr):
    from nltk import FreqDist
    import nltk
    all_tags = " ".join(feature_arr)
    words = nltk.tokenize.word_tokenize(all_tags)
    fdist = FreqDist(words)
    return fdist

    '''
    from sklearn.feature_extraction.text import CountVectorizer
    vect = CountVectorizer(min_df=1)
    # vect = TfidfVectorizer(min_df=1, analyzer='word', token_pattern=r'\b(\w\w+\S\w\w+)|\w\w+\b',
                           # use_idf=False, smooth_idf=False)
    vect.fit(feature_arr)
    featureName = vect.get_feature_names()
    freq = vect.fit_transform(featureName).toarray()
    return feature_arr
    '''
def filterRareTags(feature_arr, threshold):
    # all_tags = np.array(tags_arr)
    feature_arr_join = [ " ".join(tags) for tags in feature_arr ]
    feature_arr_split = feature_arr
    wordFreq = wordCount(feature_arr_join)
    sel_tags = []
    for tags in feature_arr_split:
        sel_tags.append( [item for item in tags if wordFreq[item] > threshold] )
    return sel_tags, wordFreq


if __name__ == '__main__':
    # read from file
    path_corp = sys.argv[1]
    path = sys.argv[2]
    outfileName = sys.argv[3]
    # id_, feature_arr, corpus = readFromPickle(path)
    id_, feature_arr = readFromAns(path)
    id_, title, content, corpus = readFromCsv(path_corp)
    feature_arr = [ tags.split(" ") for tags in feature_arr ]
    ans = feature_arr


    # parameters
    filter_rare = True
    bigram = False
    pos = False

    ###
    if filter_rare:
        ans,wordFreq = filterRareTags(feature_arr, 300)
        import operator
        sorted_wordFreq = sorted(wordFreq.items(), key=operator.itemgetter(1))


    if pos:
        ans = filterPostag(feature_arr)

    #bigram answer
    # ans = getResults(feature_arr, id_, n_top)
    if bigram:
        ans = feature_arr
        bigram,_ = bigramProcess(corpus)
        for i in range(len(id_)):
            total_permu = list(itertools.permutations(ans[i],2))
            for j in range(len(total_permu)):
                total_permu[j] = list(total_permu[j])
            after_bigram = [bigram[words] for words in total_permu]
            valid_bigram = [valid for valid in after_bigram if len(valid)==1 ]
            for k in range(len(valid_bigram)):
                ans[i] = np.append(ans[i],valid_bigram[k][0])
                index = np.argwhere(ans[i]==(valid_bigram[k][0].split('-'))[0])
                ans[i] = np.delete(ans[i], index)
                index = np.argwhere(ans[i]==(valid_bigram[k][0].split('-'))[1])
                ans[i] =np.delete(ans[i], index)
    '''
    for i in range(len(ans)):
        bigram = [ word for word in ans[i] if len(word.split("-"))>1 ]
        print(bigram)
        for j in range(len(bigram)):
            word = bigram[j].split("-")
            print(word)
            for k in range(2):
                print(word[k])
                index = np.argwhere(ans[i]==word[k])
                print(index)
                ans[i] = np.delete(ans[i], index)
            print(ans[i])
    '''
    writeResults(outfileName, id_, ans)
    print("Output file: ", outfileName)
