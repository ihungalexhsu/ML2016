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

def process_data_ref(corpus,name):
    corpus = [ clean_html(line) for line in corpus ]
    corpus = [ get_words(line) for line in corpus ]
    corpus = [" ".join(word) for word in corpus]
    lm = WordNetLemmatizer()
    #using pos tag
    # corpus_pos= generate_corpus_pos(corpus, name)
    # corpus = [" ".join([lm.lemmatize(word[0], get_wordnet_pos(word[1])) for word in sentence for sentence in corpus_pos]
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
    lm = WordNetLemmatizer()
    #using pos tag
    # corpus_pos= generate_corpus_pos(corpus, name)
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
    ofile = open(outfileName, "w",encoding='utf-8')
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

def getfeaturesWeighted(vect, corpus, title, content, start, end, num, ratio):
    if num == 0:
        features_content = vect.transform(content[start:end]).toarray()
        features_title = vect.transform(title[start:end]).toarray()
        # features_title, svd = lsa(features_title, 80)
        # features_content, svd = lsa(features_content, 80)
        features_weighted = ratio[0]*features_title + ratio[1]*features_content
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
        tags = featureName[arg[:tops]]
        tags = np.unique( np.array(tags) )
        # print(tags)
        # tags = [ re.sub(" ", "-", word) for word in tags ]
        # print(tags)
        '''
        for item in tags:
            if len(item.split(" ")) > 1:
                item = "-".join(item.split(" "))
            feature_arr.append( item )
        '''
        feature_arr.append( tags )
        # print(feature_arr[-1])
    return feature_arr

def filterPostag(feature_arr):
    features = []
    for i in range(len(feature_arr)):
        tags = nltk.pos_tag(feature_arr[i])
        filtered_featureName = [tag[0] for tag in tags if tag[1].startswith('N') or tag[1]=='VBG']
        features.append( filtered_featureName )
    return features

def filterFromList(results, stop_words):
    for i in range(len(results)):
        results[i] = [ x for x in results[i] if x not in stop_words ]
    return results

def Tokenizer(corpus):
    return [sentence.split(" ") for sentence in corpus]

def getVect(num):
    my_words = read_words( "stop_words.txt")
    # my_data = pd.read_csv( 'stop_word_list_5000.csv', delimiter=',', skipinitialspace=True).as_matrix()
    # my_data = my_data[:,1].tolist()
    # nb_stopwords = int(len(my_data)*0.5)
    # my_data = my_data[:nb_stopwords]
    # my_stop_words = text.ENGLISH_STOP_WORDS.union(my_words)
    my_stop_words = text.ENGLISH_STOP_WORDS
    if num == 1:
        vect = TfidfVectorizer(max_df=0.5, min_df=1, analyzer='word', token_pattern=r'\b(\w\w+\S\w\w+)|\w\w+\b',
                               use_idf=True, stop_words=my_stop_words)
    elif num == 2:
        vect = TfidfVectorizer(max_df=0.5, min_df=1, analyzer='word', token_pattern=r'\b(\w\w+\S\w\w+)|\w\w+\b',
                               use_idf=False, stop_words=my_stop_words, norm='l2', sublinear_tf=True)
    elif num == 3:
        vect = TfidfVectorizer(max_df=0.5, min_df=1, analyzer='word', 
                               use_idf=True, stop_words=my_stop_words, norm='l2', sublinear_tf=True)
    elif num == 4:
        vect = TfidfVectorizer(max_df=0.5, min_df=1, analyzer='word', ngram_range=(1,2), 
                               use_idf=False, stop_words=my_stop_words, norm='l2', sublinear_tf=True)
    elif num == 5:
        vect = TfidfVectorizer(max_df=0.5, min_df=1, analyzer='word', token_pattern=r'\b(\w\w+\S\w\w+)|\w\w+\b',
                               use_idf=False, stop_words=my_stop_words, norm='l2', sublinear_tf=False)
    elif num == 6:
        vect = TfidfVectorizer(max_df=0.5, min_df=1, analyzer='word', token_pattern=r'\b(\w\w+\S\w\w+)|\w\w+\b',
                               use_idf=True, stop_words=my_stop_words, norm='l2', sublinear_tf=True)
    elif num == 7:
        vect = TfidfVectorizer(max_df=0.5, min_df=1, analyzer='word', token_pattern=r'\b(\w\w+\S\w\w+)|\w\w+\b',
                               use_idf=False, stop_words=my_stop_words, norm='l2', sublinear_tf=False)
    return vect

def readFromPickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    id_ = data['id_']
    title  = data['title']
    content= data['content']
    corpus = data['corpus']
    return id_, title, content, corpus 

def readFromCsv(filename):
    origin_data = pd.read_csv( filename, quotechar='"', skipinitialspace=True).as_matrix()
    origin_data = origin_data.astype('U')
    id_ = origin_data[:, 0]
    title  = origin_data[:, 1]
    content= origin_data[:, 2]
    corpus = origin_data[:, 3]
    return id_, title, content, corpus  

def generateOutput(nb_partition, corpus, vect, title, content, featureName, ratio):
    feature_arr = []
    partion = int(len(corpus)/nb_partition)
    while(partion ==0 ):
        nb_partition = int(nb_partition/2+1)
        partion = int(len(corpus)/nb_partition)
    # threshold = 0.8
    num = 0
    count = 0
    for i in range(nb_partition):
        if i != nb_partition-1:
            features_weighted = getfeaturesWeighted(vect, corpus, title, content, 
                                                partion*i, partion*(i+1), num, ratio)
            feature_arr = getFeaturearr(feature_arr, corpus[partion*i: partion*(i+1)], 
                                    features_weighted, featureName, addThres, threshold, n_top)
        else:
            features_weighted = getfeaturesWeighted(vect, corpus, title, content, 
                                                    partion*i, len(corpus), num, ratio)
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

# sys_input3 = sys.argv[3]
def getOutputVar(addTop, addThres):
    n_top = int(6)
    if addTop:
        n_top = int(sys_input3)
    if addThres:
        threshold = float(sys_input3)
    else:
        threshold = 1
    return n_top, threshold

def bigramProcess(corpus,title,content):
    #tokenize corpus first
    print ("tokenize data")
    corpus = [nltk.word_tokenize(sentences.lower()) for sentences in corpus]
    title = [nltk.word_tokenize(sentences.lower()) for sentences in title]
    content = [nltk.word_tokenize(sentences.lower()) for sentences in content]
    print ("create bigram")
    '''
    sentence=[['quantum','mechanics'],['newtonian','mechanics'],['general','relativity'],
              ['special','relativity'],['classical','mechanics'],['fluid','dynamics'],
              ['particle','physics'],['visible','light'],['statistical','mechanics'],
              ['black','holes'],['newtonian','gravity'],['newtonian','mechanics'],
              ['electromagnetic','radiation'],['condensed','matter'],['experimental','physics'],
              ['magnetic','fields'],['string','theory'],['lagrangian','formalism'],
              ['electric','circuits'],['mathematical','physics'],['angular','momentum'],
              ['differential','geometry'],['energy','conservation'],['nuclear','physics'],
              ['rotational','dynamics'],['quantum','information'],['soft','question'],
              ['resource','recommendations'],['electrical','resistance'],['quantum','electrodynamics'],
              ['group','theory'],['quantum','gravity']]    
    for aa in range(20):
        for bb in range(len(sentence)):
            corpus.append(sentence[bb])
    '''
    for weighted in range(10):
        corpus.append(title)
    bigram = Phrases(corpus,min_count=3,threshold=9.0,delimiter=b'-')
    print ("bigram corpus")
    title = bigram[title]
    title = [ " ".join(wordlist) for wordlist in title ]
    content = bigram[content]
    content = [ " ".join(wordlist) for wordlist in content ]

    corp = []
    for i in range(len(title)):
        corp.append(  title[i] + " " + content[i] )
    corpus = np.array(corp)
    # corpus = [ title[i] + " " + content[i] for i in range(len(title))] 
    return bigram,corpus,title,content

def wordListToFreqDict(wordlist):
    wordfreq = [wordlist.count(p) for p in wordlist]
    return dict(zip(wordlist,wordfreq))

# not working ...
def filterRareTags(tags_arr, threshold):
    # all_tags = np.array(tags_arr)
    all_tags = [item for sublist in tags_arr for item in sublist]
    wordFreq = wordListToFreqDict(all_tags)
    sel_tags = []
    for tags in tags_arr:
        sel_tags.append( [item for item in tags if wordFreq[item] > threshold] )
    return sel_tags, wordFreq


if __name__ == '__main__':
    # read from file
    path = sys.argv[1]
    outfileName = sys.argv[2]

    # n_top, threshold = getOutputVar(addTop, addThres)

    # define some parameters
    addTop = True
    addThres = False
    stemmer = []
    bigram = []
    n_top = 5
    threshold = 1
    vect_type = 2
    ratio = [8,1]
    ###
    # read from imput
    for i in range(len(sys.argv)):
        li = sys.argv[i].split("=")
        print(li)
        if li[0] == "vect":
            vect_type = int(li[1])
        elif li[0] == "n_top":
            n_top = int(li[1])
        elif li[0] == "weight":
            ratio = np.array( li[1].split(":") ).astype(int)
            # print(ratio)
    print(vect_type)
 
    # process data
    # id_, title, content, corpus = readFromPickle(path)
    id_, title, content, corpus = readFromCsv(path)
    print("Successfully read in data!")

    # ==================================================
    # do whatever you want !!!

    # change type to 'U'
    title = title.astype('U')
    content = content.astype('U')

    # define vector
    vect = getVect(vect_type)
    # fit vector
    # generate output
    features = vect.fit(corpus)
    weights = np.array( vect.idf_ )
    featureName = np.array(vect.get_feature_names() )
    print("Number of words: ", len(featureName) )
    
    print ("Start to generate output!")
    nb_partition = 5000
    feature_arr = generateOutput(nb_partition, corpus, vect, title, content, featureName, ratio)
    # print(feature_arr[:10])
    # feature_arr = [ sentence.split(" ") for sentence in title ]
    # print(feature_arr[:10])
    print ("Finish generating output!")

    # ==================================================

    print ("Save original answer file.")
    saveResults(outfileName , id_, feature_arr, stemmer, n_top)
    print("File name: ", outfileName, "!!!")


    # sel_tags, wordFreq = filterRareTags(feature_arr, 10)

    # ==================================================

    '''
    #bigram answer
    ans = getResults(feature_arr, id_, n_top)
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
    writeResults(outfileName+"_bigram", id_, ans)   
    print ("Finish saving to file!") 
    '''