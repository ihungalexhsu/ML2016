#coding=utf-8
import sys
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import wordnet
from sklearn.feature_extraction import text
from gensim.models.phrases import Phrases
import os.path
import collections
import re

################ clean word #######################
def clean_corpus(corpus):
    clean_space = re.compile('[\n\/,\.\?()_]')
    clean_empty = re.compile('<.*?>|\$+[^$]+\$+|[^a-zA-Z\' ]|\\+begin[^$]+\\+end')
    corp = []
    for sentence in corpus:
        word = sentence.split(' ')
        words = [w for w in word if '@' not in w]
        words = [w for w in words if 'http' not in w]
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

def generate_corpus_pos(corpus):
    corpus_tags = [nltk.pos_tag(nltk.word_tokenize(sentence)) for sentence in corpus]
    return corpus_tags

def process_data(corpus):
    corpus = clean_corpus(corpus)
    return corpus

def saveFile(outfileName, id_, corpus, title, content):
    ofile = open(outfileName, "w",encoding='utf-8')
    ofile.write('id, title, content, corpus\n')
    for i in range(len(id_)):
        ofile.write(  str(id_[i]) + "," + title[i] + ',' + content[i] + ',' + corpus[i] + '\n' )
    ofile.close()
    return True
####################### main function ###########################
def read_words(words_file):
    return [word for line in open(words_file, 'r') for word in line.split()]

def readFromData(filename):
    origin_data = pd.read_csv( filename, quotechar='"', skipinitialspace=True).as_matrix()
    id_ = origin_data[:, 0]
    title  = origin_data[:, 1]
    content= origin_data[:, 2]
    corpus = origin_data[:, 1:3]
    corpus = corpus.astype(object)
    corpus = corpus[:, 0] + " " + corpus[:, 1]
    return id_, title, content, corpus  

def preprocessing(corpus, title, content):
    corpus = process_data(corpus)
    title  = process_data(title)
    content  = process_data(content)
    return corpus, title, content

def removeWordFromStr(sentence, short_length, long_length):
    string = [ word for word in sentence.split(" ") if len(word) > short_length ]
    string = [ word for word in sentence.split(" ") if len(word) < long_length ]
    return " ".join(string)

def deletecomponent(corpus, numremove, numremoveMax):
    my_words = read_words( "long_stop_word.txt")
    my_stop_words = text.ENGLISH_STOP_WORDS.union(my_words)
    
    corpus = [" ".join([word for word in sentence.lower().split(' ')
                    if word not in my_stop_words]) for sentence in corpus]
    corpus = [ removeWordFromStr(sentence, numremove, numremoveMax) for sentence in corpus ]
    
    clean_empty = re.compile('<.*?>|\$+[^$]+\$+|[^a-zA-Z ]|\\+begin[^$]+\\+end')
    corpus = [clean_empty.sub('', sentence) for sentence in corpus]
    corpus = [" ".join([word for word in sentence.lower().split(' ')
                    if word not in my_stop_words]) for sentence in corpus]
    corpus = [ removeWordFromStr(sentence, numremove, numremoveMax) for sentence in corpus ]

    return corpus

def bigramProcess(corpus,title,content,minCount = 5,thresholds = 10.0):
    #tokenize corpus first
    print ("... tokenize data ...")
    corpus = [[w for w in nltk.word_tokenize(sentences.lower())if w != "'s"] for sentences in corpus]
    title = [[w for w in nltk.word_tokenize(sentences.lower()) if w!= "'s"] for sentences in title]
    content = [[w for w in nltk.word_tokenize(sentences.lower()) if w!= "'s"] for sentences in content]
    print ("... create bigram ...")
    bigram = Phrases(corpus,delimiter=b'-',min_count=minCount,threshold=thresholds)
    print ("... bigram corpus ...")
    title = bigram[title]
    title = [ " ".join(wordlist) for wordlist in title ]
    content = bigram[content]
    content = [ " ".join(wordlist) for wordlist in content ]
    corp = []
    for i in range(len(title)):
        corp.append(title[i] + " " + content[i])
    corpus = np.array(corp)
    return bigram,corpus,title,content

# ====================================================

def getTopBigram(bigram, numbershow, selectNandJ):
    #return a list of bigram words
    #bigram is a gensim-Pharses which already been construct with words.
    #numbershow is a integer that the number of most common bigram that user want
    #selectNandJ is a boolean that want the return bigram words have been selected or not,
    #   if it's true the return length would less then the numbershow
    
    answer = []
    bigram_counter = collections.Counter()
    for keys in bigram.vocab.keys():
        if len(str(keys).split('-')) > 1:
            bigram_counter[keys] += bigram.vocab[keys]
    for keys,counts in bigram_counter.most_common(numbershow):
        if type(keys)!=str:
            key = keys.decode('utf-8')
        else:
            key = keys
        if selectNandJ:
            word = key.split('-')
            if len(word[0])!=0 and len(word[1])!=0:
                firstword = word[0]
                secondword = word[1]
                pos_first = nltk.pos_tag([firstword])
                pos_second = nltk.pos_tag([secondword])
                if ((pos_first[0][1].startswith('N') or pos_first[0][1].startswith('J')) 
                        and pos_second[0][1].startswith('N')):
                    answer.append(key)
                    # print(str(key) + "      "+str(counts))
        else:
            answer.append(key)
            # print(str(key) + "      "+str(counts))
    return answer

if __name__ == '__main__':
    # read from file
    if len(sys.argv) < 2:
        print ("Usage:")
        print ("python3 tag_preprocess_data.py data/test.csv test_corpus" )
        print ("Defualt: with tri-gram, long_stop_word.txt")
        sys.exit()

    path = sys.argv[1]
    outfileName = sys.argv[2]
    debug = False
    Trigram = True

    # process data
    id_, title, content, corpus = readFromData(path)
    print("Successfully load data!")
    corpus, title, content = preprocessing(corpus, title, content)
    print("Successfully preprocess data!")

    if debug:
        saveFile(outfileName + "_step1", id_, corpus, title, content)
    # Clean stop words
    corpus = deletecomponent(corpus,2, 15)
    content = deletecomponent(content,2, 15)
    title = deletecomponent(title,2, 15)
    print("Successfully delete-componet!")

    if debug:
        saveFile(outfileName + "_step2", id_, corpus, title, content)
    # create bigram
    bigram,corpus,title,content = bigramProcess(corpus,title,content)
    print("Successfully do bi-gram to data!")
    
    if debug:
        saveFile(outfileName + "_step3", id_, corpus, title, content)
    if Trigram:
        trigram,corpus,title,content = bigramProcess(corpus,title,content,3,6)
        print("Successfully do tri-gram to data!")
    
    saveFile(outfileName, id_, corpus, title, content)
    print("Successfully output data with 'id_', 'title', 'content', 'corpus' !")
