#coding=utf-8
import sys
import numpy as np
import pandas as pd
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from numpy import genfromtxt
from sklearn.feature_extraction import text
from gensim.models.phrases import Phrases
from collections import OrderedDict
from collections import defaultdict
import itertools
import bottleneck as bn # sorting
import os.path
import collections
import re

stop_words_2 = set(['螳螂捕蝉', '黄雀在后', 'a', "a's", 'able', 'about', 'above', 'according', 'accordingly', 'across', 'actually', 'after', 'afterwards', 'again', 'against', "ain't", 'all', 'allow', 'allows', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'apart', 'appear', 'appreciate', 'appropriate', 'are', "aren't", 'around', 'as', 'aside', 'ask', 'asking', 'associated', 'at', 'available', 'away', 'awfully', 'b', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'believe', 'below', 'beside', 'besides', 'best', 'better', 'between', 'beyond', 'both', 'brief', 'but', 'by', 'c', "c'mon", "c's", 'came', 'can', "can't", 'cannot', 'cant', 'cause', 'causes', 'certain', 'certainly', 'changes', 'clearly', 'co', 'com', 'come', 'comes', 'concerning', 'consequently', 'consider', 'considering', 'contain', 'containing','car', 'contains', 'corresponding', 'could', "couldn't", 'course', 'currently', 'd', 'definitely', 'described', 'despite', 'did', "didn't", 'different', 'do', 'does', "doesn't", 'doing', "don't", 'done', 'down', 'downwards', 'during', 'e','high','school', 'each', 'edu', 'eg', 'eight', 'either', 'else', 'elsewhere', 'enough', 'entirely', 'especially', 'et', 'etc', 'even', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'f', 'far', 'few', 'fifth', 'first', 'five', 'followed', 'following', 'follows', 'for', 'former', 'formerly', 'forth', 'four', 'from', 'further', 'furthermore', 'g', 'get', 'gets', 'getting', 'given', 'gives', 'go', 'goes', 'going', 'gone', 'got', 'gotten', 'greetings', 'h', 'had', "hadn't", 'happens', 'hardly', 'has', "hasn't", 'have', "haven't", 'having', 'he', "he's", 'hello', 'help', 'hence', 'her', 'here', "here's", 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'hi', 'him', 'himself', 'his', 'hither', 'hopefully', 'how', 'howbeit', 'however', 'i', "i'd", "i'll", "i'm", "i've", 'ie', 'if', 'ignored', 'immediate', 'in', 'inasmuch', 'inc', 'indeed', 'indicate', 'indicated', 'indicates', 'inner', 'insofar', 'instead', 'into', 'inward', 'is', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kept', 'know', 'knows', 'known', 'l', 'last', 'lately', 'later', 'latter', 'latterly', 'least', 'less', 'lest', 'let', "let's", 'like', 'liked', 'likely', 'little', 'look', 'looking', 'looks', 'ltd', 'm', 'mainly', 'many', 'may', 'maybe', 'me', 'mean', 'meanwhile', 'merely', 'might', 'more', 'moreover', 'most', 'mostly', 'much', 'must', 'my', 'myself', 'n', 'name', 'namely', 'nd', 'near', 'nearly', 'necessary', 'need', 'needs', 'neither', 'never', 'nevertheless', 'new', 'next', 'nine', 'no', 'nobody', 'non', 'none', 'noone', 'nor', 'normally', 'not', 'nothing', 'novel', 'now', 'nowhere', 'o', 'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'on', 'once', 'one', 'ones', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'own', 'p', 'particular', 'particularly', 'per', 'perhaps', 'placed', 'please', 'plus', 'possible', 'presumably', 'probably', 'provides', 'q', 'que', 'quite', 'qv', 'r', 'rather', 'rd', 're', 'really', 'reasonably', 'regarding', 'regardless', 'regards', 'relatively', 'respectively', 'right', 's', 'said', 'same', 'saw', 'say', 'saying', 'says', 'second', 'secondly', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves', 'sensible', 'sent', 'serious', 'seriously', 'seven', 'several', 'shall', 'she', 'should', "shouldn't", 'since', 'six', 'so', 'some', 'somebody', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'specified', 'specify', 'specifying', 'still', 'sub', 'such', 'sup', 'sure', 't', "t's", 'take', 'taken', 'tell', 'tends', 'th', 'than', 'thank', 'thanks', 'thanx', 'that', "that's", 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', "there's", 'thereafter', 'thereby', 'therefore', 'therein', 'theres', 'thereupon', 'these', 'they', "they'd", "they'll", "they're", "they've", 'think', 'third', 'this', 'thorough', 'thoroughly', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', 'twice', 'two', 'u', 'un', 'under', 'unfortunately', 'unless', 'unlikely', 'until', 'unto', 'up', 'upon', 'us', 'use', 'used', 'useful', 'uses', 'using', 'usually', 'uucp', 'v', 'value', 'various', 'very', 'via', 'viz', 'vs', 'w', 'want', 'wants', 'was', "wasn't", 'way', 'we', "we'd", "we'll", "we're", "we've", 'welcome', 'well', 'went', 'were', "weren't", 'what', "what's", 'whatever', 'when', 'whence', 'whenever', 'where', "where's", 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', "who's", 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'willing', 'wish', 'with', 'within', 'without', "won't", 'wonder', 'would', 'would', "wouldn't", 'x', 'y', 'yes', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves', 'z', 'zero', '',"'s","amp","make","http","www","wiki","arxiv","org"])

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
    clean_space = re.compile('[\n\/,\.\?()_]')
    # <xxx>, $xxx$, not alphabets and space and '_-', \begin xxx \end replaced by ''
    clean_empty = re.compile('<.*?>|\$+[^$]+\$+|[^a-zA-Z\' ]|\\+begin[^$]+\\+end')
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
    corpus_tags = [nltk.pos_tag(nltk.word_tokenize(sentence)) for sentence in corpus]
    return corpus_tags

def process_data(corpus,name):
    # process data
    corpus = clean_corpus(corpus)
    lm = WordNetLemmatizer()
    #using pos tag
    #corpus_pos= generate_corpus_pos(corpus, name)
    #corpus = [" ".join([lm.lemmatize(word[0], get_wordnet_pos(word[1])) for word in sentence])
    #         for sentence in corpus_pos]
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
        features_weighted = 5*features_title + features_content
    elif num == 1:
        features_title = vect.transform(title[start:end]).toarray()
        features_content = vect.transform(content[start:end]).toarray()
        features_weighted = 5*features_title + features_content
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

def extend_abbreviation(mapping, corpus_title):
    return [[word if mapping[word]=='0' else mapping[word] for word in sentence] 
            for sentence in corpus_title]

def construct_phrase_dict(corpus_title):
    phrases = [[word for word in sentence if word.find('-') != -1] for sentence in corpus_title]
    mapping = defaultdict(lambda: '0')
    for sentence in phrases:
        for word in sentence:
            idx = word.find('-')
            abbrev = word[0]
            while idx != -1:
                idx += 1
                abbrev += word[idx]
                idx = word.find('-', idx)
            mapping[abbrev] = word
    return mapping

def filterFromList(results, stop_words):
    for i in range(len(results)):
        results[i] = [ x for x in results[i] if x not in stop_words ]
    return results

def getVect(num):
    my_words = read_words( "stop_words.txt")
    my_stop_words = text.ENGLISH_STOP_WORDS.union(stop_words_2)
    if num == 1:
        vect = TfidfVectorizer(max_df=0.5, min_df=1, analyzer='word', 
                               use_idf=True, stop_words=my_stop_words)
    elif num == 2:
        vect = TfidfVectorizer(max_df=0.5, min_df=1, analyzer='word', token_pattern=None, 
                               use_idf=False, stop_words=my_stop_words, norm='l2', sublinear_tf=True)
    elif num == 3:
        vect = TfidfVectorizer(max_df=0.5, min_df=1, analyzer='word', token_pattern=None,
                               use_idf=True, stop_words=my_stop_words, norm='l2', sublinear_tf=True)
    return vect

def readFromData(filename):
    origin_data = pd.read_csv( filename, quotechar='"', skipinitialspace=True).as_matrix()
    id_ = origin_data[:, 0]
    title  = origin_data[:, 1]
    content= origin_data[:, 2]
    corpus = origin_data[:, 1:3]
    corpus = corpus.astype(object)
    corpus = corpus[:, 0] + " " + corpus[:, 1]
    return id_, title, content, corpus  

def generateOutput(nb_partition, corpus, vect, title, content, featureName):
    feature_arr = []
    partion = int(len(corpus)/nb_partition)
    while(partion ==0 ):
        nb_partition = int(nb_partition/2+1)
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

def deletecomponent(corpus,numremove):
    my_words = read_words( "stop_words.txt")
    my_stop_words = text.ENGLISH_STOP_WORDS.union(stop_words_2)
    corpus = [ removeWordFromStr(sentence, numremove) for sentence in corpus ]
    corpus = [" ".join([word for word in sentence.lower().split(' ')
                    if word not in my_stop_words]) for sentence in corpus]
    return corpus

sys_input3 = sys.argv[3]
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
    corpus = [[w for w in nltk.word_tokenize(sentences.lower())if w != "'s"] for sentences in corpus]
    title = [[w for w in nltk.word_tokenize(sentences.lower()) if w!= "'s"] for sentences in title]
    content = [[w for w in nltk.word_tokenize(sentences.lower()) if w!= "'s"] for sentences in content]
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
    
    for aa in range(800):
        corpus = corpus+sentence
    '''
    bigram = Phrases(corpus,delimiter=b'-')
    bigram_counter = collections.Counter()
    for key in bigram.vocab.keys():
        if len(str(key).split('-')) > 1:
            bigram_counter[key] += bigram.vocab[key]
    for key,counts in bigram_counter.most_common(200):
        #print ('{0: <100} {1}'.format(key, counts))
        sudo = (str(key) + "      "+str(counts))
    title = bigram[title]
    title = [ " ".join(wordlist) for wordlist in title ]
    content = bigram[content]
    content = [ " ".join(wordlist) for wordlist in content ]
    corp = []
    for i in range(len(title)):
        corp.append(  title[i] + " " + content[i] )
    corpus = np.array(corp)
    return bigram,corpus,title,content

if __name__ == '__main__':
    # read from file
    path = sys.argv[1]
    outfileName = sys.argv[2]
    addTop = True
    addThres = False
    n_top, threshold = getOutputVar(addTop, addThres)
    # process data
    id_, title, content, corpus = readFromData(path)
    process_type = int(sys.argv[4])
    corpus, title, content, stemmer = preprocessing(corpus, title, content, process_type)
    #bigram corpus
    corpus = deletecomponent(corpus,2)
    content = deletecomponent(content,2)
    title = deletecomponent(title,2)
    bigram,corpus,title,content = bigramProcess(corpus,title,content)
    print ("Successfully output bigram!")
    #process output
    title_backup = title
    title_tags = [nltk.pos_tag(nltk.word_tokenize(sentence)) for sentence in title]
    #filtered_title = [[tag[0] for tag in sentence if tag[1].startswith('N')] for sentence in title_tags]
    filtered_title=[]
    for ii,sentence in enumerate(title_tags):
        temp = []
        for tag in sentence:
            if(tag[0].find('-') > 0):
                tag_word = tag[0].split('-')
                if len(tag_word[1]) != 0:
                    firstword = tag_word[0]
                    secondword = tag_word[1]
                    print (tag_word)
                    print (secondword)
                    pos_first = nltk.pos_tag([firstword])
                    pos_second = nltk.pos_tag([secondword])
                    print (pos_second)
                    if ((pos_first[0][1].startswith('N') or pos_first[0][1].startswith('J')) 
                            and pos_second[0][1].startswith('N')):
                        temp.append(tag[0])
                else:
                    print (tag[0])
                    firstword = tag_word[0]
                    pos_first = nltk.pos_tag([firstword])
                    if(pos_first[0][1].startswith('N')):
                        temp.append(tag_word[0])
            else:
                if(tag[1].startswith('N')):
                    temp.append(tag[0])
        if len(temp)!=2:
            filtered_title.append(temp)
        else:
            temp = ['homework-and-exercises','quantum-mechanics']
            #temp = nltk.word_tokenize(title_backup[ii])
            filtered_title.append(temp)

    #saveResults(outfileName, id_, filtered_title, stemmer, n_top)
    answer = getResults(filtered_title, id_, n_top)
    for aa in range(len(answer)):
        temp = []
        for iii in answer[aa]:
            if iii not in temp:
                temp.append(iii)
        answer[aa] = temp
    
    writeResults(outfileName, id_, answer)
