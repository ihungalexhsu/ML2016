from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from nltk.corpus import stopwords
import nltk.stem
import numpy as np
import csv
import string
import sys

class TextProcessor:
    def __init__(self, filename):
        self.filename = filename
        self.checkIndex = []

    def readTitle(self):
        corpus = []
        with open(self.filename) as f:
            corpus = f.read().splitlines()
        return corpus

    def readCheckIndex(self, check_index):
        f = open(check_index,'r')
        for row in csv.reader(f):
            row = row[1:]
            self.checkIndex.append(row)
        self.checkIndex = self.checkIndex[1:]
        for i in range(len(self.checkIndex)):
            self.checkIndex[i] = map(int,self.checkIndex[i])
        return self.checkIndex

    def rmStopwords(self,text):
        temp = []
        cacheStopwords = stopwords.words("english")
        
        for t in text:
            s = string.maketrans(string.punctuation, ' '*len(string.punctuation))
            t = ' '.join([line.translate(s) for line in t.split()])
            t = t.decode('utf-8')
            y = ' '.join([word for word in t.lower().split() if word not in cacheStopwords])
            #print(y)
            temp.append(y)
        return temp

    def Stem(self,text):
        ps = nltk.stem.SnowballStemmer('english')
        text = [" ".join([ps.stem(word) for word in sentence.split(" ")]) for sentence in text]
        return text

    def u2ascii(self,text):
        text = [" ".join([word.encode("ascii","ignore") for word in sentence.split(" ")]) for sentence in text]


if __name__ == "__main__":
    datapath = sys.argv[1]
    output = sys.argv[2]
    #  Read in the file 
    #filename = 'title_StackOverflow.txt'
    tp = TextProcessor(datapath+'/title_StackOverflow.txt')
    corpus = tp.readTitle()
    corpus = tp.rmStopwords(corpus)
    corpus = tp.Stem(corpus)
    vectorizer = TfidfVectorizer(max_df=0.5,min_df=2,stop_words='english')

    X = vectorizer.fit_transform(corpus)
    print("X shape = ",X.shape)
    print("Processing svd...")
    svd = TruncatedSVD(n_components=20, n_iter=20, random_state=None )
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    eigentext = lsa.fit_transform(X)
    #(len(eigentext))
    # Do K-mean
    #
    print("Computing k-mean...")
    kmeans = KMeans(n_clusters=100, init='k-means++', max_iter=300, n_init=10).fit(eigentext)
    print("the distance=", kmeans.inertia_)
    #exit(-1)
    # Read in the test data
    titles = np.asarray(tp.readCheckIndex(datapath+'/title_StackOverflow'))
    ans = []
    print("evaluate the test data...")
    for t in titles:
        if kmeans.labels_[t[0]] != kmeans.labels_[t[1]]:
            ans.append(0)
        else:
            ans.append(1)
    ans = np.asarray(ans)
    print(ans.shape)
    # Write to the csv 
    f = open(output,'w')
    for row in range(ans.shape[0]+1):
        if(row==0):
            f.write('ID,Ans\n')
        else:
            temp = int(ans[row-1])
            f.write(str(row-1)+','+str(temp)+'\n')

    

    #print(X.shape)
