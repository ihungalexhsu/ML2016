from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.random_projection import sparse_random_matrix
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import nltk.stem
import numpy as np
import csv
import string
import sys

if __name__ == "__main__":
    datapath = sys.argv[1]
    output = sys.argv[2]
    #  Read in the file 
    corpus = []
    with open(datapath+'title_StackOverflow.txt') as f:
        corpus = f.read().splitlines()
    #remove stopwords from nltk
    temp = []
    nltkStopwords = stopwords.words("english")
    for t in corpus:
        #delete punctuation
        rmpunc = string.maketrans(string.punctuation, ' '*len(string.punctuation))
        trans = ' '.join([line.translate(rmpunc) for line in t.split()])
        trans = trans.decode('utf-8')
        y = ' '.join([word for word in trans.split() if word not in nltkStopwords])
        temp.append(y)
    corpus = temp
    #use snowballstemmer
    temp = nltk.stem.SnowballStemmer('english')
    corpus = [" ".join([temp.stem(word) for word in sentence.split(" ")]) for sentence in corpus]
    
    Tfidf = TfidfVectorizer(max_df=0.5,min_df=2,stop_words='english')
    X = Tfidf.fit_transform(corpus)
    
    #analyze the total tfidf vector
    '''
    feature_names = Tfidf.get_feature_names()
    indices = np.argsort(vectorizer.idf_)
    top_n = 20
    top_features = [feature_names[i] for i in indices[:top_n]]
    print top_features
    '''
    
    #Processing LSA
    svd = TruncatedSVD(n_components=20, n_iter=20, random_state=None )
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    eigentext = lsa.fit_transform(X)
    kmeans = KMeans(n_clusters=100, init='k-means++', max_iter=300, n_init=15).fit(eigentext) 
    
    #read check index
    checkIndex = []
    file = open(datapath+'check_index.csv','r')
    for row in csv.reader(file):
        data = row[1:]
        checkIndex.append(data)
    checkIndex = checkIndex[1:]
    for i in range(len(checkIndex)):
        checkIndex[i] = map(int,checkIndex[i])
    titles = np.asarray(checkIndex)
    #Data Visualization
    PCA    
    pca = PCA(n_components=2)
    X_r = pca.fit_transform(eigentext)
    #tsne
    '''
    model = TSNE(n_components=2,random_state=0)
    X_r = model.fit_transform(X_r)
    '''
    ans = []
    print("evaluate the test data...")
    for t in titles:
        if kmeans.labels_[t[0]] != kmeans.labels_[t[1]]:
            ans.append(0)
        else:
            ans.append(1)

    ans = np.asarray(ans)
    f = open(output,'w')
    for row in range(ans.shape[0]+1):
        if(row==0):
            f.write('ID,Ans\n')
        else:
            temp = int(ans[row-1])
            f.write(str(row-1)+','+str(temp)+'\n')

