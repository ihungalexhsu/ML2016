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

class Trainer:
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
            #delete punctuation
            s = string.maketrans(string.punctuation, ' '*len(string.punctuation))
            t = ' '.join([line.translate(s) for line in t.split()])
            t = t.decode('utf-8')
            y = ' '.join([word for word in t.split() if word not in cacheStopwords])
            temp.append(y)
        return temp

    def Stem(self,text):
        ps = nltk.stem.SnowballStemmer('english')
        text = [" ".join([ps.stem(word) for word in sentence.split(" ")]) for sentence in text]
        return text

if __name__ == "__main__":
    datapath = sys.argv[1]
    output = sys.argv[2]
    #  Read in the file 
    tp = Trainer(datapath+'title_StackOverflow.txt')
    corpus = tp.readTitle()
    corpus = tp.rmStopwords(corpus)
    corpus = tp.Stem(corpus)
    vectorizer = TfidfVectorizer(max_df=0.5,min_df=2,stop_words='english')
    #vectorizer = TfidfVectorizer(max_df=1.0,min_df=2)
    X = vectorizer.fit_transform(corpus)
    
    #analyze the total tfidf vector
    '''
    feature_names = vectorizer.get_feature_names()
    indices = np.argsort(vectorizer.idf_)
    top_n = 20
    top_features = [feature_names[i] for i in indices[:top_n]]
    print top_features
    '''
    
    print("Processing LSA...")
    svd = TruncatedSVD(n_components=20, n_iter=20, random_state=None )
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    eigentext = lsa.fit_transform(X)
    # Do K-mean
    #print("Computing k-mean...")
    kmeans = KMeans(n_clusters=int(sys.argv[3]), init='k-means++', max_iter=300, n_init=10).fit(eigentext) 
    #print("the distance=", kmeans.inertia_)
    # Read in the test data
    titles = np.asarray(tp.readCheckIndex(datapath+'check_index.csv'))
    # Read Answer lable
    label_ans = []
    with open(datapath+'label_StackOverflow.txt','r') as f:
        label_ans = f.read().splitlines()
    label_ans = np.asarray(map(int,label_ans))
    '''
    #analyze the cluster tfidf vector
    a = Trainer(datapath+'title_StackOverflow.txt')
    cor = a.readTitle()
    cor = a.rmStopwords(cor)
    #cor = a.Stem(cor)
    cor = np.array(cor)
    for i in range(20):
        idx = np.where(kmeans.labels_==i)
        temp = cor[idx]
        vector = TfidfVectorizer(stop_words='english')
        temp = vector.fit_transform(temp)
        feature_names = vector.get_feature_names()
        indices = np.argsort(vector.idf_)
        top_n = 3
        top_features = [feature_names[k] for k in indices[:top_n]]
        print "group "+str(i)+"  "+str(top_features)
    '''
    #Data Visualization
    PCA    
    pca = PCA(n_components=2)
    X_r = pca.fit_transform(eigentext)
    #tsne
    '''
    model = TSNE(n_components=2,random_state=0)
    X_r = model.fit_transform(X_r)
    '''
    
    plt.figure()
    colors = ['navy','turquoise','darkorange','aliceblue','aqua','blue','brown',
              'cyan','darkgray','darksalmon','firebrick', 'gold', 'green','lightpink',
              'lightgoldenrodyellow','maroon','mediumslateblue','olive','purple','tan',
              'antiquewhite','aquamarine','beige','bisque','black','blanchedalmond',
              'burlywood','chocolate','coral','crimson','darkkhaki','lawngreen',
              'lightcyan','lightgray','lemonchiffon','lavender','lavenderblush',
              'midnightblue','mintcream','mistyrose','moccasin','palegreen','orangered'
              ,'navajowhite','orchid','linen','powderblue','plum','papayawhip','peachpuff']
    
    '''
    for color, i in zip(colors, range(20)):
        plt.scatter(X_r[label_ans == i, 0],X_r[label_ans == i, 1],color = color, alpha =0.8)
    plt.legend(loc='best',shadow=False,scatterpoints=1)
    plt.title('PCA of Ans Data Visualization')
    plt.savefig('clusterans.png')
    plt.figure()
    '''
    for color, i in zip(colors, range(50)):
        plt.scatter(X_r[kmeans.labels_ == i ,0],X_r[kmeans.labels_ ==i,1],color=color,alpha = 1.0)
    plt.legend(loc='best',shadow=False,scatterpoints=1)
    plt.title('PCA of Kmeans cluster 50')
    plt.savefig('Kmeans50.png')
    #plt.show()
    ans = []
    print("evaluate the test data...")
    for t in titles:
        if kmeans.labels_[t[0]] != kmeans.labels_[t[1]]:
            ans.append(0)
        else:
            ans.append(1)

    ans = np.asarray(ans)
    print("write file...")
    # Write to the csv 
    f = open(output,'w')
    for row in range(ans.shape[0]+1):
        if(row==0):
            f.write('ID,Ans\n')
        else:
            temp = int(ans[row-1])
            f.write(str(row-1)+','+str(temp)+'\n')

