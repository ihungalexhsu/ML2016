import pandas as pd
import numpy as np
import sys
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans


def readFromData(filename):
    origin_data = pd.read_csv( filename, quotechar='"', skipinitialspace=True).fillna("none").as_matrix()
    id_ = origin_data[:, 0]
    title  = origin_data[:, 1]
    content= origin_data[:, 2]
    corpus = origin_data[:, 1:3]
    corpus = corpus.astype(object)
    corpus = corpus[:, 0] + " " + corpus[:, 1]
    title = [["".join(word) for word in sentence.split(" ")]for sentence in title]
    content = [["".join(word) for word in sentence.split(" ")]for sentence in content]
    corpus = [["".join(word) for word in sentence.split(" ")]for sentence in corpus]
    return id_, title, content, corpus

def readFromAns(filename):
    origin_data = pd.read_csv( filename, quotechar='"', skipinitialspace=True).as_matrix()
    id_ = origin_data[:, 0]
    tags  = origin_data[:, 1].tolist()
    # tags = tags.astype(object)
    tags = [[word for word in sentence.split(" ")]for sentence in tags]
    return id_, tags

def readFromOutput(filename):
    origin_data = pd.read_csv( filename, quotechar='"', skipinitialspace=True).as_matrix()
    id_ = origin_data[:, 0]
    tags  = origin_data[:, 1].tolist()
    tag_list = []
    for sentence in tags:
        tag_list.append(str(sentence).split(" "))
    return id_, tag_list

def plotError(x1, y1, x2=[], y2=[], filename="test.png"):
    import matplotlib
    # matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.pylab as pylab

    fig = plt.figure()
    X = np.arange(len(y1))
    # plt.xlabel('')
    plt.ylabel('Correct rate')

    # plt.title('V1_03_difficult')
    if len(y2) == 0:
        plt.plot(x1, y1)
    else:
        # plt.plot(x1, y1,'ro', x2, y2, 'bs')
        plt.plot(x1, y1, x2, y2)
    plt.savefig(filename,dpi=300)

def plotHist(y1, filename="test.png"):
    import matplotlib
    # matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.pylab as pylab
    import matplotlib.mlab as mlab
    plt.gcf().clear()
    bin_ = np.arange(0, 1.1, 0.1)
    # hist, bin_ = np.histogram(y1, bins=bin_, density=True)
    plt.hist(y1, bins=bin_)
    plt.savefig(filename,dpi=300)

def countAllTagsInTitle(title, tags):
    title_all = []
    for i in range(len(title)):
        title_all = title_all + title[i]
    tag_all = []
    for i in range(len(tags)):
        tag_all = tag_all + tags[i]
    tag_find = []
    for tag in tag_all:
        # if tag in title_all and tag not in tag_find:
        if tag in title_all:
            tag_find.append(tag)

    tag_find_diff = []
    for tag in tag_find:
        if tag not in tag_find_diff:
            tag_find_diff.append(tag)

    title_diff = []
    for title in title_all:
        if title not in title_diff:
            title_diff.append(title)

    tag_diff = []
    for tag in tag_all:
        if tag not in tag_diff:
            tag_diff.append(tag)

    return float(len(tag_find))/float(len(tag_all)), float(len(tag_find_diff))/float(len(tag_diff)), \
           float(len(tag_find_diff))/float(len(title_diff)), float(len(title_diff))

def getStatistics(id_, tags, title, content):
    statistics = []
    for index in range(len(id_)):
        # len(title), len(content), len(tags), num of count_title, num of
        # count_common, num of count_content
        count_title = 0
        count_common = 0
        count_content = 0
        for item in tags[index]:
            flag_title = False
            flag_content = False
            if item in title[index]:
                flag_title = True
            if item in content[index]:
                flag_content = True
            if flag_title:
                count_title+=1
            if flag_content:
                count_content+=1
            if flag_title and flag_content:
                count_common+=1
        #temp = (len(title[index]),len(content[index]),len(tags[index]),
        #        count_title,count_common,count_content)
            temp = (float(count_title)/float(len(title[index])),float(count_content)/float(len(content[index])),
                float(count_common)/float(len(content[index])),float(count_title)/float(len(tags[index])),
                float(count_common)/float(len(title[index])), float(count_title),
                float(count_content),float(count_common), float(len(tags[index])))
            statistics.append(temp)
    statistics = np.array(statistics)
    statistics[ np.isnan(statistics) ] = 0
    statistics.astype(float)
    return statistics

def plotSomeGraph(statistics, name):
    plotError(np.arange(len(statistics)), statistics[:,0], np.arange(len(statistics)), statistics[:,1],
                "tag_title_tag_content_" + str(name) + ".png" )

    plotHist(statistics[:,3], "tagInTitle" + str(name) + ".png")
    # tags in title + tags in content -
    plotHist(np.divide(statistics[:,5]+statistics[:,6]-statistics[:,7],statistics[:,8]), "tagInAll" + str(name) + ".png")

############################################################

def readmyTags(filename):
    files = np.load(filename)
    ans = files['ans']
    return ans

def monogramTags(all_tags):
    return np.array( [ all_tags[i] for i in range(len(all_tags)) if len(all_tags[i]) == 1 ] )

def bigramTags(all_tags):
    return np.array( [ all_tags[i] for i in range(len(all_tags)) if len(all_tags[i]) == 2 ] )

def trigramTags(all_tags):
    return np.array( [ all_tags[i] for i in range(len(all_tags)) if len(all_tags[i]) == 3 ] )

def jointLists(split_tags):
    return np.array( [ "-".join( split_tags[i] ) for i in range(len(split_tags)) ] )

def countDiff(tags):
    tag_diff = []
    for tag in tags:
        if tag not in tag_diff:
            tag_diff.append(tag)
    return tag_diff

def matching(myTags, trueTags):
    arr_match = np.zeros(len(myTags))
    for i in range(len(myTags)):
        if myTags[i] in trueTags:
            arr_match[i] = 1
    return arr_match

def matchingTF(myTags, trueTags):
    rightTags = []
    falseTags = []
    for i in range(len(myTags)):
        if myTags[i] in trueTags and myTags[i] not in rightTags:
            rightTags.append(myTags[i])
        elif myTags[i] not in rightTags:
            falseTags.append(myTags[i])
    return np.array(rightTags), np.array(falseTags)


def readTxt(filename):
    file = open(filename)  # 'r'
    data = file.readlines()
    data = [x.strip() for x in data] # throw away \n
    return np.array( data )

def plotArr(y):
    import matplotlib.pyplot as plt
    x = np.arange(len(y))
    plt.plot(x, y)
    plt.show()

def flatten(list2d):
    return [j for i in list2d for j in i]


############################################################

def read_words(words_file):
    return [word for line in open(words_file, 'r') for word in line.split()]

def tf_idf(corpus):
    my_words = read_words("long_stop_word.txt")
    my_stop_words = text.ENGLISH_STOP_WORDS.union(my_words)
    vect = TfidfVectorizer(max_df=0.5, min_df=2, analyzer='word',
                           use_idf=False, stop_words=my_stop_words)

    # fit vector
    corpus = [" ".join(sentence) for sentence in corpus]
    features = vect.fit_transform(corpus)
    print("Finish feature extraction!")
    print("Size = ", features.shape)

    return features, vect

def lsa(X, n_components=80):
    print("Performing dimensionality reduction using LSA")
    svd = TruncatedSVD(n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(X)

    return X, svd

def kmeans(X, svd, vectorizer, n_clusters=20):
    km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1,
                verbose=0)

    print("Clustering sparse data with %s" % km)
    km.fit(X)

    return km

def get_tags(km, vectorizer, svd, features, n_clusters):
    original_space_centroids = svd.inverse_transform(features)
    order_centroids = original_space_centroids.argsort()[:, ::-1]

    terms = np.asarray(vectorizer.get_feature_names())

    feature_arr = []
    for i in range(n_clusters):
        tmp = []
        tmp.append(terms[order_centroids[i, 0]])
        feature_arr.append(tmp)
    return feature_arr

def cluster(title, content, corpus, n_clusters=500):
    features, vect = tf_idf(corpus)
    features, svd = lsa(features, 80)
    km = kmeans(features, svd, vect, n_clusters)
    feature_arr = get_tags(km, vect, svd, features, n_clusters)

    return feature_arr

if __name__ == '__main__':
    '''
    corpus_base = 'ron/corpus_'
    ans_path = '../../ans/'
    run alex/proportion_tag.py ron/corpus_ ../../ans/
    '''

    if len(sys.argv)== 3 :
        datapath = sys.argv[1]
        anspath = sys.argv[2]
    else:
        print("Usage: python3 proportion_tag.py [datapath] [anspath]")
        print("   e.g.python3 proportion_tag.py corpus_data/corpus_robotics corpus_data/ans/robotics_o.csv")
        sys.exit()

    countTags = False
    plotTags  = True
    workOnList = True
    '''
    if(len(id_)!=len(tags)):
        print("ERROR!!! Data size and answer size mismatch")
        sys.exit()
    '''
    data_list = ['biology', 'cooking', 'crypto', 'diy', 'robotics', 'travel']

    if workOnList:
        for name in data_list:
            print("Start to do ", name)
            id_, title, content, corpus = readFromData(datapath + name)
            _, tags = readFromAns(anspath + name + "_o.csv")
            top_terms = cluster(title, content, corpus)

            if plotTags:
                print("Start to plot ...")
                # statistics = getStatistics(id_, tags, title, content)
                # plotSomeGraph(statistics, name)
                per_get, per_tag, per_right_predict, title_diff = countAllTagsInTitle(top_terms, tags)
                print(name, ": ", per_get, ", ", per_tag, ", ", per_right_predict, ", ", title_diff)

            if countTags:
                print("Start to count tags ...")
                tags1d = flatten(tags)
                tags1d = [ tags1d[i].split("-") for i in range((len(tags1d))) ]
                tags1d = countDiff(tags1d)
                bTags = jointLists( bigramTags(tags1d))
                mTags = jointLists( monogramTags(tags1d))
                tTags = jointLists( trigramTags(tags1d))
                print("all tags, mono tags, bi tags, tri tags : ")
                print(len(tags1d),",",len(mTags),",",len(bTags),",",len(tTags))

            tag_len = np.array( [ len(i) for i in tags ] )
            print(tag_len.min(), ",", tag_len.max(), ",", tag_len.mean())
    else:
        _, tags = readFromOutput(anspath)
        tags1d = flatten(tags)
        tags1d = [ tags1d[i].split("-") for i in range((len(tags1d))) ]
        tags1d = countDiff(tags1d)
        bTags = jointLists( bigramTags(tags1d))
        mTags = jointLists( monogramTags(tags1d))
        tTags = jointLists( trigramTags(tags1d))
        print("all tags, mono tags, bi tags, tri tags : ")
        print(len(tags1d),",",len(mTags),",",len(bTags),",",len(tTags))
    '''
    plotError(np.arange(len(statistics)), statistics[:,0], np.arange(len(statistics)), statistics[:,1])
    plotError(np.arange(len(statistics)), statistics[:,1], np.arange(len(statistics)), statistics[:,2], 'test_23.png')
    plotError(np.arange(len(statistics)), statistics[:,0], np.arange(len(statistics)), statistics[:,4], 'test_15.png')
    result = (statistics[:,1]-statistics[:,2])/statistics[:,1]
    result[ np.isnan(result) ] = 0
    plotError(np.arange(len(statistics)), result, [],[], 'test_2-3_2.png')
    plotError(np.arange(len(statistics)), statistics[:,5], [],[],'test_tT.png')
    plotError(np.arange(len(statistics)), statistics[:,6], [],[], 'test_cT.png')
    plotError(np.arange(len(statistics)), statistics[:,7], [],[], 'test_cC.png')
    plotError(np.arange(len(statistics)), statistics[:,3], [],[], 'test_t_Tag.png')
    plotError(np.arange(len(statistics)), statistics[:,5], np.arange(len(statistics)), statistics[:,6], 'test_tT_cT.png')
    '''
    '''
    plotHist(statistics[:,3])
    '''

