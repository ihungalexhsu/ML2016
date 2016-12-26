
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
from numpy import genfromtxt
from sklearn.feature_extraction import text
import bottleneck as bn # sorting
import re

stop_words_2 = set(['螳螂捕蝉', '黄雀在后', 'a', "a's", 'able', 'about', 'above', 'according', 'accordingly', 'across', 'actually', 'after', 'afterwards', 'again', 'against', "ain't", 'all', 'allow', 'allows', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'apart', 'appear', 'appreciate', 'appropriate', 'are', "aren't", 'around', 'as', 'aside', 'ask', 'asking', 'associated', 'at', 'available', 'away', 'awfully', 'b', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'believe', 'below', 'beside', 'besides', 'best', 'better', 'between', 'beyond', 'both', 'brief', 'but', 'by', 'c', "c'mon", "c's", 'came', 'can', "can't", 'cannot', 'cant', 'cause', 'causes', 'certain', 'certainly', 'changes', 'clearly', 'co', 'com', 'come', 'comes', 'concerning', 'consequently', 'consider', 'considering', 'contain', 'containing', 'contains', 'corresponding', 'could', "couldn't", 'course', 'currently', 'd', 'definitely', 'described', 'despite', 'did', "didn't", 'different', 'do', 'does', "doesn't", 'doing', "don't", 'done', 'down', 'downwards', 'during', 'e', 'each', 'edu', 'eg', 'eight', 'either', 'else', 'elsewhere', 'enough', 'entirely', 'especially', 'et', 'etc', 'even', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'f', 'far', 'few', 'fifth', 'first', 'five', 'followed', 'following', 'follows', 'for', 'former', 'formerly', 'forth', 'four', 'from', 'further', 'furthermore', 'g', 'get', 'gets', 'getting', 'given', 'gives', 'go', 'goes', 'going', 'gone', 'got', 'gotten', 'greetings', 'h', 'had', "hadn't", 'happens', 'hardly', 'has', "hasn't", 'have', "haven't", 'having', 'he', "he's", 'hello', 'help', 'hence', 'her', 'here', "here's", 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'hi', 'him', 'himself', 'his', 'hither', 'hopefully', 'how', 'howbeit', 'however', 'i', "i'd", "i'll", "i'm", "i've", 'ie', 'if', 'ignored', 'immediate', 'in', 'inasmuch', 'inc', 'indeed', 'indicate', 'indicated', 'indicates', 'inner', 'insofar', 'instead', 'into', 'inward', 'is', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kept', 'know', 'knows', 'known', 'l', 'last', 'lately', 'later', 'latter', 'latterly', 'least', 'less', 'lest', 'let', "let's", 'like', 'liked', 'likely', 'little', 'look', 'looking', 'looks', 'ltd', 'm', 'mainly', 'many', 'may', 'maybe', 'me', 'mean', 'meanwhile', 'merely', 'might', 'more', 'moreover', 'most', 'mostly', 'much', 'must', 'my', 'myself', 'n', 'name', 'namely', 'nd', 'near', 'nearly', 'necessary', 'need', 'needs', 'neither', 'never', 'nevertheless', 'new', 'next', 'nine', 'no', 'nobody', 'non', 'none', 'noone', 'nor', 'normally', 'not', 'nothing', 'novel', 'now', 'nowhere', 'o', 'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'on', 'once', 'one', 'ones', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'own', 'p', 'particular', 'particularly', 'per', 'perhaps', 'placed', 'please', 'plus', 'possible', 'presumably', 'probably', 'provides', 'q', 'que', 'quite', 'qv', 'r', 'rather', 'rd', 're', 'really', 'reasonably', 'regarding', 'regardless', 'regards', 'relatively', 'respectively', 'right', 's', 'said', 'same', 'saw', 'say', 'saying', 'says', 'second', 'secondly', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves', 'sensible', 'sent', 'serious', 'seriously', 'seven', 'several', 'shall', 'she', 'should', "shouldn't", 'since', 'six', 'so', 'some', 'somebody', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'specified', 'specify', 'specifying', 'still', 'sub', 'such', 'sup', 'sure', 't', "t's", 'take', 'taken', 'tell', 'tends', 'th', 'than', 'thank', 'thanks', 'thanx', 'that', "that's", 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', "there's", 'thereafter', 'thereby', 'therefore', 'therein', 'theres', 'thereupon', 'these', 'they', "they'd", "they'll", "they're", "they've", 'think', 'third', 'this', 'thorough', 'thoroughly', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', 'twice', 'two', 'u', 'un', 'under', 'unfortunately', 'unless', 'unlikely', 'until', 'unto', 'up', 'upon', 'us', 'use', 'used', 'useful', 'uses', 'using', 'usually', 'uucp', 'v', 'value', 'various', 'very', 'via', 'viz', 'vs', 'w', 'want', 'wants', 'was', "wasn't", 'way', 'we', "we'd", "we'll", "we're", "we've", 'welcome', 'well', 'went', 'were', "weren't", 'what', "what's", 'whatever', 'when', 'whence', 'whenever', 'where', "where's", 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', "who's", 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'willing', 'wish', 'with', 'within', 'without', "won't", 'wonder', 'would', 'would', "wouldn't", 'x', 'y', 'yes', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves', 'z', 'zero', ''])

def process_data(corpus):

    # process data
    lm = WordNetLemmatizer()
    corpus = [re.sub(r'(<[^<]+?>)|(\n)|(\d+)', '', sentence) for sentence in corpus]
    corpus = [sentence.translate(sentence.maketrans({key: None for key in (string.punctuation).replace("-","") }))
              for sentence in corpus]
    corpus = [" ".join([lm.lemmatize(word) for word in sentence.split(" ")]) for sentence in corpus]
    # print(corpus)
    # tags   = origin_data[:, 3]
    return corpus

def saveResults(outfileName, id_, result, n_tags=3):
	ofile = open(outfileName + '.csv', "w")
	ofile.write('\"id\",\"tags\"\n')
	for i in range(len(id_)):
		if len(result[i]) > n_tags:
			arr = result[i][:n_tags]
		else:
			arr = result[i]
		ofile.write( '"' + str(id_[i]) + '"' + "," + '"' + str(" ".join(arr)) + '"' + '\n' )
	ofile.close()
	return True

def read_words(words_file):
    return [word for line in open(words_file, 'r') for word in line.split()]

def lsa(X, n_components=80):
    print("Performing dimensionality reduction using LSA")
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

def getfeaturesWeighted(vect, title, content, start, end):
	features_title = vect.transform(title[start:end]).toarray()
	# features_title, svd = lsa(features_title, 80)
	features_content = vect.transform(content[start:end]).toarray()
	# features_content, svd = lsa(features_content, 80)
	features_weighted = 8*features_title + features_content
	return features_weighted

def getFeaturearr(feature_arr, corpus, features_weighted, featureName, addThres, threshold, n_top):
	for i in range(len(features_weighted)):
		# selectedFeature = svd.inverse_transform(features_weighted[i].reshape(1,-1))
		# print(selectedFeature.shape)
		selectedFeature = features_weighted[i]
		
		# arg = selectedFeature.argsort()[-1*n_top:][::-1]
		arg = selectedFeature.argsort()[-1*n_top:][::-1]
		if addThres == True:
			tops = tagsThreshold(threshold, selectedFeature[arg], n_top)
		else:
			tops = n_top
		# arg = arg[-1*n_top:]

		# arg = bottleneck.argpartsort(-a, 10)[:10]
		feature_arr.append( featureName[arg[:tops]] )
	return feature_arr

def filterFromList(results, stop_words):
	for i in range(len(results)):
		results[i] = [ x for x in results[i] if x not in stop_words ]
	return results

def getVect(num):
	if num == 1:
		vect = TfidfVectorizer(max_df=0.5, min_df=1, analyzer='word', 
		   		use_idf=True, stop_words=my_stop_words)
	elif num == 2:
		vect = TfidfVectorizer(max_df=0.5, min_df=1, analyzer='word', 
		   		use_idf=False, stop_words=my_stop_words)
	return vect

def filterBeforeOutput():
	saveResults('biology_test_0.05.csv', id_, filterFromList(feature_arr, my_data[:int(len(my_data)*0.05)]) )
	saveResults('biology_test_0.1.csv', id_, filterFromList(feature_arr, my_data[:int(len(my_data)*0.1)]) )
	saveResults('biology_test_0.2.csv', id_, filterFromList(feature_arr, my_data[:int(len(my_data)*0.2)]) )
	saveResults('biology_test_0.3.csv', id_, filterFromList(feature_arr, my_data[:int(len(my_data)*0.3)]) )
	saveResults(outfileName + str(3), id_, feature_arr, 3)
	saveResults(outfileName + str(4), id_, feature_arr, 4)
	saveResults(outfileName + str(5), id_, feature_arr, 5)



if __name__ == '__main__':
	# read from file
	path = sys.argv[1]
	outfileName = sys.argv[2]
	origin_data = pd.read_csv( path, quotechar='"', skipinitialspace=True).as_matrix()

	# process data
	id_ = origin_data[:, 0]

	title  = origin_data[:, 1]
	content= origin_data[:, 2]
	corpus = origin_data[:, 1:3]
	corpus = corpus.astype(object)
	corpus = corpus[:, 0] + " " + corpus[:, 1]
	# corpus = [re.sub(r'\d+', '', word) for word in corpus]

	# tags   = origin_data[:, 3]
	corpus = process_data(corpus)
	title  = process_data(title)
	content  = process_data(content)

	# define vector
	my_words = read_words( "stop_words.txt")
	# my_data = pd.read_csv( 'stop_word_list_5000.csv', delimiter=',', skipinitialspace=True).as_matrix()
	# my_data = my_data[:,1].tolist()
	# nb_stopwords = int(len(my_data)*0.5)
	# my_data = my_data[:nb_stopwords]
	# my_stop_words = text.ENGLISH_STOP_WORDS.union(my_words)
	my_stop_words = text.ENGLISH_STOP_WORDS.union(stop_words_2)
	## bug
	# my_stop_words = my_stop_words.union(stop_words_2)
	# my_stop_words = my_stop_words.union(my_data)

	vect = getVect(2)

	# fit vector
	# generate output
	features = vect.fit(corpus)
	feature_arr = []

	n_top = int(6)
	addTop = True
	if addTop:
		n_top = int(sys.argv[3])

	weights = np.array( vect.idf_ )
	featureName = np.array( vect.get_feature_names() )
	addThres = False

	if addThres:
		threshold = float(sys.argv[3])
	else:
		threshold = 1


	print("Start to generate output!")
	nb_partition = 100
	partion = int(len(corpus)/nb_partition)
	# threshold = 0.8
	count = 0
	for i in range(nb_partition):
		features_weighted = getfeaturesWeighted(vect, title, content, partion*i, partion*(i+1))
		feature_arr = getFeaturearr(feature_arr, corpus[partion*i: partion*(i+1)], features_weighted, 
			featureName, addThres, threshold, n_top)
		if i == nb_partition-1:
			features_weighted = getfeaturesWeighted(vect, title, content, partion*i, len(corpus))
			feature_arr = getFeaturearr(feature_arr, corpus[partion*i: len(corpus)], features_weighted, 
				featureName, addThres, threshold, n_top)
		print("Part: ", i+1, "/", nb_partition)
		# print("features: ", len(feature_arr))
	print("Finish generating output!")
	# save to files

	# feature_arr = filterFromList(feature_arr, my_data)

	saveResults(outfileName + str(n_top), id_, feature_arr, n_top)


	# saveResults(biology_test_0.3.csv, id_, filterFromList(feature_arr, my_data[:len(my_data)*0.4]) )
	# saveResults(outfileName, id_, filterFromList(feature_arr, my_data[:len(my_data)*0.5]) )

	print("Finish save to file!")

	'''
	for i in range(5):
		selectedFeature = features_weighted[i]
		arg = selectedFeature.argsort()[-1*n_top:][::-1]
		selected = np.concatenate( (featureName[arg], weights[arg]), axis=0)
		selected = selected.reshape((2,n_top)).T
		print(selected)
		feature_arr.append(selected)
	'''

