#coding=utf-8
import sys
import numpy as np
import pandas as pd
import nltk

def generate_corpus_pos(corpus):
    corpus_tags = [nltk.pos_tag(nltk.word_tokenize(sentence)) for sentence in corpus]
    return corpus_tags
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

####################### main function ###########################
def read_words(words_file):
    return [word for line in open(words_file, 'r') for word in line.split()]

def readFromData(filename):
    origin_data = pd.read_csv( filename, quotechar='"', skipinitialspace=True).fillna("none").as_matrix()
    id_ = origin_data[:, 0]
    title  = origin_data[:, 1]
    content= origin_data[:, 2]
    corpus = origin_data[:, 1:3]
    corpus = corpus.astype(object)
    corpus = corpus[:, 0] + " " + corpus[:, 1]
    return id_, title, content, corpus  

def filterRareTags(feature_arr, threshold):
    feature_arr_join = [ " ".join(tags) for tags in feature_arr ]
    feature_arr_split = feature_arr
    wordFreq = wordCount(feature_arr_join)
    sel_tags = []
    for tags in feature_arr_split:
        sel_tags.append( [item for item in tags if wordFreq[item] > threshold] )
    return sel_tags, wordFreq

def wordCount(feature_arr):
    from nltk import FreqDist
    all_tags = " ".join(feature_arr)
    words = nltk.tokenize.word_tokenize(all_tags)
    fdist = FreqDist(words)
    return fdist

if __name__ == '__main__':
    # read from file
    path = sys.argv[1]
    outfileName = sys.argv[2]    
    id_, title, content, corpus = readFromData(path)
    title_tags = [nltk.pos_tag(nltk.word_tokenize(sentence)) for sentence in title]
    filtered_title=[]
    for ii,sentence in enumerate(title_tags):
        #temp = []
        temp = ['quantum-mechanics']
        for tag in sentence:
            if(tag[0].find('-') > 0):
                # accumulate one "-"
                tag_word = tag[0].split('-')
                if len(tag_word[1]) != 0:
                    firstword = tag_word[0]
                    secondword = tag_word[1]
                    pos_first = nltk.pos_tag([firstword])
                    pos_second = nltk.pos_tag([secondword])
                    if ((pos_first[0][1].startswith('N') or pos_first[0][1].startswith('J') 
                         or pos_first[0][0]=='double') and (pos_second[0][1].startswith('N') 
                                                            or pos_second[0][0]=='of')):
                        if len(tag_word) < 4:
                            temp.append(tag[0])
                else:
                    firstword = tag_word[0]
                    pos_first = nltk.pos_tag([firstword])
                    if(pos_first[0][1].startswith('N')):
                        temp.append(tag_word[0])
            else:
                if(tag[1].startswith('N')):
                    temp.append(tag[0])
        if len(temp)!=0:
            filtered_title.append(temp)
        else:
            temp = ['quantum-mechanics']
            filtered_title.append(temp)
    filtered_title, freqlist = filterRareTags(filtered_title, 15)
    answer = getResults(filtered_title, id_, 20)
    for aa in range(len(answer)):
        temp = []
        for iii in answer[aa]:
            if iii not in temp:
                temp.append(iii)
        answer[aa] = temp
    #mostcommon = freqlist.most_common(3000) 
    #for l in range(len(mostcommon)):
    #    print (str(mostcommon[l][0])+","+str(mostcommon[l][1]))
    writeResults(outfileName, id_, answer)
