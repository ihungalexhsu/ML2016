from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing import sequence
from nltk.stem import WordNetLemmatizer
import seq2seq
from seq2seq.models import Seq2Seq
import pandas as pd
import numpy as np
import string
import sys
import re

datapath = sys.argv[1]
outfileName = sys.argv[2]

def readCorpus(): 
    filename = ['biology.csv','cooking.csv','crypto.csv','diy.csv','robotics.csv','travel.csv']
    totalData = np.empty((0,4),dtype='object')
    for i in range(6):
        origin_data = pd.read_csv(datapath+filename[i], quotechar='"', skipinitialspace=True).as_matrix()
        totalData = np.append(totalData,origin_data,axis=0)
    return totalData

def process_data():
    origin_data = readCorpus()
    # process data
    id_ = origin_data[:, 0]
    title = origin_data[:, 1]
    context = origin_data[:,2]
    tags = origin_data[:,3]
    lm = WordNetLemmatizer()
    title = [re.sub(r'(<[^<]+?>)|(\n)|(\d+)', '', sentence) for sentence in title]
    title = [sentence.translate(sentence.maketrans({key: None for key in (string.punctuation).replace("-","")})) for sentence in title]
    title = [" ".join([lm.lemmatize(word) for word in sentence.split(" ")]) for sentence in title]
    context = [re.sub(r'(<[^<]+?>)|(\n)|(\d+)', '', sentence) for sentence in context]
    context = [sentence.translate(sentence.maketrans({key: None for key in (string.punctuation).replace("-","")})) for sentence in context]
    context = [" ".join([lm.lemmatize(word) for word in sentence.split(" ")]) for sentence in context]
    return (title,context,tags)

if __name__ == '__main__':
    title, context,tags = process_data()
    model = Seq2Seq(
