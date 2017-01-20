import sys
import numpy as np
import pandas as pd
import pickle

def readFromData(filename):
    origin_data = pd.read_csv( filename, quotechar='"', skipinitialspace=True).as_matrix()
    id_ = origin_data[:, 0]
    title  = origin_data[:, 1]
    content= origin_data[:, 2]
    corpus = origin_data[:, 1:3]
    corpus = corpus.astype(object)
    corpus = corpus[:, 0] + " " + corpus[:, 1]
    return id_, title, content, corpus 


if __name__ == '__main__':
    path = sys.argv[1]
    outfileName = sys.argv[2]
    id_, title, content, corpus = readFromData(path)

    # np.savez("corpus", corpus, title, content)
    data = { 'corpus': corpus,
    'title': title,
    'content': content,
    'id_' : id_ }

    with open(outfileName, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


    '''
    with open('data.pickle', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data = pickle.load(f)
    '''