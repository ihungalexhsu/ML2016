import sys
import numpy as np
import pandas as pd
import pickle


if __name__ == '__main__':
    path = sys.argv[1]

    with open(path, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data = pickle.load(f)
    for k, v in data.items():
        print(k)
    '''
    id_
    content
    corpus
    title = data['title']