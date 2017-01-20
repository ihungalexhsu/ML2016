import subprocess
import sys
import re
import numpy as np

def createCorpus(pyName, data_path, data_list, outputName):
    for i in data_list:
        print (i)
        p = subprocess.Popen(['python3', pyName, data_path + i + '.csv', outputName + i], \
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        print (out)
    return True

if __name__ == '__main__':
    '''
    python3 validation.py tag_gen_ans_tfidf.py ... ...
    '''
    data_list = ['biology', 'cooking', 'crypto', 'diy', 'robotics', 'travel']
    data_path = '../../../data/'
    pyName = 'preprocess_data.py'
    outputName = 'corpus_'
    createCorpus(pyName, data_path, data_list, outputName)