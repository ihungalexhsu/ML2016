import pandas as pd
import sys


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
    tags  = origin_data[:, 1]
    tags = [["".join(word) for word in sentence.split(" ")]for sentence in tags]
    return id_, tags  

if __name__ == '__main__':
    if len(sys.argv)== 3 :
        datapath = sys.argv[1]
        anspath = sys.argv[2]
    else:
        print("Usage: python3 proportion_tag.py [datapath] [anspath]")
        print("   e.g.python3 proportion_tag.py corpus_data/corpus_robotics corpus_data/ans/robotics_o.csv")
        sys.exit()
    id_, title, content, corpus = readFromData(datapath)
    _, tags = readFromAns(anspath)
    
    if(len(id_)!=len(tags)):
        print("ERROR!!! Data size and answer size mismatch")
        sys.exit()
    
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
                float(count_common)/float(len(content[index])),float(count_title)/float(len(tags[index])))
        statistics.append(temp)

