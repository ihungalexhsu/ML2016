#Final Project for NTU Mechine Learning Course

12/10 Eric
add files: 
train.sh 

test.sh 

tag_prediction.py (use tf/idf to find tags)
- usage
	python3 tag_prediction.py <input file> <output file>
	python3 tag_prediction.py ../../data/test.csv test_test.csv

f1_score.py 
- usage
	python3 f1_score biology_o.csv biology_test.csv
- input answer file and my testing file
- to get the score of validation (done)

get_ans_truth.py
- usage
	python3 get_ans_truth.py data/biology.csv
- input .csv files and extract the columns of id and tags (finished)
- used for validation
execution.txt ( put some useful commands inside)

*Note:
This data should be read in by panda library because of the format.
Please refer to 'get_ans_truth.py'
