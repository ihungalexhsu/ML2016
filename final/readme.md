#Final Project for NTU Mechine Learning Course

## Files structure
- ans
	- biology_o.csv
	- cooking_o.csv
	- crypto_o.csv
	- diy_o.csv
	- robotics_o.csv
	- travel_o.csv
- src
	- Transfer_Learning_on_Stack_Exchange_Tags
		- data
			- Original csv files
		- old_code
			- Put some old codes here ...
		- corpus
			- Preprocessed data corpus for all subjects
		- preprocess.py
		- tag_gen_ans_tfidf.py
		- chaoAn.py
- readme.md

## How to execute the program?

### Data preprocessing
- Run preprocess data
	- ./run_preprocess.sh $1 $2
	- $1: < file to do preprocess\>
	- $2: < save file name\>
	- e.g.
	- ./run_preprocess.sh ../../data/test.csv corpus/corpus_test
```
./run_preprocess.sh ../../data/test.csv corpus/corpus_test
```
Run preprocessed data on all subjects
# [biology', 'cooking', 'crypto', 'diy', 'robotics', 'travel']
```
./run_preprocess.sh ../../data/biology.csv corpus/corpus_biology
./run_preprocess.sh ../../data/cooking.csv corpus/corpus_cooking
./run_preprocess.sh ../../data/crypto.csv corpus/corpus_crypto
./run_preprocess.sh ../../data/diy.csv corpus/corpus_diy
./run_preprocess.sh ../../data/robotics.csv corpus/corpus_robotics
./run_preprocess.sh ../../data/travel.csv corpus/corpus_travel
```
### Generate answer

- Run TF-IDF
	- ./run_tfidf.sh $1 $2 $3 $4 $5
	- $1: <preprocessed corpus file>
	- $2: <output file name>
	- $3: <vect type (1-7)>
	- $4: <number of answers>
	- $5: <title: content weighting>
	- e.g.
	- ./run_tfidf.sh corpus/corpus_robotics testing_results vect=2 n_top=3 weight=8:1
```
./run_tfidf.sh corpus/corpus_test test_tfidf vect=2 n_top=3 weight=8:1
```

- Run Title-only
	- ./run_POS_tag.sh $1 $2
	- $1: <preprocessed corpus file>
	- $2: <output file name>
	- e.g.
	- python3 tag_gen_ans_POS_tagging.py corpus/corpus_test testing
	- ./run_POS_tag.sh corpus/corpus_test testing_results

```
./run_POS_tag.sh corpus/corpus_test testing_results
```

### Post-processing data
- Run post-process
	- ./run_postprocess.sh <corpus name> <my answer path> <output file name>
	- $1: <corpus name>
	- $2: <my answer path>
	- $3: <output file name>
```
./run_postprocess.sh corpus/corpus_test testing_results testing_results_post
```


### Validation
To Run validation, you have to specify the file you run and other parameters.
For example, if I want to test by 'tag_gen_ans_tfidf.py' with parameters 'vect=2 n_top=3 weight=8:1'
I should type:
```
python3 validation.py  tag_gen_ans_tfidf.py vect=2 n_top=3 weight=8:1
```















### Some update record

=======================
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
