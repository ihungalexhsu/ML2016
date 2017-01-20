python3 tag_gen_ans_tfidf.py $1 $2 $3 $4 $5


command <<HERE
./run_tfidf.sh $1 $2 $3 $4 $5
$1: <preprocessed corpus file>
$2: <output file name>
$3: <vect type (1-7)>
$4: <number of answers>
$5: <title: content weighting>
e.g.
./run_tfidf.sh corpus/corpus_robotics testing_results vect=2 n_top=3 weight=8:1
HERE