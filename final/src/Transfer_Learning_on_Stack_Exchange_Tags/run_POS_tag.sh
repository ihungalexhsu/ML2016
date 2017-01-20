python3 tag_gen_ans_POS_tagging.py $1 $2 

command <<HERE
./run_POS_tag.sh $1 $2
$1: <preprocessed corpus file>
$2: <output file name>

e.g.
python3 tag_gen_ans_POS_tagging.py corpus/corpus_test testing
./run_POS_tag.sh corpus/corpus_test testing_results
HERE