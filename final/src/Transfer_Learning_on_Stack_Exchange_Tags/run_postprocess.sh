python3 tag_postprocess_ans.py $1 $2 $3


command <<HERE

./run_postprocess.sh <corpus name> <my answer path> <output file name>
$1: <corpus name>
$2: <my answer path>
$3: <output file name>
e.g.
python3 tag_postprocess_ans.py <corpus name> <my answer path> <output file name>
./run_postprocess.sh corpus/corpus_test testing_results testing_results_post
./run_postprocess.sh corpus/corpus_robotics testing_results testing_results_post
HERE