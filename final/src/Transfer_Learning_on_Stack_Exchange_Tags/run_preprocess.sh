python3 preprocess_data.py $1 $2


command <<HERE
python3 preprocess_data.py ../../data/robotics.csv corpus/testing
python3 preprocess_data.py ../../data/test.csv corpus/corpus_test

./run_preprocess.sh $1 $2
$1: <file to do preprocess>
$2: <save file name>
e.g.
./run_preprocess.sh ../../data/test.csv corpus/corpus_test
./run_preprocess.sh ../../data/robotics.csv corpus/testing
HERE