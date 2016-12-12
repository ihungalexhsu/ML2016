python3 f1_score.py $1 $2


command <<HERE
python3 f1_score.py truth.csv test.csv 
# $1: directory path contains (all_label.p, all_unlabel.p, test.p) 
# $2: input_model
# $3: prediction.csv
# ./test.sh data/ model prediction.csv
HERE
