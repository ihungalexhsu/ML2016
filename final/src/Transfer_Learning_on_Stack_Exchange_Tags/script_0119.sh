
python3 validation.py  tag_gen_ans_tfidf.py vect=2 n_top=3 weight=1:1
python3 validation.py  tag_gen_ans_tfidf.py vect=2 n_top=3 weight=8:1
python3 validation.py  tag_gen_ans_tfidf.py vect=2 n_top=3 weight=1:8


command <<HERE

python3 validation.py tag_gen_ans_tfidf.py vect=2 n_top=4
python3 validation.py tag_gen_ans_tfidf.py vect=2 n_top=5

python3 validation.py tag_gen_ans_tfidf.py vect=2 n_top=3
python3 validation.py tag_gen_ans_tfidf.py vect=5 n_top=3
python3 validation.py tag_gen_ans_tfidf.py vect=6 n_top=3

HERE
