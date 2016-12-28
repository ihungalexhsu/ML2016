python3 tag_preprocess_data.py ../../data/test_small.csv test_corpus pre_type=2
python3 tag_gen_ans.py 	test_corpus  test_test vect=2 n_top=5
python3 tag_postprocess_ans.py test_test output.csv


command <<HERE

*Usage:
python3 tag_preprocess_data.py <input origin csv file> <output processed-data> pre_type=2
python3 tag_gen_ans.py 	<input processed-data>  <output results> vect=2 n_top=5
python3 tag_postprocess_ans.py <input results> <output file name>

ex:
python3 tag_preprocess_data.py ../../data/biology.csv pre_type=2
python3 tag_gen_ans.py 	../../data  biology_test vect=2 n_top=5
python3 tag_postprocess_ans.py fileName output.csv


===========

python3  tag_prediction_tfidf.py ../../data/biology.csv biology_test.csv 7
python3  f1_score.py ../../ans/biology_o.csv biology_test.csv 

python3  tag_prediction_tfidf.py ../../data/biology.csv biology_test.csv 8
python3  f1_score.py ../../ans/biology_o.csv biology_test.csv 

python3  tag_prediction_tfidf.py ../../data/biology.csv biology_test.csv 9
python3  f1_score.py ../../ans/biology_o.csv biology_test.csv 



python3  tag_prediction_tfidf.py ../../data/biology.csv biology_test 6
run  tag_prediction_tfidf.py ../../data/biology.csv biology_test 6

python3  tag_prediction_tfidf.py ../../data/test.csv test_test_3.csv 3
python3  tag_prediction_tfidf.py ../../data/test.csv test_test_4.csv 4
python3  tag_prediction_tfidf.py ../../data/test.csv test_test_5.csv 5
python3  tag_prediction_tfidf.py ../../data/test.csv test_test 6

python3 ref/tfIdf.py ../../data/biology.csv
python3 ref/tfIdf.py ../../data/test.csv
python3 ref/tfidf_2.py ../../data/biology.csv
python3 ref/tfidf_2.py ../../data/biology.csv biology_test.csv

run  tag_prediction_tfidf.py ../../data/biology.csv biology_test.csv 6
run f1_score.py ../../ans/biology_o.csv biology_test.csv

run f1_score.py ../../ans/biology_o.csv tf_idf.csv
run f1_score.py ../../ans/biology_o.csv biology_test.csv
run f1_score.py ../../ans/biology_o.csv biology_test_0.05.csv
run f1_score.py ../../ans/biology_o.csv biology_test_0.1.csv
run f1_score.py ../../ans/biology_o.csv biology_test_0.2.csv
run f1_score.py ../../ans/biology_o.csv biology_test_0.3.csv


python3  tag_prediction_tfidf.py ../../data/biology.csv biology_test.csv 3
python3  f1_score.py ../../ans/biology_o.csv biology_test.csv 

=====
python3  tag_prediction_tfidf.py ../../data/biology.csv biology_test.csv 0.3
python3  f1_score.py ../../ans/biology_o.csv biology_test.csv 

python3  tag_prediction_tfidf.py ../../data/biology.csv biology_test.csv 0.4
python3  f1_score.py ../../ans/biology_o.csv biology_test.csv 

python3  tag_prediction_tfidf.py ../../data/biology.csv biology_test.csv 0.5
python3  f1_score.py ../../ans/biology_o.csv biology_test.csv 

python3  tag_prediction_tfidf.py ../../data/biology.csv biology_test.csv 0.6
python3  f1_score.py ../../ans/biology_o.csv biology_test.csv 

python3  tag_prediction_tfidf.py ../../data/biology.csv biology_test.csv 0.7
python3  f1_score.py ../../ans/biology_o.csv biology_test.csv 

python3  tag_prediction_tfidf.py ../../data/biology.csv biology_test.csv 0.8
python3  f1_score.py ../../ans/biology_o.csv biology_test.csv 

python3  tag_prediction_tfidf.py ../../data/biology.csv biology_test.csv 0.9
python3  f1_score.py ../../ans/biology_o.csv biology_test.csv 

python3  tag_prediction_tfidf.py ../../data/biology.csv biology_test.csv 0.95
python3  f1_score.py ../../ans/biology_o.csv biology_test.csv 

HERE