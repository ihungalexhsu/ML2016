import sys

vect_type = []
for i in range(len(sys.argv)):
	li = sys.argv[i].split("=")
	print(li)
	if li[0] == "vect":
		vect_type = int(li[1])
	elif: li[0] == "n_top":
		n_top = int(li[1])
print(vect_type)

# python3 parse_argv.py 
# python3 bigram_corpus_alex.py data/test.csv output (n_top) (type1) (tfidf)
# run parse_argv.py in=data/test.csv out=output n_top=6 pre_type=1 vect=2