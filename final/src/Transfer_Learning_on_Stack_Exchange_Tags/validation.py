import subprocess
import sys
import matplotlib.pyplot as plt
import re
import numpy as np

iteration = str(1)

def plot(y):
	print("plot!")
	# plot
	plotX = np.arange( len(y))
	plotY = np.array(y)
	plt.plot(plotX, plotY, linewidth=1.0)
	# plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
	plt.show()

def testing_data2(acc):
	p = subprocess.Popen(['./test', 'modellist.txt', 'testing_data2.txt', 'result2.txt'], \
		stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	out, err = p.communicate()
	file = open("acc.txt", "w")
	file.write(str(acc))
	file.close()

def createCorpus(pyName, data_path, data_list, outputName):
	for i in data_list:
		p = subprocess.Popen(['python3', pyName, data_path + i + '.csv', outputName + i], \
			stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out, err = p.communicate()
		print(out)
		print(err)
	return True

# f1_score.py corpus_biology ...
def getScore(pyName, ans_truth, ans_test):
	p = subprocess.Popen(['python3', pyName, ans_truth, ans_test], \
		stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	out, err = p.communicate()
	out = float( get_num(str(out)) )
	return out

def aveScoreFromDatalist(pyScore, ans_path, ans_ending, outputName_base):
	score = []
	for data in data_list:
		print("...", data, "...")
		score.append( getScore(pyScore, ans_path + data + ans_ending, outputName_base + data) )
	score = np.array(score)
	return score

def runTestFile(run_file, corpus_base, run_argv, data_list, outputName_base):
	for data in data_list:
		run_command = ['python3', run_file, corpus_base+data, outputName_base+data]
		run_command = run_command + run_argv
		p = subprocess.Popen(run_command,
			stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out, err = p.communicate()
		print(out)
		print(err)
	return True

def get_num(text):
	cleanr = re.compile('[^0-9.]')
	cleantext = re.sub(cleanr, '', text)
	return cleantext

def append2file(filename, run_file, run_argv, data_list, score):
	with open(filename, "a") as myfile:
		myfile.write(run_file + " " + " ".join(run_argv) + "\n")
		for i in range(len(data_list)):
			myfile.write(data_list[i] + ' : ' + str(score[i])+ "\n")
		myfile.write("Ave: " + str(score.mean())+ "\n")

if __name__ == '__main__':
	'''
	- usage:
	python3 validation.py <run file name> spec ...
	- example:
	run validation.py tag_gen_ans_tfidf.py vect=2 n_top=3
	'''
	data_list = ['biology', 'cooking', 'crypto', 'diy', 'robotics', 'travel']
	corpus_base = 'ron/corpus_'
	ans_path = '../../ans/'
	ans_ending = '_o.csv'
	pyScore = 'f1_score.py'
	outputName_base = 'val_'
	score = []

	# create corpus
	createCorpus('ron/preprocess_data.py', '../../data/', data_list, corpus_base)

	if len(sys.argv) > 2:
		run_file = sys.argv[1]
		run_argv = sys.argv[2:]
	else:
		print("Usage: <skip input_corpus and output_file_name>")
		print("python3 validation.py <run file name> spec ...")
		print("ex: python3 validation.py tag_gen_ans_tfidf.py vect=2 n_top=3")
		sys.exit()

	# flags ...
	gen_ans = True
	gen_score = True


	if gen_ans:
		print("Generate answer for through data list...")
		runTestFile(run_file, corpus_base, run_argv, data_list, outputName_base)

	if gen_score:
		print("Calculate f1 score for all answers...")
		score = aveScoreFromDatalist(pyScore, ans_path, ans_ending, outputName_base)

		for i in range(len(data_list)):
			print(data_list[i], ' : ',  score[i])
		print("Ave: ", score.mean())
		append2file("val_result", run_file, run_argv, data_list, score)
