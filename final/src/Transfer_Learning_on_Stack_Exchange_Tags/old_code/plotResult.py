from scipy.misc import imresize
from scipy.misc import imread
from scipy.misc import imsave
import numpy as np
import re
import os
import sys

def get_num(text):
	cleanr = re.compile('[^0-9.]')
	cleantext = re.sub(cleanr, '', text)
	return cleantext


def readTxt(filename):
	file = open(filename)  # 'r'
	data = file.readlines()
	data = [x.strip() for x in data] # throw away \n
	data = [ get_num(sentence) for sentence in data ]
	numbers = []
	for i in data:
		if i[0] == '.':
			numbers.append(i[1:])
		else:
			numbers.append(i)
	'''
	for sentence in data:
		numbers.append(re.findall("\d+\.\d+", sentence)[0])
	# data = [re.findall("\d+\.\d+", sentence) for sentence in data]
	'''
	file.close()
	return np.array(numbers).astype(np.longdouble)
	# return np.array(numbers)

def getAve(arr, seq_length):
	times = int(arr.shape[0]/seq_length)
	ave = arr[:(seq_length*times)]
	ave = ave.reshape(times, seq_length)
	return ave.mean(axis=0)
	# return ave.mean(axis=1)

def plot(loss, filename):
	import matplotlib
	import matplotlib.pyplot as plt

	print("plot!")
	# plot
	plt.xlabel('Times of reduce resolution')
	plt.ylabel('RMSE loss')
	plotX = np.arange( len(loss))
	plotY = np.array(loss)
	N = 50
	plt.plot(plotX, plotY, linewidth=1.0)
	# plt.show()
	plt.savefig(filename,dpi=300)

'''
- This code can extract the result from many folders
- Average the result
- plot the graph 
- save the results
Usage:
python3 analysis/src/plotResults.py analysis/pose/euroc/MH_01_easy 6 analysis/pose/euroc/MH_01_easy/
'''
# python3 plotResults.py folder_name order savePath
#  run plotResults.py ../pose/euroc/MH_01_easy 6
if __name__=="__main__":
	if (len(sys.argv) == 2):
		file_name = sys.argv[1]
	else:
		print("Usage: python3 plotResults.py folder_name order")
	weight = np.array([13196,15404,10432,25918,2771,19279])
	weight = weight/weight.sum()


	# ../pose/euroc/MH_01_easy
	val_result = readTxt(file_name)
	val_result = val_result.reshape(int((len(val_result)/8)), 8)
	val_result[:,7] = np.inner(val_result[:, 1:7], weight)
	np.savetxt('result.csv', val_result.T, delimiter=',', fmt='%.4f')

	# np.savetxt(path_save + '/ave_results', aves, delimiter=',', fmt='%.4f')
	# plot(aves[:,1], path_save + 'test.png')
