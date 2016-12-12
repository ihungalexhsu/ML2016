from numpy import genfromtxt
import pandas as pd
import sys

def saveResults(outfileName, result):
	y = result
	ofile = open(outfileName, "w")
	ofile.write('"id","tags"\n')
	for i in range(len(y)):
		ofile.write( str(i) + "," + str(int(y[i])) + '\n' )
	ofile.close()
	return True

# ans_truth = genfromtxt( sys.argv[1], quotechar='"') 
file_name = sys.argv[1]
ans_truth = pd.read_csv(file_name, quotechar='"', skipinitialspace=True)

ans_truth = ans_truth[['id', 'tags']]

outfile_name = file_name[:-4] + '_o' + file_name[-4:]
# ans_truth.to_csv(file_name, columns=[['id', 'tags']], quotechar='"', index=False, encoding='utf-8')
ans_truth.to_csv(outfile_name, quoting=1, quotechar='"', index=False, encoding='utf-8')

# ans_truth = ans_truth[:, [0, 3]]

