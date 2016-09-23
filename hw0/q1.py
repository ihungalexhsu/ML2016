import numpy
import sys

column = int(sys.argv[1])
inputfile = sys.argv[2]
dataset = numpy.loadtxt(inputfile,usecols=(column,))
ans = numpy.sort(dataset)
numpy.savetxt("ans1.txt",ans.flatten(),newline=',')
