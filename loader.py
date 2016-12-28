import numpy as np
import pickle
import random
from util import *

def getTest(input_size, start, finish):
	test = []
	test_l = []

	for i in range(start, finish):
		name = 'pGames/volume' + str(i) + '.p'
		t = open(name)
		cur_dict = pickle.load(t)	
		cur_test = cur_dict['x']
		cur_l = cur_dict['x_labels']
		for j in range(len(cur_test)):
			test.append(cur_test[j])
			test_l.append(cur_l[j])
		t.close()

	temp_test = np.zeros((len(test), 2, input_size))

	for i in range(len(test)):
		first = bitifyFEN(test[i][0])	
		second = bitifyFEN(test[i][1])	
		elem = [first,second]
		temp_test[i] = elem

	return (temp_test, test_l)

def getTrain(input_size, total, volume_size):
	whiteWins = np.zeros((total, input_size))
	blackWins = np.zeros((total, input_size))

	for i in range(total/volume_size):
		print("Loading batch number " + str(i))
		f = open('pGames/volume' + str(i) + '.p')
		full_data = pickle.load(f)
		curX = full_data['x']
		curX = np.array(curX)
		curL = full_data['x_labels']
		curL = np.array(curL)
		f.close()
				
		for j in range(volume_size):
			if curL[j][0] == 1:
				first = bitifyFEN(curX[j][0])	
				second = bitifyFEN(curX[j][1])
			else:
				first = bitifyFEN(curX[j][1])	
				second = bitifyFEN(curX[j][0])

			whiteWins[i*volume_size+j] = first
			blackWins[i*volume_size+j] = second 
	
	return (whiteWins, blackWins)
