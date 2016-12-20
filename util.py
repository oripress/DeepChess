import numpy as np
import chess
import chess.pgn
import random
import itertools
import pickle

pieces = {
	'p': 1,
	'P': -1,
	'n': 2,
	'N': -2,
	'b': 3,
	'B': -3,
	'r': 4,
	'R': -4,
	'q': 5,
	'Q': -5,
	'k': 6,
	'K': -6
		}

def shortenString(s):
	s = s[:s.rfind(" ")]
	return s;

def beautifyFEN(f):
	for i in range(4):
		f = shortenString(f)
	
	toMove = f[-1]
	if toMove == 'w':
		toMove = 7
	else:
		toMove = -7

	f = shortenString(f)

	newf = []

	for char in f:
		if char.isdigit():
			for i in range(int(char)):
				newf.append(0)
		elif char != '/':
			#newf.append(('pPnNbBrRqQkK'.find(char)+1))
			newf.append(pieces[char])
	
	newf.append(toMove)
#	print(f)
#	print(newf)
	return	newf

def bitifyFEN(f):
	arrs = []
	result = []
	s = 	{
		'1' : 0,
		'2' : 1,
		'3' : 2,
		'4' : 3,
		'5' : 4,
		'6' : 5,
		'-1' : 6,
		'-2' : 7,
		'-3' : 8,
		'-4' : 9,
		'-5' : 10,
		'-6' : 11,
		}
		 	
	for i in range(12):
		arrs.append(np.zeros(64))

	for i in range(64):
		c = str(int(f[i]))
		if c != '0':
			c = s[c]
		 	#c = s[int(round(c))]
			arrs[c][i] = 1

	for i in range(12):
		result.append(arrs[i])
	
	result = list(itertools.chain.from_iterable(result))
	
	if f[64] == -7:
		result.append(1)
	else:
		result.append(0)
	
	return result

def arrToBin(arr):
	first = arr[0]
	second = arr[1]

	#r1 = np.array_str(first) 
	#r2 = np.array_str(second)

	#r1 = r1.translate(None, '[] ')
	#r2 = r2.translate(None, '[] ')

	#print(r1)
	#for i in range(65):
	#	r1 = r1.join(str(first[i]))
	#	r2 = r2.join(str(second[i]))

	r1 = bitifyFEN(first)
	r2 = bitifyFEN(second)
	return [r1,r2]
	
def convert():
	batches = 0;
	final = []
	final_l = []
	for i in range(24):
		print(i)
		name = 'pGames/volume' + str(batches) + '.p'
		f = open(name)
		data = pickle.load(f) 
		x = data['x']
		x_l = data['x_labels']
		for j in range(25000):
			if j%1000 == 0:	
				print(j)
			final.append(arrToBin(x[j]))
			final_l.append(x_l[j])	
		f.close()
		
		if i > 0 and i%5 == 0:
			temp = open('cvol' + str(i) + '.p', "wb")
			full_data = {"x": final, "x_labels": final_l}
			pickle.dump(full_data, temp)
			temp.close()
			final = []
			final_l = []

	f = open('converted.p', "wb")
	full_data = {"x": final, "x_labels": final_l}
	pickle.dump(full_data, f)
	f.close()
	return

#convert()
#bitifyFEN(beautifyFEN('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e3 0 1'))

