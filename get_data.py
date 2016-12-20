import numpy as np
import chess
import chess.pgn
import random
import pickle
from util import *

DATA = "data/ccrl.pgn"

BATCH_SIZE = 25000
TOTAL_BATCHES = 24
TWOTIMES_BATCH_SIZE = 50000
'''
BATCH_SIZE = 2 
TOTAL_BATCHES = 24
TWOTIMES_BATCH_SIZE = 4
'''
SPG = 6 

def getGood(node):
	total = 0
	good = []
	while not node.is_end():
		next_node = node.variation(0)
		x = (node.board().san(next_node.move))
		total = total + 1
		if 'x' not in x: 
			if total > 6:
				good.append(total)
		node = next_node
	return good

def traverse(node, moves, arr, curIndex):
	total = 1 
	arrSize = len(arr) 
	board = chess.Board()
	while not node.is_end():
		move = node.main_line().next()
		board.push(chess.Move.from_uci(str(move)))
		if total in moves:
			arr[curIndex] = beautifyFEN(board.fen())
			curIndex = curIndex + 1
			if curIndex == arrSize:
				return curIndex
			moves.remove(total)
			if not moves:
				return curIndex
		next_node = node.variation(0)
		node = next_node
		total = total + 1

	print('bug, shouldnt have reached here')
	return curIndex 

def addGameData(game, arr, curIndex):
	goodmoves = getGood(game)
	picked = []
	for i in range(SPG):
		if not goodmoves:
			break
		nu = random.choice(goodmoves)
		goodmoves.remove(nu)
		picked.append(nu)
	if not picked:
		return curIndex
	curIndex = traverse(game, picked, arr, curIndex)
	return curIndex

def iterateOverFile():
	whiteWins = np.zeros((BATCH_SIZE, 65))
	blackWins = np.zeros((BATCH_SIZE, 65))
	total = 0
	runningTotal = 0
	doneBatches = 0
	whiteIndex = 0
	blackIndex = 0
	
	pgn = open(DATA)

	while doneBatches < TOTAL_BATCHES:	
		g = chess.pgn.read_game(pgn)
		if not g:
			break
		if g.headers["Result"] == "1-0" and whiteIndex < BATCH_SIZE:
			whiteIndex = addGameData(g, whiteWins, whiteIndex)
		elif g.headers["Result"] == "0-1" and blackIndex < BATCH_SIZE:
			blackIndex = addGameData(g, blackWins, blackIndex)

		total = whiteIndex + blackIndex

		if total % 500 == 0:
			print(doneBatches*BATCH_SIZE + total/2)

		if (total > 0 and total % TWOTIMES_BATCH_SIZE == 0):
			name = 'volume' + str(doneBatches) + '.p'
			print(name)
			amount = BATCH_SIZE

			saveP(addLabels((whiteWins, blackWins)), name)		
			doneBatches = doneBatches + 1
			total = 0
			whiteIndex = 0
			blackIndex = 0	


def addLabels(arrs):
	x_labels = np.zeros((BATCH_SIZE,2))
	x = np.zeros((BATCH_SIZE,2,65))

	w = arrs[0]
	b = arrs[1]
	np.random.shuffle(w)
	np.random.shuffle(b)

	for i in range(BATCH_SIZE):
		cur = [w[i],b[i]] 
		label = [1,0]
		
		if random.random() > 0.5:			
		#	cur = np.concatenate((b[i],w[i]), axis=0)
			cur = [b[i],w[i]] 
			label = [0,1]
		
		x[i] = cur
		x_labels[i] = label
	return (x, x_labels)

def saveP(arr, name):
	full_data = {"x": arr[0], "x_labels": arr[1]}
	f = open('pGames/' + name, "wb")
	pickle.dump(full_data, f)
	f.close()

iterateOverFile()

