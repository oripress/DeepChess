import chess
import tfdeploy as td
import itertools
import copy
from util import *

model = td.Model("model.pkl")
x = model.get("input")
y = model.get("output")

def netPredict(first, second):
	global model
	global x
	global y

	x_1 = bitifyFEN(beautifyFEN(first.fen()))
	x_2 = bitifyFEN(beautifyFEN(second.fen()))
	toEval = [[x_1], [x_2]]
	result = y.eval({x: toEval})

	if result[0][0] > result [0][1]:
		return (first, second)
	else:
		return (second, first)

def alphabeta(node, depth, alpha, beta, maximizingPlayer):
	if depth == 0:
		return node
	if maximizingPlayer:
		v = -1
		for move in node.legal_moves:
			cur = copy.copy(node)
			cur.push(move)
			if v == -1:
				v = cur
				bestMove = move
			else:
				v = netPredict(alphabeta(cur, depth-1, alpha, beta, False), v)[0]
			if alpha == -1:
				alpha = v
			else:
				alpha = netPredict(alpha, v)[0] 
			if beta != 1:
				if netPredict(alpha, beta)[0] == alpha:
					break
		return v 
	else:
		v = 1
		for move in node.legal_moves:
			cur = copy.copy(node)
			cur.push(move)
			if v == 1:
				v = cur
			else:
				v = netPredict(alphabeta(cur, depth-1, alpha, beta, True), v)[1]
			if beta == 1:
				beta = v
			else:
				beta = netPredict(beta, v)[1] 
			if alpha != -1:
				if netPredict(alpha, beta)[0] == alpha:
					break
		return v 

def computerMove(board):
	depth = 3 
	alpha = -1
	beta = 1
	v = -1
	for move in board.legal_moves:
		cur = copy.copy(board)
		cur.push(move)
		if v == -1:
			v = cur
			bestMove = move
		else:
			v = netPredict(alphabeta(cur, depth-1, alpha, beta, False), v)[0]
			if alpha == -1:
				alpha = v
			else:
				newAlpha = netPredict(alpha, v)[0] 
				if newAlpha != alpha:
					bestMove = move
					alpha = newAlpha
	print(bestMove)	
	board.push(bestMove)
	return board

def playerMove(board):
	while True:
		try:
			move = raw_input("Enter your move \n")
			board.push_san(move)
			break
		except ValueError:
			print("Illegal move, please try again")

	return board

def playGame():
	moveTotal = 0;
	board = chess.Board()
	while board.is_game_over() == False:
		print(board)
		if moveTotal % 2 == 1:
			board = playerMove(board)
		else:
			board =	computerMove(board)
		moveTotal = moveTotal+1

	print("Game is over")
		

#firstBoard = bitifyFEN(beautifyFEN(firstBoard.fen()))
#secondBoard = bitifyFEN(beautifyFEN(secondBoard.fen()))
#firstBoard = firstBoard + secondBoard
#for elem in secondBoard:
#	firstBoard.append(elem)
#print(firstBoard)
#result = y.eval({x: firstBoard})
#playGame()
#print(result)i
#computerMove(chess.Board())a
playGame()
