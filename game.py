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
				v = alphabeta(cur, depth-1, alpha, beta, False) 
			if alpha == -1:
				alpha = v
		
			v = netPredict(v, alphabeta(cur, depth-1, alpha, beta, False))[0]
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
				v = alphabeta(cur, depth-1, alpha, beta, True) 
			if beta == 1:
				beta = v
			
			v = netPredict(v, alphabeta(cur, depth-1, alpha, beta, True))[1]
			beta = netPredict(beta, v)[1] 
			if alpha != -1:
				if netPredict(alpha, beta)[0] == alpha:
					break
		return v 

def computerMove(board, depth):
	alpha = -1
	beta = 1
	v = -1
	for move in board.legal_moves:
		cur = copy.copy(board)
		cur.push(move)
		if v == -1:
			v = alphabeta(cur, depth-1, alpha, beta, False)
			bestMove = move
			if alpha == -1:
				alpha = v
		else:
			new_v = netPredict(alphabeta(cur, depth-1, alpha, beta, False), v)[0]
			if new_v != v:
				bestMove = move
				v = new_v
			alpha = netPredict(alpha, v)[0] 

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
	depth = raw_input("Enter search depth \n")
	depth = int(depth)
	while board.is_game_over() == False:
		print(board)
		if moveTotal % 2 == 1:
			board = playerMove(board)
		else:
			board =	computerMove(board, depth)
		moveTotal = moveTotal+1
	
	print(board)
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
