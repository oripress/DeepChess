import tensorflow as tf
import tfdeploy as td
import numpy as np
import math
from loader import *

TRAIN_AUTOENCODER = 0 
TRAIN_NET = 1

TOTAL_AE = 250000
TOTAL_MLP = 750000

BS_AE = 20
BS_MLP = 50
EPOCHS_AE = 50 
EPOCHS_MLP = 201 
RATE_AE = 0.005
DECAY_AE = 0.98
RATE_MLP = 0.005
DECAY_MLP = 0.98

BIAS = 0.15

N_INPUT = 769 
ENCODING_1 = 600 
ENCODING_2 = 400 
ENCODING_3 = 200
ENCODING_4 = 100

HIDDEN_1 = 200
HIDDEN_2 = 400 
HIDDEN_3 = 200
HIDDEN_4 = 100 
N_OUT = 2

VOLUME_SIZE = 25000

export_path = 'net/exports'

#Get the data from the game files
validation_test, validation_test_l = getTest(N_INPUT, 40, 44)
whiteWins, blackWins = getTrain(N_INPUT, TOTAL_MLP, VOLUME_SIZE)

# init
def weight_variable(n_in, n_out):
  cur_dev = math.sqrt(3.0/(n_in+n_out))
  initial = tf.truncated_normal([n_in, n_out], stddev=cur_dev)
  return tf.Variable(initial)

def bias_variable(n_out):
  initial = tf.constant(BIAS, shape=[n_out])
  return tf.Variable(initial)

def getBatchMLP(start, size):
	global whiteWins 
	global blackWins 

	xR = []
	lR = []
	
	for i in range(start,start+size):
		if random.random() > 0.5:
			elem = [whiteWins[i], blackWins[i]]
			elem_l = [1,0]
		else:
			elem = [blackWins[i], whiteWins[i]]
			elem_l = [0,1]
		xR.append(elem)
		lR.append(elem_l)
	return (xR, lR)

def getBatchAE(start, size):
	global whiteWins 
	global blackWins 
	
	size = size/2
	start = start*size
	xR = []
	for i in range(start,start+size):
		xR.append(whiteWins[i])
		xR.append(blackWins[i])
		random.shuffle(xR)
	return xR


learning_rate = tf.placeholder(tf.float32, shape=[])

raw_x = tf.placeholder(tf.float32, shape=[None, N_INPUT])
x = tf.placeholder(tf.float32, shape=[None, 2, N_INPUT], name="input")
first_board = tf.placeholder(tf.float32, shape=[None, N_INPUT])
second_board = tf.placeholder(tf.float32, shape=[None, N_INPUT])
y_ = tf.placeholder(tf.float32, shape=[None,2])


weights = {
	'e1' : weight_variable(N_INPUT, ENCODING_1),	
	'e2' : weight_variable(ENCODING_1, ENCODING_2),	
	'e3' : weight_variable(ENCODING_2, ENCODING_3),	
	'e4' : weight_variable(ENCODING_3, ENCODING_4),	
	'd1' : weight_variable(ENCODING_4, ENCODING_3),	
	'd2' : weight_variable(ENCODING_3, ENCODING_2),	
	'd3' : weight_variable(ENCODING_2, ENCODING_1),	
	'd4' : weight_variable(ENCODING_1, N_INPUT),	
	'w1' : weight_variable(HIDDEN_1, HIDDEN_2),	
	'w2' : weight_variable(HIDDEN_2, HIDDEN_3),	
	'w3' : weight_variable(HIDDEN_3, HIDDEN_4),	
	'w4' : weight_variable(HIDDEN_4, N_OUT)	
}

biases = {
	'e1' : bias_variable(ENCODING_1),	
	'e2' : bias_variable(ENCODING_2),	
	'e3' : bias_variable(ENCODING_3),	
	'e4' : bias_variable(ENCODING_4),	
	'd1' : bias_variable(ENCODING_3),	
	'd2' : bias_variable(ENCODING_2),	
	'd3' : bias_variable(ENCODING_1),	
	'd4' : bias_variable(N_INPUT),	
	'b1' : bias_variable(HIDDEN_2),	
	'b2' : bias_variable(HIDDEN_3),	
	'b3' : bias_variable(HIDDEN_4),	
	'out' : bias_variable(N_OUT)	
}

def fully_connected(current_layer, weight, bias):
	next_layer = tf.add(tf.matmul(current_layer, weight), bias)
	next_layer = tf.maximum(0.01*next_layer, next_layer)
	return next_layer

def encode(c, weights, biases, level):	
	e1 = fully_connected(c, weights['e1'], biases['e1'])
	if level == 1:
		return e1

	e2 = fully_connected(e1, weights['e2'], biases['e2'])
	if level == 2:
		return e2	

	e3 = fully_connected(e2, weights['e3'], biases['e3'])
	if level == 3:
		return e3
	
	e4 = fully_connected(e3, weights['e4'], biases['e4'])
	return e4

def decode(d, weights, biases, level):
	pred  = fully_connected(d, weights['d'+ str(5-level)], biases['d' + str(5-level)])
	return pred

def singleEncode(c, weights, biases, level):
	pred  = fully_connected(c, weights['e'+ str(level)], biases['e' + str(level)])
	return pred

def model(games, weights, biases):
	first_board = games[:,0,:]
	second_board = tf.squeeze(tf.slice(games, [0,1,0], [-1, 1, -1]), squeeze_dims=[1])
	[first_board, second_board] = tf.unpack(games, axis=1)
	
	firstboard_encoding = encode(first_board, weights, biases, 4)
	secondboard_encoding = encode(second_board, weights, biases, 4)

	h_1 = tf.concat(1, [firstboard_encoding,secondboard_encoding])
	h_2 = fully_connected(h_1, weights['w1'], biases['b1'])
	h_3 = fully_connected(h_2, weights['w2'], biases['b2'])
	h_4 = fully_connected(h_3, weights['w3'], biases['b3'])

	pred = tf.add(tf.matmul(h_4, weights['w4']), biases['out'], name="output")
	return pred


#layer by layer loss

encoded_1 = decode(encode(raw_x, weights, biases, 1), weights, biases, 1)
l2_loss_1 = tf.reduce_mean(tf.nn.l2_loss(tf.sub(encoded_1, raw_x)))

half_encoded2 = encode(raw_x, weights, biases, 1)
encoded_2 = decode(singleEncode(half_encoded2, weights, biases, 2), weights, biases, 2)
l2_loss_2 = tf.reduce_mean(tf.nn.l2_loss(tf.sub(encoded_2, half_encoded2)))

half_encoded3 = encode(raw_x, weights, biases, 2)
encoded_3 = decode(singleEncode(half_encoded3, weights, biases, 3), weights, biases, 3)
l2_loss_3 = tf.reduce_mean(tf.nn.l2_loss(tf.sub(encoded_3, half_encoded3)))

half_encoded4 = encode(raw_x, weights, biases, 3)
encoded_4 = decode(singleEncode(half_encoded4, weights, biases, 4), weights, biases, 4)
l2_loss_4 = tf.reduce_mean(tf.nn.l2_loss(tf.sub(encoded_4, half_encoded4)))

ae_train_step_1 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(l2_loss_1, var_list=[weights['e1'], biases['e1'], weights['d4'], biases['d4']])
ae_train_step_2 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(l2_loss_2, var_list=[weights['e2'], biases['e2'], weights['d3'], biases['d3']])
ae_train_step_3 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(l2_loss_3, var_list=[weights['e3'], biases['e3'], weights['d2'], biases['d2']])

ae_train_step_4 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(l2_loss_4, var_list=[weights['e4'], biases['e4'], weights['d1'], biases['d1']])


#full model
y = model(x, weights, biases)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y,y_))
mlp_train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

init = tf.initialize_all_variables()
saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init)
	
	if TRAIN_AUTOENCODER:
		total_batch = int(TOTAL_AE/BS_AE)
		for epoch in range(EPOCHS_AE):
			cur_rate = RATE_AE * (DECAY_AE**epoch)
			for i in range(total_batch): 
				batch_xs = getBatchAE(i, BS_AE)
				_, cost = sess.run([ae_train_step_1, l2_loss_1], feed_dict={raw_x: batch_xs, learning_rate: cur_rate})

			saver.save(sess, 'net/encoder_ae1.ckpt')
			print("AE_1 Epoch:", '%04d' % (epoch+1),
			"cost=", "{:.9f}".format(cost))

		for epoch in range(EPOCHS_AE):
			cur_rate = RATE_AE * (DECAY_AE**epoch)
			for i in range(total_batch): 
				batch_xs = getBatchAE(i, BS_AE)
				_, cost = sess.run([ae_train_step_2, l2_loss_2], feed_dict={raw_x: batch_xs, learning_rate: cur_rate})

			saver.save(sess, 'net/encoder_ae2.ckpt')
			print("AE_2 Epoch:", '%04d' % (epoch+1),
			"cost=", "{:.9f}".format(cost))
		
		for epoch in range(EPOCHS_AE):
			cur_rate = RATE_AE * (DECAY_AE**epoch)
			for i in range(total_batch): 
				batch_xs = getBatchAE(i, BS_AE)
				_, cost = sess.run([ae_train_step_3, l2_loss_3], feed_dict={raw_x: batch_xs, learning_rate: cur_rate})

			saver.save(sess, 'net/encoder_ae3.ckpt')
			print("AE_3 Epoch:", '%04d' % (epoch+1),
			"cost=", "{:.9f}".format(cost))
		
		for epoch in range(EPOCHS_AE):
			cur_rate = RATE_AE * (DECAY_AE**epoch)
			for i in range(total_batch): 
				batch_xs = getBatchAE(i, BS_AE)
				_, cost = sess.run([ae_train_step_4, l2_loss_4], feed_dict={raw_x: batch_xs, learning_rate: cur_rate})

			saver.save(sess, 'net/encoder_ae4.ckpt')
			print("AE_4 Epoch:", '%04d' % (epoch+1),
			"cost=", "{:.9f}".format(cost))
	
	if TRAIN_NET:
		#saver.restore(sess, 'net/encoder_ae3.ckpt')
		total_batch = int(TOTAL_MLP/BS_MLP)
		for epoch in range(EPOCHS_MLP):
			cur_rate = RATE_MLP * (DECAY_MLP**epoch)
			whiteWins = np.random.permutation(whiteWins)
			blackWins = np.random.permutation(blackWins)

			for i in range(total_batch): 
				batch_xs, batch_ys = getBatchMLP(i*BS_MLP, BS_MLP)
				_, cost = sess.run([mlp_train_step, cross_entropy], feed_dict={x: batch_xs, y_:batch_ys, learning_rate: cur_rate})

			print("MLP Epoch:", '%04d' % (epoch+1),
			"cost=", "{:.9f}".format(cost))

			model = td.Model()
			model.add(y, sess)
			model.save("model.pkl")
			saver.save(sess, 'net/net.ckpt')
			if epoch%3 == 2:	
				correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
				accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
				v_acc = sess.run(accuracy, feed_dict = {x: validation_test, y_: validation_test_l})
				print("Validation accuracy", "{:.9f}".format(v_acc))
		
		#This code can be used to check the training accuracy
		#train_test, train_test_l = getTest(N_INPUT, 0, 40)
		#t_acc = sess.run(accuracy, feed_dict = {x: train_test, y_: train_test_l})
		#print("Train accuracy", "{:.9f}".format(t_acc))

