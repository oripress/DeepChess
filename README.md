# DeepChess: An Implementation

I came across [DeepChess](http://www.cs.tau.ac.il/~wolf/papers/deepchess.pdf) and decided to implement it to learn TensorFlow and to experiment with Deep Learning methods.

Please note: while the program is available to play, I've never actually trained the full network with the full amount of games available, because of time constraints. I'm currently still training the network and will update this page with the results when they come in.
Update: Achieved 97.2% accuracy on the validation set (the engine is playable but there still a few minor bugs).

###To play:###
Install [python-chess](https://pypi.python.org/pypi/python-chess), and then from the main directory, run: `python game.py`


###Some notes:####
The basic idea of the paper is that we can get a deep network to play chess by teaching it an evaluation function that takes in 2 positions and outputs the better one. The network can then be used in a modified Alpha-Beta pruning algorithm, where instead of comparing between two positions' evaluations (as numbers), we compare between the positions themselves.
##
The network consists of two main components, namely "Pos2Vec", and a fully connected MLP. The "Pos2Vec" component is a Deep Belief Network that consists of 4 stacked autoencoders, that are trained layer by layer, unsupervised. Two identical "Pos2Vec" components lay side by side and feed into a 4 layer MLP. The whole network is trained on 1 million pairs of positions, wherein the pre-training serves as the inital weights of the "Pos2Vec" components.
The network trains on the [CCRL dataset](http://www.computerchess.org.uk/ccrl/4040/games.html). 
