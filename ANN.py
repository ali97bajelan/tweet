import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets


class NeuralNetwork:
    def __init__(self, X, y, classes, hidden_layers, hidden_nodes, activation_func, epsilon=0.01, regularization=0.01, iteration=20000, mini_batch=False,dynamic_LR=False):
        self.X=X
        self.y=y
        self.no_classes=classes
        self.no_hidden_layers=hidden_layers
        self.no_hidden_nodes=hidden_nodes
        self.W=[]
        self.b=[]
        self.activation_function=activation_func
        self.epsilon=epsilon
        self.regularization=regularization
        self.iteration=iteration
        self.mini_batch_mode = mini_batch
        self.dynamic_learning_rate=dynamic_LR
        #self.welcome()
        self.initialize_weights()
    def welcome(self):
        print('\t\t','*'*5,'\twelcome\t','*'*5)
        print('ANN with ths parametr is running:\nNumber of example = {}\tNumber of classes = {}\nNumber of hidden layers = {}\tSize of layers = {}\nLearning Rate = {}\tIterations = {}\nMini Batch mode = {}\tDynamic Learning Rate = {}'.format(len(self.X),self.no_classes,self.no_hidden_layers,self.no_hidden_nodes,self.epsilon,self.iteration,self.mini_batch_mode,self.dynamic_learning_rate))

    def initialize_weights(self):
        nodes=[len(self.X[0])]+[i for i in self.no_hidden_nodes] +[self.no_classes]
        weights=[]
        np.random.seed(0)
        for i in range(self.no_hidden_layers+1):
            w=np.random.randn(nodes[i], nodes[i+1]) / np.sqrt(nodes[i])
            weights.append(w)
        self.W=np.asarray(weights, dtype=np.ndarray)

        biases=[]
        bs=[[0 for j in range(i)] for i in nodes[1:]]
        self.b=np.array(bs, dtype=np.ndarray)
    def get_mini_batch(self,size=16):
        sample_index = np.random.choice(len(self.X), size=size)
        sample_x = np.asarray([self.X[i] for i in sample_index])
        sample_y = np.asarray([self.y[i] for i in sample_index])
        return sample_x,sample_y
    def update_learning_rate(self,i):
        self.epsilon = self.epsilon*(1-i/self.iteration)

    def calculate_loss(self):
        a, probs=self.feed_forward()
        corect_logprobs=-np.log(probs[range(len(self.X)), self.y])
        data_loss=np.sum(corect_logprobs)
        return 1./len(self.X) * data_loss
    def predict(self,x):
        a, probs=self.feed_forward(x)
        return np.argmax(probs, axis=1)
    def feed_forward(self,x=None):
        '''
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        '''
        if x is None:
            x = self.X
        Z=[x.dot(self.W[0])+self.b[0]]
        a=[]
        for i in range(1, self.no_hidden_layers+1):
            ai=self.activation_function['forward'](Z[-1])
            a.append(ai)
            Z.append(ai.dot(self.W[i])+self.b[i])
        exp_scores=np.exp(Z[-1])
        probs=exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return a, probs

    def feed_back(self, a, probs,x=None,y=None):
        '''Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
        '''
        if x is None:
            x = self.X
            y = self.y
        probs[range(len(x)), y] -= 1
        delta=[probs]
        dw=[]
        db=[]
        # a[i] to a[-(i+1)] in two line
        for i in range(self.no_hidden_layers):
            dwi=(a[-(i+1)].T).dot(delta[-1])
            dbi=np.sum(delta[-1], axis=0, keepdims=True)
            deltai=delta[-1].dot(self.W[-(i+1)].T) *self.activation_function['derived'](a[-(i+1)])
            dw.append(dwi)
            db.append(dbi)
            delta.append(deltai)
        dw1=np.dot(x.T, delta[-1])
        db1=np.sum(delta[-1], axis=0)
        dw.append(dw1)
        db.append(db1)

        dw.reverse()
        db.reverse()
        return dw, db

    def learn(self):
        for i in range(self.iteration):
            if self.mini_batch_mode:
                x,y = self.get_mini_batch()
                a, probs=self.feed_forward(x)
                dw, db=self.feed_back(a, probs,x,y)
            else:
                a, probs=self.feed_forward()
                dw, db=self.feed_back(a, probs)
            for i in range(self.no_hidden_layers+1):
                dw[i] += self.regularization*self.W[i]
                self.W[i] += -self.epsilon*dw[i]
                self.b[i] += -self.epsilon*db[i]
            if self.dynamic_learning_rate and i%100:
                self.update_learning_rate(i)
        loss = self.calculate_loss()
        #print('Loss : {}'.format(loss))
        #print('\t\t','*'*5,'\tbye\t','*'*5)
        #print(self.W)
        #print(self.b)



#ann=NeuralNetwork(X=X, y=y, classes=2, hidden_layers=1, hidden_nodes=[5],epsilon=0.1,regularization=0.01 ,iteration=20000,activation_func=functions['tanh'],mini_batch=True,dynamic_LR=True)
#ann.learn()

