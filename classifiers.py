import math
import random
import re
from nltk.stem import PorterStemmer
import numpy as np
import matplotlib.pyplot as plt
# from nltk.corpus import stopwords - {won,over}
stopwords = ['i', "i'm", 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
             'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',  'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",  "won't", 'wouldn', "wouldn't"]


class NaiveBayes:
    def __init__(self, freqs):
        self.freqs = freqs
        self.logprior = self.calculate_logprior()
    def calculate_logprior(self):
        obama = 0
        mccain = 0
        for w,freq in self.freqs.items():
            obama+= freq['label']
            mccain+= freq['f']-freq['label']
        return math.log(obama/mccain)
    def predict(self, tweet):
        p = 0 # self.logprior , if use 0 model works better! 
        p_flag = 0
        for word in tweet[1]:
            if word in self.freqs:
                freq = self.freqs[word]
                obama = (freq['label']+1)/(freq['f']+1)
                mccain = (freq['f']-freq['label']+1)/(freq['f']+1)
                p += math.log(obama/mccain)
        if p > 0:
            p_flag = 1

        return {'label':p_flag,'value':p}


class LogesticRegression:
    def __init__(self, freqs, tweets, alpha, iteration):
        '''
        freqs : a dictunary of all word in train tweets
        freqs [word] :{'f':frequency over all tweets , 'label':sum( 0 for maccain 1 for obama)}
        '''
        self.freqs = freqs
        self.tweets = tweets
        self.learning_rate = alpha
        self.iteration = iteration
        self.theta = np.zeros((3, 1))
        self.train()

    def extract_features(self, tweet):
        '''
        Input:
            tweet: a list of words for one clean tweet
        Output:
            x: a feature vector of dimension (1,3) -> [obama,mccain,bias]
        '''
        x = np.array([0,0,0],dtype=np.float)
        for word in tweet[1]:
            if word in self.freqs:
                freq = self.freqs[word]
                obama = freq['label'] 
                mccain = (freq['f']-freq['label'])
                x[0] += obama
                x[1] += mccain
        return x

    def gradientDescent(self, x, y):
        m = x.shape[0]
        for i in range(self.iteration):
            z = np.dot(x, self.theta)
            h = self.sigmoid(z)
            # calculate the cost function for logestic regressin
            J = -1./m * (np.dot(y.T, np.log(h)) + np.dot((1-y).T, np.log(1-h)))
            # update the weights theta
            self.theta = self.theta - \
                (self.learning_rate/m) * np.dot(x.T, (h-y))
        J = float(J)
        return J

    def train(self):
        train_size = len(self.tweets)
        X = np.zeros((train_size, 3)) #3
        for i in range(train_size):
            X[i, :] = self.extract_features(self.tweets[i])
        train_y = [i[0] for i in self.tweets]
        Y = np.array(train_y).reshape(-1, 1)
        # Apply gradient descent
        J = self.gradientDescent(X, Y)
        # print("The cost after training is {}".format(J))

    def predict(self, tweet):
        x = self.extract_features(tweet)
        y_pred = self.sigmoid(np.dot(x, self.theta))
        flag = 0
        if y_pred > 0.5:
            flag = 1

        return {'label':flag,'value':y_pred}


    def sigmoid(self, z):
        h = 1 / (1 + np.exp(-z))
        return h

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[-()\"#/@;:<>{}=~|.?,%$!]", " ", text)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    return text


def read_file(file_name):
    file = open(file_name, 'r+')
    data = []
    stemmer = PorterStemmer()
    for line in file.readlines():
        tweet = clean_text(line)
        label = int(tweet[0])//4  # McCain:0 , Obama:1
        words = tweet[1:].split()
        stem_words = []
        for w in words:
            if w not in stopwords:
                stem = stemmer.stem(w)
                stem_words.append(stem)
            # else:
                # pass
                # print(w) over , won
        case = [label, stem_words]
        # print(case)
        data.append(case)
    # print(data)
    return data


def frequency(data):
    freq = {}
    for tweet in data:
        label = tweet[0]
        for word in tweet[1]:
            if len(word) < 3:
                # print(word)
                continue
            if word in freq:
                freq[word]['f'] += 1
                freq[word]['label'] += label

            else:
                freq[word] = {}
                freq[word]['f'] = 1
                freq[word]['label'] = label
    sorted_dic = {k: v for k, v in sorted(
        freq.items(), key=lambda item: item[1]['f'], reverse=True)}
    # print(sorted_dic)
    return freq
def plot_decision_boundary(X, y, model,title=''):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.title(title)
    plt.show()
def extract_features(frequency,tweets):
    X = []
    y = []
    for tweet in tweets:
        x = np.array([0,0],dtype=np.float)
        for word in tweet[1]:
            if word in frequency:
                freq = frequency[word]
                obama = freq['label'] / freq['f']
                mccain = (freq['f']-freq['label'])/freq['f']
                x[0] += obama
                x[1] += mccain
        X.append(x)    
        y.append(tweet[0])
    X = np.asarray(X)
    y = np.asarray(y)
    #print(X)
    #print(y)
    #plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    #plt.show()
    return X,y

