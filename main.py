import classifiers
from ANN import NeuralNetwork
import random
import numpy as np

functions = {
    'sigmoid': {'forward': lambda a: 1 / (1 + np.exp(-a)), 'derived': lambda d: d*(1 - d)},
    'tanh': {'forward': lambda a: np.tanh(a), 'derived': lambda d: 1 - np.power(d, 2)}
}


def combine(naive_bayes_value, logestic_regression_value, test):
    # -infinity < nb < +infinity
    # 0 < lr < 1
    nb_delta = max(naive_bayes_value) - min(naive_bayes_value)
    lr_delta = max(logestic_regression_value) - min(logestic_regression_value)
    c = nb_delta/lr_delta
    results = []
    values = []
    for i in range(len(naive_bayes_value)):
        lr_normalized = (logestic_regression_value[i] * c) - c/2
        comb = (lr_normalized + naive_bayes_value[i])/2
        r = 0
        if comb > 0:
            r = 1
        results.append(r)
        values.append(comb)
    y_test = [tweet[0] for tweet in test]
    accuracy = sum([1 if y_test[i] == results[i]
                    else 0 for i in range(len(test))])/len(test)

    return accuracy, values


def naive_bayes(freqs, test):
    model = classifiers.NaiveBayes(freqs)
    results = [model.predict(tweet)['label'] for tweet in test]
    values = [model.predict(tweet)['value'] for tweet in test]
    y_test = [tweet[0] for tweet in test]
    accuracy = sum([1 if y_test[i] == results[i]
                    else 0 for i in range(len(test))])/len(test)
    return accuracy, values


def logestic_regression(freqs, train, test):
    model = classifiers.LogesticRegression(
        freqs, train, alpha=10**-5, iteration=70)
    results = [model.predict(tweet)['label'] for tweet in test]
    values = [model.predict(tweet)['value'] for tweet in test]
    y_test = [tweet[0] for tweet in test]
    accuracy = sum([1 if y_test[i] == results[i]
                    else 0 for i in range(len(test))])/len(test)
    return accuracy, values


def neural_network(freqs, train, test, debug=False):
    x, y = classifiers.extract_features(freqs, train)
    model = NeuralNetwork(x, y, classes=2, hidden_layers=1, hidden_nodes=[
                          3], activation_func=functions['tanh'], mini_batch=True)
    model.learn()
    x_test, y_test = classifiers.extract_features(
        classifiers.frequency(test), test)
    results = [model.predict(tweet) for tweet in x_test]
    accuracy = sum([1 if y_test[i] == results[i]
                    else 0 for i in range(len(test))])/len(test)
    if debug:
        classifiers.plot_decision_boundary(
            x_test, y_test, model, title='test data')
        for i in range(len(test)):
            print('label:{} , clean tweet: {}'.format(test[i][0], test[i][1]))
            print('predict:{}'.format(results[i]))

    return accuracy, model


def main():
    tweets = classifiers.read_file('StrictOMD.csv')
    random.shuffle(tweets)
    no_tweets = len(tweets)
    nb_acc = 0
    lr_acc = 0
    nn_acc = 0
    com_acc = 0
    for i in range(10):
        start = int(i/10*no_tweets)
        end = int((i+1)/10*no_tweets)
        test_data = tweets[start:end]
        train_data = tweets[:start]+tweets[end:]
        freqs = classifiers.frequency(train_data)
        nb_a, nb_v = naive_bayes(freqs, test_data)
        lr_a, lr_v = logestic_regression(freqs, train_data, test_data)
        # debug for show decision boundary and print each tweet with its prediction
        nn_a, nn_model = neural_network(
            freqs, train_data, test_data, debug=False)
        com_a, com_v = combine(naive_bayes_value=nb_v,
                               logestic_regression_value=lr_v, test=test_data)

        nb_acc += nb_a
        lr_acc += lr_a
        nn_acc += nn_a
        com_acc += com_a
        print('\t\t*** Partition {} , {} test data ***\nNaive Bayes : {:.3f}\nLogestic Regression : {:.3f}\nCombine NB & LR : {:.3f}\nNeural Network : {:.4f}'.format(i+1, len(test_data),
                                                                                                                                                                      nb_a, lr_a, com_a, nn_a))

    print('\n\t\t***** Average accuracy *****\nNaive Bayes : {:.4f}\nLogestic Regression : {:.4f}\nCombine NB & LR : {:.4f}\nNeural Network : {:.4f}'.format(
        nb_acc/10, lr_acc/10, com_acc/10, nn_acc/10))


main()

#                ***** Average accuracy *****
# Naive Bayes : 0.7358
# Logestic Regression : 0.7369
# Combine NB & LR : 0.7613
# Neural Network : 0.9934
