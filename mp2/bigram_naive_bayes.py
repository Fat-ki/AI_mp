# bigram_naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""

import reader
import math
from tqdm import tqdm
from collections import defaultdict, Counter

'''
utils for printing values
'''
useless_words = ["a", "to", "it", "the", "is", "was", "an", "as", "how"]


def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")


def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
    print(f"Positive prior: {pos_prior}")


"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""


def load_data(trainingdir, testdir, stemming=True, lowercase=True, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir, testdir, stemming, lowercase,
                                                                       silently)
    for tdoc, ddoc in zip(train_set, dev_set):
        for uword in useless_words:
            if uword in tdoc:
                tdoc.remove(uword)
            if uword in ddoc:
                ddoc.remove(uword)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""


def bigram_bayes(train_set, train_labels, dev_set, unigram_laplace=0.01, bigram_laplace=0.01, bigram_lambda=0.7,
                 pos_prior=0.5, silently=False):
    print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior)

    # record the total numbers of words of pos and neg in train
    b_word_totals = Counter()
    # record the word in pos or neg in train
    b_word_counts = defaultdict(Counter)
    # record all non-repeated words
    b_vocabulary = set()

    u_word_totals = Counter()
    # record the word in pos or neg in train
    u_word_counts = defaultdict(Counter)
    # record all non-repeated words
    u_vocabulary = set()

    for doc, label in zip(train_set, train_labels):
        for i in range(len(doc) - 1):
            b_word = doc[i] + ' ' + doc[i + 1]
            b_word_counts[label][b_word] += 1
            b_word_totals[label] += 1
            b_vocabulary.add(b_word)
        for word in doc:
            u_word_counts[label][word] += 1
            u_word_totals[label] += 1
            u_vocabulary.add(word)

    yhats = []
    for doc in tqdm(dev_set, disable=silently):
        log_probs = {0: 0, 1: 0}
        for label in range(0, 2):
            u = unigram(label, doc, u_vocabulary, u_word_counts, u_word_totals, unigram_laplace, pos_prior, silently)
            b = bigram(label, doc, b_vocabulary, b_word_counts, b_word_totals, bigram_laplace, pos_prior, silently)
            log_probs[label] = (1 - bigram_lambda) * u + bigram_lambda * b
        yhats.append(max(log_probs, key=log_probs.get))
    return yhats


def unigram(label, doc, vocabulary, word_counts, word_totals, bigram_laplace, pos_prior, silently=False):
    yhats = []
    log_probs = {0: math.log(1 - pos_prior), 1: math.log(pos_prior)}
    for word in doc:
        # P(word,T=label)
        if word in vocabulary:
            word_freq = word_counts[label][word] + bigram_laplace
            word_prob = word_freq / (word_totals[label] + bigram_laplace * (len(vocabulary) + 1))
        else:
            word_freq = bigram_laplace
            word_prob = word_freq / (word_totals[label] + bigram_laplace * (len(vocabulary) + 1))
        log_probs[label] += math.log(word_prob)
    # yhats.append(max(log_probs, key=log_probs.get))
    return log_probs[label]


def bigram(label, doc, vocabulary, word_counts, word_totals, bigram_laplace, pos_prior, silently=False):
    yhats = []
    log_probs = {0: math.log(1 - pos_prior), 1: math.log(pos_prior)}
    for i in range(len(doc) - 1):
        # P(word,T=label)
        word = doc[i] + ' ' + doc[i + 1]

        if word in vocabulary:
            word_freq = word_counts[label][word] + bigram_laplace
            word_prob = word_freq / (word_totals[label] + bigram_laplace * (len(vocabulary) + 1))
        else:
            word_freq = bigram_laplace
            word_prob = word_freq / (word_totals[label] + bigram_laplace * (len(vocabulary) + 1))
        log_probs[label] += math.log(word_prob)
    # yhats.append(max(log_probs, key=log_probs.get))
    return log_probs[label]
