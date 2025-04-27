# naive_bayes.py
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
util for printing values
'''
useless_words = ["a", "to", "it", "the", "is", "was", "an", "as", "how"]
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""

def naive_bayes(train_set, train_labels, dev_set, laplace=1.0, pos_prior=0.5, silently=False):
    print_values(laplace,pos_prior)

    yhats = []
    # record the number of pos and neg in train
    class_totals = Counter()
    # record the total numbers of words of pos and neg in train
    word_totals = Counter()
    # record the word in pos or neg in train
    word_counts = defaultdict(Counter)
    # record all non-repeated words
    vocabulary = set()

    # record
    for doc, label in zip(train_set, train_labels):
        class_totals[label] += 1
        for word in doc:
            if word in useless_words:
                continue
            word_counts[label][word] += 1
            word_totals[label] += 1
            vocabulary.add(word)
    # predict

    total_docs = sum(class_totals.values())
    # log(P(C|W1,…Wk))  ∝  log(P(C))+∑nk=1log(P(Wk|C))
    # P0 = class_totals[0] / total_docs
    # P1 = class_totals[1] / total_docs
    for doc in tqdm(dev_set, disable=silently):
        log_probs = {0: math.log(1-pos_prior), 1: math.log(pos_prior)}
        for label in range(0, 2):
            for word in doc:
                # P(word,T=label)
                if word in useless_words:
                    continue
                if word in vocabulary:
                    word_freq = word_counts[label][word]+laplace
                    word_prob = word_freq / (word_totals[label] + laplace * (len(vocabulary) + 1))
                else:
                    word_freq = laplace
                    word_prob = word_freq / (word_totals[label] + laplace * (len(vocabulary) + 1))
                log_probs[label] += math.log(word_prob)
        yhats.append(max(log_probs, key=log_probs.get))
    return yhats

