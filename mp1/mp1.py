# mp1.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023

import sys
import argparse
import configparser
import copy

import reader
import naive_bayes as nb

"""
This file contains the main application that is run for this MP.
"""

def compute_accuracies(predicted_labels, dev_labels):
    yhats = predicted_labels
    assert len(yhats) == len(dev_labels), "predicted and gold label lists have different lengths"
    accuracy = sum([yhats[i] == dev_labels[i] for i in range(len(yhats))]) / len(yhats)
    tp = sum([yhats[i] == dev_labels[i] and yhats[i] == 1 for i in range(len(yhats))])
    tn = sum([yhats[i] == dev_labels[i] and yhats[i] == 0 for i in range(len(yhats))])
    fp = sum([yhats[i] != dev_labels[i] and yhats[i] == 1 for i in range(len(yhats))])
    fn = sum([yhats[i] != dev_labels[i] and yhats[i] == 0 for i in range(len(yhats))])
    return accuracy, fp, fn, tp, tn

# print value and also percentage out of n
def print_value(label, value, numvalues):
   print(f"{label} {value} ({value/numvalues * 100}%)")

# print out performance stats
def print_stats(accuracy, false_positive, false_negative, true_positive, true_negative, numvalues):
    print(f"Accuracy: {accuracy}")
    print_value("False Positive", false_positive,numvalues)
    print_value("False Negative", false_negative,numvalues)
    print_value("True Positive", true_positive,numvalues)
    print_value("True Negative", true_negative,numvalues)
    print(f"total number of samples {numvalues}")


"""
Main function
    You can modify the default parameter settings given below, 
    instead of constantly typing your favorite values at the command line.
"""
def main(args):
    train_set, train_labels, dev_set, dev_labels = nb.load_data(args.training_dir,args.development_dir,args.stemming,args.lowercase)
    
    predicted_labels = nb.naive_bayes(train_set, train_labels, dev_set, args.laplace, args.pos_prior)

    accuracy, false_positive, false_negative, true_positive, true_negative = compute_accuracies(predicted_labels,dev_labels)
    nn = len(dev_labels)
    print_stats(accuracy, false_positive, false_negative, true_positive, true_negative, nn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS440 MP1 Naive Bayes')
    parser.add_argument('--training', dest='training_dir', type=str, default = 'data/movie_reviews/train',
                        help='the directory of the training data')
    parser.add_argument('--development', dest='development_dir', type=str, default = 'data/movie_reviews/dev',
                        help='the directory of the development data')

    # When doing final testing, reset the default values below to match your settings in naive_bayes.py
    parser.add_argument('--stemming',dest="stemming", type=bool, default=True,
                        help='Use porter stemmer')
    parser.add_argument('--lowercase',dest="lowercase", type=bool, default=True,
                        help='Convert all word to lower case')
    parser.add_argument('--laplace',dest="laplace", type=float, default = 0.01,
                        help='Laplace smoothing parameter')
    parser.add_argument('--pos_prior',dest="pos_prior", type=float, default = 0.75,
                        help='Positive prior, i.e. percentage of test examples that are positive')

    args = parser.parse_args()
    main(args)
# 0.834