"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import math
from collections import defaultdict, Counter
from math import log

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5  # exact setting seems to have little or no effect


def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0.0)  # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0.0))  # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0.0))  # {tag0:{tag1: # }}

    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.
    init_prob = {"START": 1.0}
    tag_list = set()

    for sentence in sentences:
        prev_tag = None
        for word, tag in sentence:
            tag_list.add(tag)

            emit_prob[tag][word] += 1
            if prev_tag is not None:
                trans_prob[prev_tag][tag] += 1
            prev_tag = tag

    for tag in tag_list:
        total_emissions = sum(emit_prob[tag].values())
        vocab_size = len(emit_prob[tag])

        emit_prob[tag]["UNK"] = emit_epsilon / (total_emissions + emit_epsilon * (vocab_size + 1))

        for word in emit_prob[tag]:
            emit_prob[tag][word] = (emit_prob[tag][word] + emit_epsilon) / (total_emissions + emit_epsilon*(vocab_size + 1))

    for prev_tag in tag_list:
        total_transitions = sum(trans_prob[prev_tag].values())
        tag_size = len(trans_prob[prev_tag])

        for curr_tag in tag_list:
            if curr_tag in trans_prob[prev_tag]:
                trans_prob[prev_tag][curr_tag] = (trans_prob[prev_tag][curr_tag] + epsilon_for_pt) / (total_transitions + epsilon_for_pt*(tag_size+1))
            else:
                trans_prob[prev_tag][curr_tag] = epsilon_for_pt / (total_transitions + epsilon_for_pt*(tag_size+1))

    return init_prob, emit_prob, trans_prob


def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = {}  # This should store the log_prob for all the tags at current column (i)
    predict_tag_seq = {}  # This should store the tag sequence to reach each tag at column (i)
    # print(f"prev prob: {prev_prob}")
    # print(f"pre predict seq: {prev_predict_tag_seq}")
    # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.

    if i == 0:
        for curr_tag in emit_prob.keys():

            if word in emit_prob[curr_tag]:
                emission_log_prob = log(emit_prob[curr_tag][word])
            else:
                emission_log_prob = log(emit_prob[curr_tag]["UNK"])
            total_log_prob = emission_log_prob

            log_prob[curr_tag] = total_log_prob
            predict_tag_seq[curr_tag] = [curr_tag]
    else:
        for curr_tag in emit_prob.keys():

            best_log_prob = float('-inf')
            best_prev_tag = None

            if word in emit_prob[curr_tag]:
                emission_log_prob = log(emit_prob[curr_tag][word])
            else:
                emission_log_prob = log(emit_prob[curr_tag]["UNK"])

            for prev_tag in prev_prob.keys():
                transition_log_prob = log(trans_prob[prev_tag][curr_tag])

                total_log_prob = prev_prob[prev_tag] + transition_log_prob + emission_log_prob
                if total_log_prob > best_log_prob:
                    best_log_prob = total_log_prob
                    best_prev_tag = prev_tag

            log_prob[curr_tag] = best_log_prob
            predict_tag_seq[curr_tag] = prev_predict_tag_seq[best_prev_tag] + [curr_tag]
    return log_prob, predict_tag_seq


def viterbi_1(train, test, get_probs=training):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = get_probs(train)
    predicts = []
    for sen in range(len(test)):
        sentence = test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[t] = []

        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob,
                                                            trans_prob)

        # TODO:(III) 
        # according to the storage of probabilities and sequences, get the final prediction.
        final_tag = max(log_prob, key=log_prob.get)
        best_tag_seq = predict_tag_seq[final_tag]

        predicted_sentence = [(sentence[i], best_tag_seq[i]) for i in range(length)]
        predicts.append(predicted_sentence)
    return predicts
