"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""
from collections import defaultdict
def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    train_dict = defaultdict(lambda: defaultdict(int))
    tags = defaultdict(int)
    for sentence in train:
        for word, tag in sentence:
            train_dict[word][tag] += 1
            tags[tag] += 1

    word_to_most_frequent_tag = {}
    for word, tag_freq in train_dict.items():
        # Find the tag with the highest frequency
        most_frequent_tag = max(tag_freq, key=tag_freq.get)
        word_to_most_frequent_tag[word] = most_frequent_tag
    most_freq_tag = max(tags, key=tags.get)

    res = []
    for sentence in test:
        tmp = []
        for word in sentence:
            if word in word_to_most_frequent_tag:
                tmp.append((word, word_to_most_frequent_tag[word]))
            else:
                tmp.append((word, most_freq_tag))
        res.append(tmp)
    return res