import os
import math
from collections import defaultdict


class NaiveBayesClassifier:
    def __init__(self):
        self.class_totals = defaultdict(int)
        self.word_totals = defaultdict(int)
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.vocabulary = set()

    def train(self, data, labels):
        """
        Train the Naive Bayes classifier.

        :param data: A list of documents (each document is a list of words).
        :param labels: A list of class labels corresponding to each document.
        """
        for doc, label in zip(data, labels):
            self.class_totals[label] += 1
            for word in doc:
                self.word_counts[label][word] += 1
                self.word_totals[label] += 1
                self.vocabulary.add(word)

    def predict(self, doc):
        """
        Predict the class label for a given document.

        :param doc: A list of words (document).
        :return: The predicted class label.
        """
        log_probs = {}
        total_docs = sum(self.class_totals.values())

        for label in self.class_totals:
            log_probs[label] = math.log(self.class_totals[label] / total_docs)

            for word in doc:
                word_freq = self.word_counts[label][word]
                word_prob = (word_freq + 1) / (self.word_totals[label] + len(self.vocabulary))
                log_probs[label] += math.log(word_prob)

        return max(log_probs, key=log_probs.get)


# Example usage
if __name__ == "__main__":
    # Sample data: List of documents and their corresponding labels
    data = [
        ["love", "this", "movie"],
        ["hate", "this", "movie"],
        ["best", "film", "ever"],
        ["worst", "film", "ever"]
    ]
    labels = ["positive", "negative", "positive", "negative"]

    # Initialize and train the classifier
    nb = NaiveBayesClassifier()
    nb.train(data, labels)

    # Predict the class of a new document
    test_doc = ["love", "this", "film"]
    prediction = nb.predict(test_doc)
    print(f"Prediction: {prediction}")
