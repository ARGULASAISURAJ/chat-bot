from rasa.nlu.components import Component
import pickle
from rasa.nlu.model import Metadata
from rasa.shared.nlu.constants import INTENT, TEXT
from rasa.nlu.constants import TOKENS_NAMES

# nltk needs to be installed
import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import WhitespaceTokenizer
import os

import typing
from typing import Any, Optional, Text, Dict

SENTIMENT_MODEL_FILE_NAME = "sentiment_classifier.pkl"

class SentimentAnalyzer(Component):
    """A custom sentiment analysis component"""
    name = "sentiment"
    defaults = {}
    language_list = ["en"]

    def __init__(self, component_config=None):
        super(SentimentAnalyzer, self).__init__(component_config)

    def train(self, training_data, cfg, **kwargs):
        # Load the labels from the text file, retrieve training tokens for share_problems intent and after formatting data train the classifier.

        with open('labels.txt', 'r') as f:
            labels = f.read().splitlines()

        with open('training_data.txt', 'r') as f:
            training_data = f.read().splitlines()

        tokens = [WhitespaceTokenizer().tokenize(t) for t in training_data]
        processed_tokens = [self.preprocessing(t) for t in tokens]
        labeled_data = [(t, x) for t,x in zip(processed_tokens, labels)]
        self.clf = NaiveBayesClassifier.train(labeled_data)

    def convert_to_rasa(self, value, confidence):
        """Convert model output into the Rasa NLU compatible output format."""

        entity = {"value": value,
                  "confidence": confidence,
                  "entity": "sentiment",
                  "extractor": "sentiment_extractor"}

        return entity
        

    def preprocessing(self, tokens):
        """Create bag-of-words representation of the training examples."""
        
        return ({word: True for word in tokens})


    def process(self, message, **kwargs):
        """Retrieve the tokens of the new message, pass it to the classifier
            and append prediction results to the message class."""
        
        if not self.clf:
            # component is either not trained or didn't receive enough training data
            entity = None
        else:
            tokens = [t.text for t in message.get(TOKENS_NAMES[TEXT])]
            tb = self.preprocessing(tokens)
            pred = self.clf.prob_classify(tb)

            sentiment = pred.max()
            confidence = pred.prob(sentiment)

            entity = self.convert_to_rasa(sentiment, confidence)

            message.set("entities", [entity], add_to_output=True)


    def persist(self, file_name, model_dir):
        """Persist this model into the passed directory."""
        classifier_file = os.path.join(model_dir, SENTIMENT_MODEL_FILE_NAME)
        pickle.dump(self, open(classifier_file, 'wb'))
        return {"classifier_file": SENTIMENT_MODEL_FILE_NAME}

    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             model_dir=None,
             model_metadata=None,
             cached_component=None,
             **kwargs):
        file_name = meta.get("classifier_file")
        classifier_file = os.path.join(model_dir, file_name)
        return pickle.load(open(classifier_file, 'rb'))