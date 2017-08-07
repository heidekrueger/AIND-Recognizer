import warnings
from asl_data import SinglesData
import numpy as np


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Likelihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer

    for i in range(test_set.num_items):
        X, l = test_set.get_item_Xlengths(i)

        probs_i = {}
        best_score = -np.inf
        best_guess = None

        for candidate_word in models.keys():
            # get ll and add it to probability dict
            try:
              ll = models[candidate_word].score(X, l)
            except Exception:
              ll = -np.inf

            probs_i[candidate_word] = ll

            if ll > best_score:
                best_score = ll
                best_guess = candidate_word

        probabilities.append(probs_i)
        guesses.append(best_guess)

    # return probabilities, guesses
    return probabilities, guesses
