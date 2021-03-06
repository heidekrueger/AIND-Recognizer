import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores

        best_score = np.Inf   
        best_model = None #self.base_model(self.min_n_components)
        best_n = None

        n_obs = len(self.X)
      
        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(n)
                model.fit(self.X, self.lengths)


                # finding the number of free params:
                # let d = n_features, n = n_states
                # transition probabilities: n*(n-1)
                #       (Note: in class only (n-1) transitions to next state,
                #              but hmmlearn uses full transition matrix)
                # insertiean probs: n-1 (hmmlearn "learns" which state to start in)
                # For each state n:
                #   d means
                #   d variance entries (we assume diagonal covariance)
                # thus p = (n+1)(n-1) + 2nd 

                p = n**2 + 2*n*model.n_features - 1
                score = -2 * model.score(self.X, self.lengths) + p * np.log(n_obs)

                if score < best_score:
                    best_score = score
                    best_model = model
                    best_n = n

            except Exception:
                continue

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_score = -np.Inf   
        best_model = None #self.base_model(self.min_n_components)
        best_n = None

        n_obs = len(self.X)
      
        for n in range(self.min_n_components, self.max_n_components+1):
            try: 

                model = self.base_model(n)
                model.fit(self.X, self.lengths)

                logL = model.score(self.X, self.lengths)

                # now get anti_logL for each other word

                count = 0
                sum_anti_ll = 0

                for word in self.hwords:
                    if word == self.this_word:
                        continue

                    anti_X, anti_length = self.hwords[word]

                    sum_anti_ll += model.score(anti_X, anti_length)
                    count += 1

                # get final score according to DIC:

                score = logL - 1/float(count) * sum_anti_ll

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_n = n

            except Exception:
                continue

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = - np.Inf   
        # use model(min_n) instead of None in order to return current bestif failing a try block:
        # Update: actually, DO use None (in case min_n also fails)
        best_model = None #self.base_model(self.min_n_components)
        best_n = None

        # If fewer samples then 3, adjust accordingly
        k = min(3, len(self.sequences))

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                if self.verbose:
                    print("Training model with n={}".format(n))

                model = self.base_model(n)
                score = 0
                # train and score for each fold, then get avg. test score

                split = KFold(n_splits=k).split(self.sequences)
                for train_idx, test_idx in split:
                    X_train, l_train = combine_sequences(train_idx, self.sequences)
                    X_test, l_test = combine_sequences(test_idx, self.sequences)

                    model.fit(X_train, l_train)
                    score += model.score(X_test, l_test)

                score /= k

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_n = n

                    if self.verbose:
                        print("n={} was best model so far, w/ score of {}".format(n, score))
            except Exception:
                continue


        if self.verbose:
            print("Best model is n={}".format(best_n))

        # train best model on full dataset (unless None)
        if best_model:
            return best_model.fit(self.X, self.lengths)

        return None
