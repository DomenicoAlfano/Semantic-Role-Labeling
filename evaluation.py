class Scorer:

    def __init__(self):

        self.false_positive = None

        self.false_negative = None

        self.cc = None

        self._reset()

    def _reset(self):

        self.false_positive = 0

        self.false_negative = 0

        self.cc = 0

    def __call__(self, y_test, y_pred):

        for id_sent, sentence in enumerate(y_test):

            for idx, arg in enumerate(sentence):

                if arg == 1:

                    if y_pred[id_sent][idx] != 1:

                        self.false_negative += 1

                else:

                    if y_pred[id_sent][idx] == 1:

                        self.false_positive += 1

                    elif arg != y_pred[id_sent][idx]:

                        self.false_negative += 1

                    else:

                        self.cc += 1

    def compute_metrics(self):

        epsilon = 1e-8

        precision = self.cc / (self.cc + self.false_positive + epsilon)

        recall = self.cc / (self.cc + self.false_negative + epsilon)

        f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

        return precision*100, recall*100, f1_score*100
