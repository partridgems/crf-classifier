import numpy as np
from math import exp

class CRF(object):

    def __init__(self, label_codebook, feature_codebook):
        self.label_codebook = label_codebook
        self.feature_codebook = feature_codebook
        num_labels = len(self.label_codebook)
        num_features = len(self.feature_codebook)
        self.feature_parameters = np.zeros((num_labels, num_features))
        self.transition_parameters = np.zeros((num_labels, num_labels))

    def train(self, training_set, dev_set):
        """Training function

        Feel free to adjust the hyperparameters (learning rate and batch sizes)
        """
        self.train_sgd(training_set, dev_set, 0.001, 200)

    def train_sgd(self, training_set, dev_set, learning_rate, batch_size):
        """Minibatch SGF for training linear chain CRF

        This should work. But you can also implement early stopping here
        i.e. if the accuracy does not grow for a while, stop.
        """
        num_labels = len(self.label_codebook)
        num_features = len(self.feature_codebook)

        num_batches = len(training_set) / batch_size
        total_expected_feature_count = np.zeros((num_labels, num_features))
        total_expected_transition_count = np.zeros((num_labels, num_labels))

        """Save our go-back point"""
        old_transition = np.copy(self.transition_parameters)
        old_feature = np.copy(self.feature_parameters)
        old_accuracy = sequence_accuracy(self, dev_set)

        print 'With all parameters = 0, the accuracy is %2.2f%%' % (old_accuracy*100)
        for i in range(10):
            print 'Beginning trip', i+1, 'through the training set.'
            for j in range(num_batches):
                batch = training_set[j*batch_size:(j+1)*batch_size]
                total_expected_feature_count.fill(0)
                total_expected_transition_count.fill(0)
                total_observed_feature_count, total_observed_transition_count = \
                    self.compute_observed_count(batch)

                for sequence in batch:
                    transition_matrices = self.compute_transition_matrices(sequence)
                    alpha_matrix = self.forward(sequence, transition_matrices)
                    beta_matrix = self.backward(sequence, transition_matrices)
                    expected_feature_count, expected_transition_count = \
                            self.compute_expected_feature_count(sequence, alpha_matrix, beta_matrix, transition_matrices)
                    total_expected_feature_count += expected_feature_count
                    total_expected_transition_count += expected_transition_count

                feature_gradient = (total_observed_feature_count - total_expected_feature_count) / len(batch)
                transition_gradient = (total_observed_transition_count - total_expected_transition_count) / len(batch)

                self.feature_parameters += learning_rate * feature_gradient
                self.transition_parameters += learning_rate * transition_gradient
                # print sequence_accuracy(self, dev_set)

            """Finished a trip through the data, check for convergence"""
            new_accuracy = sequence_accuracy(self, dev_set)
            print '%2.2f%%' % (new_accuracy*100)
            if new_accuracy > old_accuracy: # We're still improving
                """Update parameters"""
                np.copyto(old_transition, self.transition_parameters)
                np.copyto(old_feature, self.feature_parameters)
                old_accuracy = new_accuracy
            else: # We've stopped improving, go back to last best
                print 'Stopped improving!'
                self.transition_parameters = old_transition
                self.feature_parameters = old_feature
                return


    def compute_transition_matrices(self, sequence):
        """Compute transition matrices (denoted as M on the slides)

        Compute transition matrix M for all time steps.

        We add one extra dummy transition matrix at time 0
        for the base case or not. But this will affect how you implement
        all other functions.

        The matrix for the first time step does not use transition features
        and should be a diagonal matrix.

        TODO: Implement this function

        Returns :
            a list of transition matrices
        """
        transition_matrices = []
        num_labels = len(self.label_codebook)
        transition_matrix = np.zeros((num_labels, num_labels))
        transition_matrices.append(transition_matrix)

        # special diagonal matrix of priors
        transition_matrix = np.zeros((num_labels, num_labels))
        # build matrix a row at a time (for each 'from' label),
        # except in this edge case, ignore the transition features
        for fromIndex in range(len(self.label_codebook)):
            transition_matrix[fromIndex,fromIndex] = exp(
                # just feature lambdas
                sum(self.feature_parameters[fromIndex,sequence[0].feature_vector])
            )

        transition_matrices.append(transition_matrix)

        for t in range(1,len(sequence)):
            # compute transition matrix
            transition_matrix = np.zeros((num_labels, num_labels))

            # build matrix a row at a time (for each 'from' label)
            for fromIndex in range(len(self.label_codebook)):
                transition_matrix[fromIndex] = [exp(
                    # transition lambdas plus feature lambdas
                    self.transition_parameters[fromIndex, toIndex]
                    + sum(self.feature_parameters[fromIndex,sequence[t].feature_vector])
                ) for toIndex in range(len(self.label_codebook))]

            transition_matrices.append(transition_matrix)

        return transition_matrices

    def forward(self, sequence, transition_matrices):
        """Compute alpha matrix in the forward algorithm

        TODO: Implement this function
        """
        num_labels = len(self.label_codebook)
        alpha_matrix = np.ones((num_labels, len(sequence) + 1))
        for t in range(1,len(sequence) + 1):
            # alpha_matrix[:,[t]] = transition_matrices[t].dot(alpha_matrix[:,[t-1]])
            alpha_matrix[:,t] = alpha_matrix[:,t-1].dot(transition_matrices[t])
        return alpha_matrix

    def backward(self, sequence, transition_matrices):
        """Compute beta matrix in the backward algorithm

        TODO: Implement this function
        """
        num_labels = len(self.label_codebook)
        beta_matrix = np.ones((num_labels, len(sequence) + 1))
        time = range(1,len(sequence) + 1)
        time.reverse()
        for t in time:
            beta_matrix[:,[t-1]] = transition_matrices[t].dot(beta_matrix[:,[t]])
        return beta_matrix

    def decode(self, sequence):
        """Find the best label sequence from the feature sequence

        TODO: Implement this function

        Returns :
            a list of label indices (the same length as the sequence)
        """
        transition_matrices = self.compute_transition_matrices(sequence)
        decoded_sequence = [[a for a in self.label_codebook.values()] for b in range(len(sequence))]
        scores = [[a for a in self.label_codebook.values()] for b in range(len(sequence))]

        # log probabilities for first step
        for label in self.label_codebook.values():
            decoded_sequence[0][label] = [label]
            scores[0][label] = transition_matrices[1][label, label]

        # for each inductive step, compute the best score for the each current state given any previous state
        for i in range(1,len(sequence)):
            for newLabel in self.label_codebook.values():
                # bestPrevLabel = np.argmax(scores[i-1])                                  #
                # possibles = [transition_matrices[i+1][bestPrevLabel,newLabel]]          #
                # compute two possible values for this state given previous state, keep max
                possibles = [transition_matrices[i+1][prevLabel,newLabel] + scores[i-1][prevLabel]
                    for prevLabel in range(len(self.label_codebook))]
                bestPrevLabel = np.argmax(possibles)
                scores[i][newLabel] = max(possibles)
                # assert(max(possibles) == possibles[bestPrevLabel]) # argmax sanity check
                decoded_sequence[i][newLabel] = decoded_sequence[i-1][bestPrevLabel] + [newLabel]

        return decoded_sequence[-1][np.argmax(scores[-1])]


        # The below implementation is my earlier broken Viterbi mentioned in my write up
        # -------------------------------------------------------------------------------
        # decoded_sequence = range(len(sequence))
        # score = {label: sum(self.feature_parameters[label, sequence[0].feature_vector])
        #      for label in self.label_codebook.values()}
        # decoded_sequence[0] = score.keys()[score.values().index(max(score.values()))]
        # for i in range(1,len(sequence)):
        #     # log probabilities for each step
        #     scores = {label: sum(self.feature_parameters[label, sequence[i].feature_vector])
        #         + self.transition_parameters[decoded_sequence[i-1],label]
        #         for label in self.label_codebook.values()}
        #     decoded_sequence[i] = scores.keys()[scores.values().index(max(scores.values()))]
        # return decoded_sequence

    def compute_observed_count(self, sequences):
        """Compute observed counts of features from the minibatch

        This is implemented for you

        Returns :
            A tuple of
                a matrix of feature counts
                a matrix of transition-based feature counts
        """
        num_labels = len(self.label_codebook)
        num_features = len(self.feature_codebook)

        feature_count = np.zeros((num_labels, num_features))
        transition_count = np.zeros((num_labels, num_labels))
        for sequence in sequences:
            for t in range(len(sequence)):
                if t > 0:
                    transition_count[sequence[t-1].label_index, sequence[t].label_index] += 1
                feature_count[sequence[t].label_index, sequence[t].feature_vector] += 1

        return feature_count, transition_count

    def compute_expected_feature_count(self, sequence,
            alpha_matrix, beta_matrix, transition_matrices):
        """Compute expected counts of features from the sequence

        TODO: Complete this function by implementing
        expected transition feature count computation
        Be careful with indexing on alpha, beta, and transition matrix

        Returns :
            A tuple of
                a matrix of feature counts
                a matrix of transition-based feature counts
        """
        num_labels = len(self.label_codebook)
        num_features = len(self.feature_codebook)

        feature_count = np.zeros((num_labels, num_features))
        transition_count = np.zeros((num_labels, num_labels))
        sequence_length = len(sequence)
        Z = np.sum(alpha_matrix[:,-1])

        #gamma = alpha_matrix * beta_matrix / Z
        gamma = np.exp(np.log(alpha_matrix) + np.log(beta_matrix) - np.log(Z))
        for t in range(sequence_length):
            for j in range(num_labels):
                feature_count[j, sequence[t].feature_vector] += gamma[j, t]

        # for t in range(1,sequence_length):
        #     for i in range(num_labels):
        #         for j in range(num_labels):
        #             transition_count[i,j] += np.exp(np.log(alpha_matrix[i,t-1]) +
        #             np.log(transition_matrices[t][i,j]) + np.log(beta_matrix[j,t]) - np.log(Z))

        for t in range(sequence_length - 1):
            transition_count += (transition_matrices[t] * np.outer(alpha_matrix[:, t], beta_matrix[:,t+1])) / Z

        return feature_count, transition_count

def sequence_accuracy(sequence_tagger, test_set):
    correct = 0.0
    total = 0.0
    for sequence in test_set:
        decoded = sequence_tagger.decode(sequence)
        assert(len(decoded) == len(sequence))
        total += len(decoded)
        for i, instance in enumerate(sequence):
            if instance.label_index == decoded[i]:
                correct += 1
    return correct / total
