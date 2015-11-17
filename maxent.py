# -*- mode: Python; coding: utf-8 -*-

from classifier import Classifier
import numpy as np
from math import exp, log
import scipy.misc
import sys

class MaxEnt(Classifier):

    def __init__(self, labels, features, model={}):
        super(MaxEnt, self).__init__(model)
        self.ling_features = features
        self.labels = labels

    def get_model(self):
        return {'params':self.model_params,
                'labels':self.labels,
                'features':self.ling_features}

    def set_model(self, model):
        """The constructor has a bug (in the provided code). This is my fix."""
        if len(model) == 0: return

        self.model_params = model['params']
        self.labels = model['labels']
        self.ling_features = model['features']

    model = property(get_model, set_model)

    def train(self, instances, dev_instances=None):
        """Construct a statistical model from labeled instances."""
        self.model_params = np.zeros( (len(self.labels), len(self.ling_features)) )

        """Train until converged"""
        self.train_sgd(instances, dev_instances, 0.00005, 30)


    def train_sgd(self, train_instances, dev_instances, learning_rate, batch_size):
        """Train MaxEnt model with Mini-batch Stochastic Gradient """
        gradient = np.zeros(( len(self.labels), len(self.ling_features) ))
        """maintain a window average of likelihood for convergence"""
        old_params =  np.copy(self.model_params) # This will be our 'go back' point when it stops improving
        old_likelihood = float("inf")
        print "%5.3f     %2.1f%%" % (self.nloglikelihood(dev_instances), self.accuracy(dev_instances))

        while True: # While not converged
            for index, instance in enumerate(train_instances):
                gradient += self.gradient_per_instance(instance)
                """update params with gradient at batch_size intervals and check likelihood"""
                if index % batch_size == 0:
                    """Finished a batch, time to update gradient"""
                    self.model_params += gradient * learning_rate

            """Finished a trip through the data. Check for convergence"""
            likelihood = self.nloglikelihood(dev_instances)
            print "%5.3f     %2.1f%%" % (likelihood, self.accuracy(dev_instances))
            if likelihood < old_likelihood: # We're still improving
                """Update parameters"""
                np.copyto(old_params, self.model_params)
                old_likelihood = likelihood
                gradient[:] = 0
            else: # We've stopped improving. Return with last good parameters
                print 'Stopped improving!'
                self.model_params = old_params
                return

    def decode(self, instance): return self.classify(instance)
    def classify(self, instance):
        """compute feature dot model_params as a proxy for likelihood for each label"""
        labelscore = {lab: sum(self.model_params[index, instance.feature_vector])
        for lab, index in self.labels.items()}
        """return the label with the highest likelihood"""
        return max(labelscore, key=lambda k: labelscore[k])

    def sequence_accuracy(self, instances): return self.accuracy(instances)
    def accuracy(self, instances):
        """Classify and compute accuracy over a set"""
        return float(sum([100 for ins in instances if self.classify(ins)==ins.label]))/len(instances)

    def gradient_per_instance(self, instance):
        """Compute the gradient function for this instance"""
        gradient = np.zeros(( len(self.labels), len(self.ling_features) ))
        """Observed"""
        gradient[self.labels[instance.label], instance.feature_vector] += 1
        """minus Expected (computed one label at a time)"""
        for label, idx in self.labels.items():
            gradient[idx, instance.feature_vector] -= self.posterior(instance, label)
        return gradient

    def posterior(self, instance, label=None):
        """Compute the posterior for this instance and specified label or
        the label from the instance"""
        if not label:
            label = instance.label

        """exp( l dot f - logsumexp( [l dot f foreach label]))"""
        return exp(
            sum(self.model_params[self.labels[label], instance.feature_vector])
            - scipy.misc.logsumexp(
                [sum(self.model_params[lab, instance.feature_vector])
                for lab in range(len(self.labels))]
                )
            )

    def nloglikelihood(self, instances):
        return -(sum(log(self.posterior(inst)) for inst in instances)
            - sum([lam**2 for row in self.model_params for lam in row])) # Penalty term, sigma=1
