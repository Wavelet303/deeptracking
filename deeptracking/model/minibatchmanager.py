"""
    Utility to load and handle model's dataset

    date : 2016-09-14
"""
__author__ = "Mathieu Garon"
__version__ = "0.0.1"

import numpy as np
import time
from deeptracking.model.parallelminibatch import ParallelMinibatch


class MinibatchManager(ParallelMinibatch):
    def __init__(self, dataset, minibatch_size, max_parallel_buffer=0):
        ParallelMinibatch.__init__(self, max_parallel_buffer)
        self.label_tensor = None
        self.prior_tensor = None
        self.dataset = dataset
        self.minibatch_size = minibatch_size
        self.label_tensor = self.dataset.get_labels()
        self.prior_tensor = self.dataset.get_priors()
        self.image_size = self.dataset.extract_image_size()

    def load_minibatch(self, task):
        input_minibatch = np.ndarray((len(task), 8, self.image_size, self.image_size), dtype=np.float32)
        prior_minibatch = np.ndarray((len(task), 7), dtype=np.float32)
        label_minibatch = np.ndarray((len(task), 6), dtype=np.float32)
        for i, permutation in enumerate(task):
            self.dataset.load_input(permutation, input_minibatch, i)
            prior_minibatch[i] = self.prior_tensor[permutation]
            label_minibatch[i] = self.label_tensor[permutation]
        return input_minibatch, prior_minibatch, label_minibatch

    def get_sample(self, index):
        input_minibatch = np.ndarray((1, 8, self.image_size, self.image_size), np.float32)
        self.dataset.load_input(index, input_minibatch, 0)
        return input_minibatch

    def compute_minibatches_permutations_(self):
        return self.dataset.get_permutations(self.minibatch_size)


