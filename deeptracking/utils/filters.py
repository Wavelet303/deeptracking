import collections


class MeanFilter:
    def __init__(self, length):
        self.value_list = collections.deque(maxlen=length)

    def compute_mean(self, value):
        self.value_list.append(value)
        mean = None
        for value in self.value_list:
            if mean == None:
                mean = value
            else:
                mean += value
        return mean/len(self.value_list)
