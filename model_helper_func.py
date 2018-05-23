

import random
from random import shuffle
import math
import gc #garbage collector

def seperate_data_lables(data):
    """
    seperates data and their respective labels 
    returns two lists; one a list of data elements (40-leng numeric arrays),
    second list is the labels at their their corresponding indexes 
    """
    features = []
    labels = []
    for element in data:
        #print(element)
        features.append(element[0])
        labels.append(element[1][0])
    return features, labels   


def batches(batch_size, features, labels):
    """
    creates batches of features and labels
    returns: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    output_batches = []

    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        output_batches.append(batch)
    return output_batches
