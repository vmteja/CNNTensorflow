import sys

from model import *
from data_loader import *
from train_test import cv_5fold_trainer


def analyze_network_description(file_name):
    data = []
    with open(file_name, "r") as fp:
        for line in fp:
            temp = line.rstrip('\n').split(' ')
            data.append(temp)
    for index, val in enumerate(data[:-1]):
        data[index][0] = int(data[index][0])
        data[index][1] = int(data[index][1])
    data[-1][0] = int(data[-1][0])
    return data


if __name__ == "__main__":

    if len(sys.argv) != 8:
        raise Exception("Invalid number of arguments")

    cost = sys.argv[1]
    network_description = sys.argv[2]
    epsilon = sys.argv[3]
    max_updates = int(sys.argv[4])
    class_letter = sys.argv[5]
    model_file_name = sys.argv[6]
    train_folder_name = sys.argv[7]

    # analysze network description
    network_info = analyze_network_description(network_description)

    # reading the data from disk and converting to arrays. Here 'Y' is a one hot enoded label
    # data = load_data(train_folder_name, class_letter)
    train_features, train_labels = read_data(train_folder_name, class_letter)

    # train_model takes data as two lists : features and their respective labels along with other features

    cv_5fold_trainer(model_file_name, network_info, train_features, train_labels, max_updates, cost)
    # session = train_model(network_info, train_features, train_labels, max_updates, cost)

print("testing the model")  # test the trained_model
# test_model(model_file_name, network_info, train_features, train_labels)
