import sys
from random import shuffle
import tensorflow as tf
from sklearn.model_selection import KFold
import model


def __save_session(session, model_file):
    saver = tf.train.Saver()
    with tf.variable_scope('', reuse=True):
        saver.save(session, model_file)
        print('Trained Model Saved.')

#
# def __train(session, network_info, train_features, train_labels, max_updates, cost):
#     return


def __load_session(filename):
    session = tf.Session()
    saver = tf.train.Saver()
    saver.restore(session, filename)
    return session


# def __test(model_file, test_features, test_labels):
#     model.test_model(model_file, test_features, test_labels)


def cv_5fold_trainer(model_file, network_info, X, y, max_updates, cost):
    session = tf.Session()
    # model.train_model(session, network_info, X, y, X, y, max_updates, cost)
    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.train_model(session, network_info, X_train, y_train, X_test, y_test, max_updates, cost)

# MODE_HANDLERS = {
#     "train": trainer,
#     "5fold": cv_5fold_trainer,
#     "test": tester
# }
#
# if __name__ == "__main__":
#     mode = sys.argv[1]
#     model_file = sys.argv[2]
#     data_folder = sys.argv[3]
#     try:
#         handler = MODE_HANDLERS[mode]
#         data = data_loader.load(data_folder)
#         handler(model_file, data)
#     except KeyError:
#         print "Invalid Mode", mode
