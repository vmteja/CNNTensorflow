
import tensorflow as tf
import numpy as np
from model_helper_func import *
import os

num_classes = 2
batch_size = 64

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,
                   num_input_channels,
                   filter_size,
                   num_filters,
                   use_pooling=True):

    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)
    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    layer = tf.nn.relu(layer)
    return layer, weights

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases
    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

def neural_network(network_info, X):
    no_layers = len(network_info)-1

    #initialize layers, weights
    layers_conv = []
    weights_conv = []
    for i in range(no_layers+1):
        temp = []
        layers_conv.append(temp)
        weights_conv.append(temp)

    for i in range(no_layers):
        if i == 0:
           layers_conv[i], weights_conv[i] =  new_conv_layer(input=X, num_input_channels=3,
                                                            filter_size = network_info[0][0],
                                                            num_filters = network_info[0][1],
                                                            use_pooling=True)
        else:
           layers_conv[i], weights_conv[i] =  new_conv_layer(input=layers_conv[i-1], num_input_channels=network_info[i-1][1],
                                                            filter_size = network_info[i][0],
                                                            num_filters = network_info[i][1],
                                                            use_pooling=True)
    i = no_layers-1
    layer_flat, num_features = flatten_layer(layers_conv[i])

    layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features,
                             num_outputs=network_info[-1][0],
                             use_relu=True)

    layer_fc2 = new_fc_layer(input=layer_fc1, num_inputs=network_info[-1][0],
                             num_outputs=num_classes,
                             use_relu=False)
    return layer_fc2

def train_model(session, network_info, train_features, train_labels, test_features, test_labels, max_updates, cost):
    # set parameters 
    img_size = 25
    num_channels = 3

    # training parameters
    learning_rate = 0.01
    no_epochs = max_updates

    # defining X and Y for model 
    img_size_flat = img_size * img_size *3
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
    features = tf.reshape(x, [-1, img_size, img_size, num_channels])
    labels = tf.placeholder(tf.float32, shape=[None, 2], name='y_true')
    y_true_cls = tf.argmax(labels, dimension=1)

    # re-sizing the read data 
    train_features = np.array(train_features)
    train_features = train_features.reshape(len(train_features),img_size_flat)
    train_labels = np.array(train_labels)

    #data size
    data_size = len(train_features)

    # network architecuture
    layer_fc2 = neural_network(network_info, features)

    # output prediction
    y_pred = tf.nn.softmax(layer_fc2)
    y_pred_cls = tf.argmax(y_pred, dimension=1)

    # Calculate accuracy
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Define loss
    cost_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=labels))

    # selecting regularization using input parameter  
    weights = tf.trainable_variables()
    if cost == 'cross-l1':
       l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.005, scope=None)
       regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
    elif cost == 'cross-l2':
       l2_regularizer = tf.contrib.layers.l2_regularizer(scale=0.005, scope=None)
       regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights)
    else:
       regularization_penalty = 0

    # adding regularization for the cost function
    regularized_loss = cost_loss + regularization_penalty

    # defining optimizer 
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(regularized_loss)

    # cal the no of batches 
    no_batches = math.ceil(data_size/batch_size)


    # not initilizing as we are getting session from top function
    init = tf.global_variables_initializer()
    session.run(init)
    training_accuracy = []
    # sess.run() # starting computational graph
    for epoch in range(no_epochs):
        batch_count = 0
        # Loop over all batches
        for batch_features, batch_labels in batches(batch_size, train_features, train_labels):
            feed_dict_train = {x : batch_features, labels: batch_labels}
            #sess.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})
            session.run(optimizer, feed_dict= feed_dict_train)
            training_accuracy.append(session.run(accuracy, feed_dict=feed_dict_train))
    print "training accuracy is : %.3f"%(sum(training_accuracy) / len(training_accuracy))

        # printing accuracy for each epoch
        #acc = sess.run(accuracy, feed_dict=feed_dict_train)
        #print "training accuracy at epoch %d is : %.3f"%(epoch,acc)

    print("--- trainig complete ----")
#     return session
#
# def test_model(sess, test_features, test_labels):

    print("--- testing start ---")
    # parameters
    img_size = 25
    img_size_flat = img_size * img_size *3

    # re-sizing the read data
    test_features = np.array(test_features)
    test_features = test_features.reshape(len(test_features),img_size_flat)
    test_labels = np.array(test_labels)

    # size of train data
    data_size = len(test_features)

    # sess.run()
    test_accuracies = []
    nbatches = batches(batch_size, test_features, test_labels)
    for batch_features, batch_labels in nbatches:
        feed_dict_train = {x: batch_features, labels: batch_labels}
        # sess.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})
        # session.run(optimizer, feed_dict=feed_dict_train)
        # printing status for every 5 batches
        # percent_done = (batch_count / no_batches) * 100
        # print "trained model with %d percent  of total batches" %(percent_done) # replace it with progress bar
        test_accuracies.append(session.run(accuracy, feed_dict=feed_dict_train))
    print "testing  accuracy is : %.3f" % (sum(test_accuracies) / len(test_accuracies))

    #test_accuracy = session.run(accuracy, feed_dict={features: test_features, labels:test_labels })
    #print('Test Accuracy: {}'.format(test_accuracy))

