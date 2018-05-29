# CNNTensorflow
 Aim - To implement a convolutional neural network program in TensorFlow capable of learning the zener card images to perform image classfication of 5 classes.

The neural net training program can be run from the command line with a line like:

python conv_train.py cost network_description epsilon max_updates class_letter model_file_name train_folder_name 

The parameters, epsilon, max_updates, class_letter, model_file_name, train_folder_name are the same as in Hw2. An update here will mean one epoch. cost should be one of cross, cross-l1, cross-l2, or ctest which says whether training will be done using just cross entropy, cross entropy with L1 regularization, cross entropy with L2 regularization, or no training just testing (epsilon max_updates are then ignored). network_description is the name of a file that should consist of a sequence of rows in the format:

feature_size num_features

This should be followed by a row with a number of units for a dense layer. For example, a network_description file might look like:

5 4

6 8

6 16

64

This would specify a neural net with the following layers: the first two layers would consist of a convolutional layer with 4 feature maps using a 5x5 filter followed by a maxpool layer, the next two layers would consist of a convolutional layer with 8 feature maps using a 6x6 filter followed by a maxpool layer, the next two layers consist of a convolutional layer with 16 feature maps using a 6x6 filter followed by a maxpool layer, finally, the last two layers consist of a dense layer of 64 units all of whose ouputs connect to a single sigmoid perceptron. All other units use relu activations.

Training is done with 5-fold cross-validation. Program will output a final average cost value for the training data (over the five different training sets) and a final average cost for the validation data (over the five different training sets).

