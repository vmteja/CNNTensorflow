# CNNTensorflow
 implement a convolutional neural network program in TensorFlow capable of learning the zener card images

Your neural net training program will be run from the command line with a line like:

python conv_train.py cost network_description epsilon max_updates class_letter model_file_name train_folder_name 

The parameters, epsilon, max_updates, class_letter, model_file_name, train_folder_name are the same as in Hw2. An update here will mean one epoch. cost should be one of cross, cross-l1, cross-l2, or ctest which says whether training will be done using just cross entropy, cross entropy with L1 regularization, cross entropy with L2 regularization, or no training just testing (epsilon max_updates are then ignored). network_description is the name of a file that should consist of a sequence of rows in the format:

feature_size num_features

This should be followed by a row with a number of units for a dense layer. For example, a network_description file might look like:

5 4
6 8
6 16
64

This would specify a neural net with the following layers: the first two layers would consist of a convolutional layer with 4 feature maps using a 5x5 filter followed by a maxpool layer, the next two layers would consist of a convolutional layer with 8 feature maps using a 6x6 filter followed by a maxpool layer, the next two layers consist of a convolutional layer with 16 feature maps using a 6x6 filter followed by a maxpool layer, finally, the last two layers consist of a dense layer of 64 units all of whose ouputs connect to a single sigmoid perceptron. For the purposes of this homework all other units use relu activations.

You should train with 5-fold cross-validation. Your program should output a final average cost value for the training data (over the five different training sets) and a final average cost for the validation data (over the five different training sets). You can use the last trained model as what you write to disk. When you do testing with your trained model, use a different data set then what you used to train and validate with.

Once you have written the above program, I would like you to design and conduct experiments which investigate the following:

    Fix a network approximately like LeNet-5. If you plot max_updates versus training cost and validation cost does the validation cost ever reach a minimum? How does a lower validation cost affect accuracy on test data?
    Start with a network approximately like LeNet-5. How does the choice of regularization effect the training convergence rate of your model?
    Start with a network approximately like LeNet-5. What is the effect of increasing or decreasing the number of layers on training accuracy? What about varying feature size or number of features/layer?
