
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from sklearn.model_selection import train_test_split


# In[2]:


filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

#more filters, featuer map will b
# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128 


# In[3]:


filenames = os.listdir("data")
label_to_num = {"O":0, "P":1, "Q":2, "S":3, "W":4}

X_list = []
y_list = []
for f in filenames:
    img = image.img_to_array(image.load_img("data/{}".format(f)))
    
    X_list.append(img)
    y_list.append(label_to_num[f.split("_")[1][0]])
    
    
    
y_len = len(y_list)
y_onehot = np.zeros((y_len, 5))
y_onehot[np.arange(y_len),y_list]=1


# In[4]:


x_train, x_test, y_train, y_test = train_test_split(np.array(X_list), y_onehot)


# In[5]:


img_size = 25

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size *3

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
#https://en.wikipedia.org/wiki/Channel_(digital_image)
#channels mean number of primary colors
num_channels = 3

# Number of classes, one class for each of 10 digits.
num_classes = 5


# In[6]:


x_train = x_train.reshape(len(x_train),img_size_flat)
x_test = x_test.reshape(len(x_test),img_size_flat)


# In[7]:


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))



def new_biases(length):
    #equivalent to y intercept
    #constant value carried over across matrix math
    return tf.Variable(tf.constant(0.05, shape=[length]))


# In[8]:



def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.


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


# In[9]:


def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    num_features = layer_shape[1:4].num_elements()
    
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features


# In[10]:


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


# In[11]:


x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')


# In[12]:


x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])


# In[13]:


y_true = tf.placeholder(tf.float32, shape=[None, 5], name='y_true')


# In[14]:


y_true_cls = tf.argmax(y_true, dimension=1)


# In[15]:


layer_conv1, weights_conv1 =     new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)
    
    
    

layer_conv2, weights_conv2 =     new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)
    
    
    
layer_flat, num_features = flatten_layer(layer_conv2)



layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)


layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)



# In[16]:



y_pred = tf.nn.softmax(layer_fc2)

y_pred_cls = tf.argmax(y_pred, dimension=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)

cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[25]:


session = tf.Session()
session.run(tf.global_variables_initializer())


# In[26]:


train_batch_size = 64
total_iterations = 0

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        
        indices = np.random.choice(len(x_train), train_batch_size)
        
        x_batch, y_true_batch = x_train[indices], y_train[indices]

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))
            
            images = x_test

            # Get the associated labels.
            labels = y_test

            # Create a feed-dict with these images and labels.
            feed_dict = {x: images,
                         y_true: labels}

            # Calculate the predicted class using TensorFlow.
            acc = session.run(accuracy, feed_dict=feed_dict)

            # Convenience variable for the true class-numbers of the test-set.


            # Print the accuracy.
            msg = "Accuracy on Test-Set: {0:.1%}"
            print(msg.format(acc))

    
    
    
    
    


# In[33]:


optimize(1000)

