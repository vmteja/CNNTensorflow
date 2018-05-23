
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pylab', 'inline')
import numpy as np
from keras.preprocessing import image
import os
import keras

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


# In[44]:


filenames = os.listdir("data")
label_to_num = {"O":0, "P":1, "Q":2, "S":3, "W":4}

X_list = []
y_list = []
for f in filenames:
    img = image.img_to_array(image.load_img("data/{}".format(f)))
    
    X_list.append(img)
    y_list.append(label_to_num[f.split("_")[1][0]])
    


# In[46]:


X = np.array(X_list)
y = keras.utils.to_categorical(y_list)


# In[49]:


X_train, X_test, y_train, y_test = train_test_split(X,y)


# In[50]:


# Normalize inputs from 0-255 to 0.0-1.0
x_train = x_train.astype('float32')
x_train = x_train / 255.0

x_test = x_test.astype('float32')
x_test = x_test / 255.0


# In[63]:




model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


# In[ ]:


model.fit(X_train, y_train,
          batch_size=32,
          epochs=100,
          validation_data=(X_test, y_test),
          shuffle=True)

