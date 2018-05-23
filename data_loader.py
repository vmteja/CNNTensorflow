import numpy as np
import os
from PIL import Image
from keras.preprocessing import image


#01 for same model name
#10 for diff

def read_data(train_folder_name, class_letter):
    filenames = os.listdir(train_folder_name)

    X_list = []
    y_list = []
    for f in filenames:
        img = image.img_to_array(image.load_img("train_data/{}".format(f)))
        X_list.append(img)
        symbol_name = f.split("_")[1][0]
        if symbol_name.lower() == class_letter.lower():
           y_list.append(1)
        else:
           y_list.append(0)

    y_len = len(y_list)
    y_onehot = np.zeros((y_len, 2))
    y_onehot[np.arange(y_len),y_list]=1
    return np.array(X_list), y_onehot

def flatten_image(foldername):
    filenames = os.listdir(foldername)
    X_2D_vector = np.array([np.array(Image.open(foldername + "/" + filename).convert('L'), 'f') for filename in filenames])
    X_flatten = []
    for x in X_2D_vector:
        b = x.flatten()
        X_flatten.append(b)

    #print np.array(X_flatten)
    return np.array(X_flatten)

def label_encoder(foldername, class_letter):
    labels=[]
    filenames = os.listdir(foldername)
    for filename in filenames:
        if(filename[2].lower() == class_letter.lower()):
            labels.append([0,1])
        else:
            labels.append([1,0])
            # 01 for same model name
            # 10 for diff
    #print labels
    return labels

def load_data(foldername, class_letter):
    a = flatten_image(foldername)
    b = label_encoder(foldername, class_letter)
    data = zip(a,b)
    return data


