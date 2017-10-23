import tensorflow as tf
from math import ceil, floor
import numpy as np
import time

# *** DATA ***
def split_set(ids, repartition_perc, seed = '-1'):
    """
    Split the list of ids between train, dev and test set.
    
    Argument:
    ids               -- numpy array of the ids of picture to split
    repartition_perc  -- [train_set, dev_set, test_set] in % 
    seed              -- positive number used as seed for the random function 
    
    Returns:
    train_set_ids, dev_set_ids, test_set_ids
    """
    
    # test the inputs of the function
    if len(repartition_perc) > 1 and len(repartition_perc) < 4 and not sum(repartition_perc) == 100 
    or repartition_perc[0] < 1:
        # TODO return Error
    
    # Determine the size of the different sets
    len_list_ids = len(ids)
    
    train_set_size = math.ceil(repartition_perc[0] * len_list_ids / 100)
    
    if len(repartition_perc) > 1:
        dev_set_size = math.floor(repartition_perc[1] * len_list_ids / 100)
    else:
        dev_set_size = len_list_ids - train_set_size
    
    if len(repartition_perc) > 2:
        test_set_size = len_list_ids - (train_set_size + dev_set_size)
    else:
        test_set_size = 0
    
    # shuffle ids
    np.random.seed(seed)
    
    permutation = list(np.random.permutation(len_list_ids))
    shuffled_ids = ids[permutation]
    
    # create the different sets
    train_set = shuffled_ids[0:train_set_size]
    dev_set = shuffled_ids[train_set_size:(train_set_size + dev_set_size)]
    test_set = shuffled_ids[(train_set_size + dev_set_size):]
    
    return [train_set, dev_set, test_set]
    
    
    

def load_training_batch(list_ids):

# Weights for either fully connected or convolutional layers of the network
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# Bias elements in either a fully connected or a convolutional layer
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Convolution typically used
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Max pool to reduce to 1/4 of the input
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

# Actual layer used
def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, W) + b)

# Layer without the ReLU part to be used at the end of the CNN
def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b