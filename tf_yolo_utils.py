import tensorflow as tf
import numpy as np
from math import ceil, floor    # To create the sets and batches              
import time                     # To generate some random seeds
import numbers                  # To test if a variable is a number

# *** DATA ***
def split_random_set(ids, repartition_perc, seed = -1):
    """
    Split the list of ids between train, dev and test set.
    
    Argument:
    ids               -- numpy array of the ids of picture to split
    repartition_perc  -- [train_set_percentage, dev_set_percentage, test_set_percentage] in % 
    seed              -- positive number used as seed for the random function 
    
    Returns:
    train_set_ids, dev_set_ids, test_set_ids
    """
    # if repartition_perc is an int, cast it to an array 
    if isinstance(repartition_perc, numbers.Number) :
        repartition_perc = np.array([repartition_perc])
    
    # test the inputs of the function
    if (len(repartition_perc) > 1 and len(repartition_perc) < 4 and not sum(repartition_perc) == 100):
        raise Exception('The sum of the percentages must be 100')
        
    if repartition_perc[0] < 1:
        raise Exception('The train_set_percentage must be >0')
    
    if len(repartition_perc) > 3:
        raise Exception('The repartition between sets must be of the form: [train_set_percentage, dev_set_percentage, test_set_percentage]')
    
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
    if seed > -1:
        np.random.seed(seed)
    else:
        np.random.seed(int(time.time()))
    
    permutation = list(np.random.permutation(len_list_ids))
    shuffled_ids = ids[permutation]
    
    # create the different sets
    train_set = shuffled_ids[0:train_set_size]
    dev_set = shuffled_ids[train_set_size:(train_set_size + dev_set_size)]
    test_set = shuffled_ids[(train_set_size + dev_set_size):]
    
    return train_set, dev_set, test_set

def create_batches(ids, batch_size):
    """
    Split a set of ids in batches
    
    Arguments:
    set_ids           -- array containing the list of ids to split in batches
    batch_size        -- size of the batches to create
    
    Returns:
    batches_ids  -- list of batches(aka array of ids)
    """
    
    m = len(ids)
    list_batches = []

    # Partition set of IDs. Minus the end case.
    num_complete_batches = math.floor(m/batch_size) # number of batches of size batch_size in your partitionning
    for k in range(0, num_complete_batches):
        batch_ids = ids[k * batch_size : k * batch_size + batch_size]
        list_batches.append(batch_ids)
    
    # Handling the end case (last batch < batch_size)
    if m % batch_size != 0:
        batch_ids = ids[num_complete_batches * batch_size : m]
        list_batches.append(batch_ids)
    
    return list_batches

def prepare_dataset(ids, repartition_perc, batch_size, seed = False):
    """
    Split the array of ids between the different random sets composed of multiple batches
    
    Arguments:
    ids               -- array containing the list of ids of the data
    batch_size        -- size of the batches to create
    repartition_perc  -- [train_set_percentage, dev_set_percentage, test_set_percentage] in % 
    seed              -- positive number used as seed for the random function 
    
    Returns:
    batches_train_set, batches_dev_set, batches_test_set
    """
    
    # If the seed is not set
    if not seed:
        seed = int(time.time())
    
    # Randomize and split the initial set
    train_set, dev_set, test_set = split_random_set(ids, repartition_perc, seed)
    
    # Create batches
    batches_train_set = create_batches(train_set, batch_size)
    batches_dev_set = create_batches(dev_set, batch_size)
    batches_test_set = create_batches(test_set, batch_size)
    
    return batches_train_set, batches_dev_set, batches_test_set

# def load_batch(ids):

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