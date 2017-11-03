import tensorflow as tf
import numpy as np
import math                     # To create the sets and batches              
import time                     # To generate some random seeds
import numbers                  # To test if a variable is a number

# Import kitti_utils from a different project
import sys
sys.path.insert(0, '/data2/lucas/Projects/Kitti2012')
from kitti_utils import *    # fcts. to manage the kitti dataset

# List of all the possible classes
LIST_CLASSES = ['Car', 'Van', 'Truck',
                'Pedestrian', 'Person_sitting', 
                'Cyclist', 'Tram', 'Misc', 
                'DontCare']

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

# *** CONVERSION ***

def convert_labels_to_array(labels, S, B, im_shape, list_classes = LIST_CLASSES):
    """
    Convert the dictionary to an array similar to the output of the YOLO CNN
    
    Argument:
    labels        -- Dictionary containing information about boxes
    S             -- Number of cell grid per dimension
    B             -- Number of boxes per cell grid
    im_shape      -- [image_height, image_width]
    list_classes  -- List of all the classes possible (use to create the one-hot vector)
    
    Returns:
    arr_labels    -- Array of length (S * S * (B * 5 + C)) containing the labels
    elems_discard -- List containing discarded elements [x_cell, y_cell, box_center_x_norm, 
                     box_center_y_norm, box_width_norm, box_height_norm, box_IoU, class_index]
    """
    # Get the number of classes from the list of classes
    C = len(list_classes)
    
    # Get cells shape from the image shape
    cell_width = im_shape[1] / S
    cell_height = im_shape[0] / S
    
    # Create the empty numpy array to contain all the labels
    arr_labels = np.zeros(S * S * (B * 5 + len(list_classes)))
    
    # Create an empty list to store all the object discarded
    elems_discard = []
    
    for label in labels:
        # --- EXPORT DATA ---

        # Extract from Paper:
        # "Each bounding box consists of 5 predictions: x, y, w, h,
        # and confidence. The (x, y) coordinates represent the center
        # of the box relative to the bounds of the grid cell. The width
        # and height are predicted relative to the whole image."

        # *** POSITION DATA ***

        # Width and Height of the box in pixel size
        box_width = label['bbox']['x_max'] - label['bbox']['x_min']  
        box_height = label['bbox']['y_max'] - label['bbox']['y_min']

        # Normalize the width and height of the box
        box_width_norm = box_width / im_shape[1]
        box_height_norm = box_height / im_shape[0]

        # Center of the box in pixel coordinates
        x_center = (label['bbox']['x_min'] + (box_width/2))
        y_center = (label['bbox']['y_min'] + (box_height/2))

        # Coordinates of the cell the center of the object is in (from 0 to ...)
        x_cell = math.floor(x_center / im_shape[1] * S)
        y_cell = math.floor(y_center / im_shape[0] * S)

        # Coordinates of the center of the box relative to the box
        box_center_x_norm = (x_center - x_cell * cell_width) / cell_width
        box_center_y_norm = (y_center - y_cell * cell_height) / cell_height

        box_IoU = 1   # As it is the ground truth

        box_info = [box_center_x_norm, box_center_y_norm,
                    box_width_norm, box_height_norm, box_IoU]

        # *** CLASS DATA ***

        class_proba = np.zeros(len(list_classes))

        index_class = list_classes.index(label['type'])

        class_proba[index_class] = 1

        # --- ADD DATA ---
        # Add the label to the array of labels

        # Extract the grid cell data and the different info
        idx_start = y_cell * S * (B * 5 + C) + x_cell * (B * 5 + C)
        idx_end = idx_start + (B * 5 + C)
        cell_data = arr_labels[idx_start : idx_end]

        # Extract objects and class probabilities
        boxes_info = cell_data[:-C].reshape((B,-1))
        cell_class_proba = cell_data[-C :]

        # If the cell is empty
        if not cell_class_proba.any():
            # Add the box in the first position
            boxes_info[0][:] = box_info
            # Add the class probabilities
            cell_class_proba = class_proba

        else:
            # Compute area of the boxes
            box_area = np.prod(box_info[2:4])
            
            # Create a boolean vector to compare all the boxes area
            box_bigger_than_boxes = np.zeros(B) # Initialize Boolean vector
            
            for x in range(B):
                current_box_area = np.prod(boxes_info[x][2:4])
                box_bigger_than_boxes[x] = box_area > current_box_area

            # Compare the classes
            same_class = np.array_equal(class_proba, cell_class_proba)
                
            is_class_misc_or_dontcare = np.argmax(class_proba) > 6

            # The object is the same class as the object already inside
            if same_class:
                insert_id = np.argmax(box_bigger_than_boxes) if box_bigger_than_boxes.any() else B
                boxes_info = np.insert(boxes_info, insert_id, box_info, axis=0)
                
                # Get the last object which is going to be discarded
                if boxes_info[-1].any():
                    elems_discard.append([x_cell, y_cell] + boxes_info[-1].tolist() + [index_class])
                
                # Discard the last entry of the boxes_info to only keep B objects
                boxes_info = boxes_info[:B]

            elif (not same_class) and box_bigger_than_boxes[0] and not is_class_misc_or_dontcare:
                # Get the elements in the list before they get discarded
                for i in range(B):
                    if boxes_info[i].any():
                        elems_discard.append([x_cell, y_cell] + boxes_info[i].tolist() + [index_class])
                
                # If different class and box bigger than box_1
                boxes_info *= 0
                boxes_info[0] = box_info
                cell_class_proba = class_proba
            
            else:
                elems_discard.append([x_cell, y_cell] + box_info + [index_class])

        # Create new cell_data
        new_cell_data = np.concatenate((boxes_info.flatten(), cell_class_proba))

        # Insert new data in the cell array
        arr_labels[idx_start : idx_end] = new_cell_data 
        
    return arr_labels, elems_discard

def create_box_info_from_arr(box_info, box_type, box_type_proba, im_shape, S, cell_coord):
    """
    Create the dictionary from the info given
    
    Argument:
    box_info          -- [x_center_norm, y_center_norm, width_box_norm, height_box_norm, confidence]
    box_type          -- String containing the name of the type (e.g. 'Car')
    box_type_proba    -- Probability of the type
    im_shape          -- [image_height, image_width]
    S                 -- Number of cell grid per dimension
    cell_coord.       -- [cell_x, cell_y]
    
    Returns:
    Return dictionary containing {'bbox', 'type', 'score'}
    """
    # Get cells shape from the image shape
    cell_width = im_shape[1] / S
    cell_height = im_shape[0] / S
    
    # Get box shape
    box_width = box_info[2] * im_shape[1]
    box_height = box_info[3] * im_shape[0]
    
    # Get coordinates of the box's center
    box_center_x = (box_info[0] * cell_width) + (cell_coord[0] * cell_width)
    box_center_y = (box_info[1] * cell_height) + (cell_coord[1] * cell_height)
    
    # Get bbox values
    x_min = round(box_center_x - box_width/2, 2)
    x_max = round(box_center_x + box_width/2, 2)
    
    y_min = round(box_center_y - box_height/2, 2)
    y_max = round(box_center_y + box_height/2, 2)
    
    # Create the bbox dictionary
    bbox = {'x_max': x_max,
            'x_min': x_min,
            'y_max': y_max,
            'y_min': y_min}
    
    # Compute the score
    IoU = box_info[4]
    score = IoU * box_type_proba
    
    # Create final dictionary
    box = {'bbox': bbox,
           'type': box_type,
           'score': score}
    
    return box

def convert_array_to_labels(arr_labels, S, B, im_shape, list_classes = LIST_CLASSES, data_per_obj = 5):
    """
    Convert an array of labels into a dictionary of labels
    
    Argument:
    arr_labels    -- array containing the labels
    S             -- Number of cell grid per dimension
    B             -- Number of boxes per cell grid
    im_shape      -- [image_height, image_width]
    list_classes  -- List of all the classes possible (use to create the one-hot vector)
    data_per_obj  -- Number of data points per objects (5 in the original implementation of YOLO)
    
    Returns:
    Return list of dictionaries containing the labels
    """
    # Get the number of classes from the list of classes
    C = len(list_classes)
    
    # Reshape the array to correspond to the grid shape
    arr_labels = arr_labels.reshape((S, S, data_per_obj*B+C))

    # Create list of labels

    labels = []

    for cell_x in range(S):
        for cell_y in range(S):
            cell_grid = arr_labels[cell_y][cell_x]

            if cell_grid.any():
                # Extract the info from the cell
                
                # -- TODO --
                # separate class_proba from boxes_info
                # Then use reshape on boxes info to separate 
                # them and be able to use iteration on the list
                
                boxes_info = cell_grid[:-C].reshape((B,-1))
                class_proba = cell_grid[-C:]

                # Find the type of the objects
                cell_type = list_classes[np.argmax(class_proba)]
                cell_type_proba = np.max(class_proba)
                
                for x in range(B):
                    if boxes_info[x].any():
                        labels.append(
                            create_box_info_from_arr(box_info = boxes_info[x],
                                                     box_type = cell_type,
                                                     box_type_proba = cell_type_proba,
                                                     im_shape = im_shape,
                                                     S = S,
                                                     cell_coord = [cell_x, cell_y])
                        )    
    return labels

# *** IMPORT BATCH ***
def load_batch_for_training(ids, im_size, S, B, list_classes = LIST_CLASSES):
    """
    Load a batch of training data (image + labels)
    
    Arguments:
    ids               -- array containing the list of ids of the data
    im_size           -- size to which the input images need to be resized
    S                 -- Number of cell grid per dimension
    B                 -- Number of boxes per cell grid
    list_classes      -- List of all the classes possible
    
    Returns:
    batch_im [batch_size, im_size, im_size, 3], batch_labels [batch_size, S*S*(5*B+C)]
    """
    
    batch_size = len(ids)
    C = len(list_classes)
    
    batch_input = np.zeros((batch_size, im_size, im_size, 3))
    batch_labels = np.zeros((batch_size, S*S*(5*B+C)))
    
    for i in range(batch_size):
        # Import image
        im = import_im(ids[i], 'train')
        
        # Import labels
        labels = import_labels(ids[i], 'train')
        
        arr_labels, _ = convert_labels_to_array(labels, S, B, im.shape, list_classes)
        batch_labels[i, :] = arr_labels
        
        # Resize the image and add it to the batch_input
        batch_input[i, :, :, :] = misc.imresize(im, (im_size, im_size, 3))
        
    return [batch_input, batch_labels]

# *** LAYERS ***

# Weights for either fully connected or convolutional layers of the network
def weight_variable(shape):
    initial = tf.random_normal(shape, stddev=0.35, dtype = tf.float32)
    return tf.Variable(initial)

# Bias elements in either a fully connected or a convolutional layer
def bias_variable(shape):
    initial = tf.random_normal(shape, stddev = 0.35, dtype = tf.float32)
    return tf.Variable(initial)

# Convolution typically used
def conv2d(x, W, input_strides):
    return tf.nn.conv2d(x, W, strides=input_strides, padding='VALID')

# Max pool to reduce to 1/4 of the input
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

# Convolutional layer with strides and alpha adjustable
def conv_layer(input_layer, shape, strides = [1, 1, 1, 1], alpha = 0.1):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    
    # Reproducing explicit padding used by darknet
    # https://github.com/johnwlambert/YoloTensorFlow229/blob/1bf43059b37c8a1ed4be0fa0bb2f5d79ba881fb3/yolo.py#L107
    pad = [int(shape[0]/2), int(shape[1]/2)]
    input_layer_padded = tf.pad(input_layer, 
                                paddings = [[0, 0], [pad[0], pad[0]], 
                                            [pad[1], pad[1]], [0, 0]])
    
    Z = conv2d(input_layer_padded, W, strides) + b
    return tf.maximum(Z, alpha * Z)
    

# Layer without the ReLU part to be used at the end of the CNN
def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b