import tensorflow as tf

def print_tensor(tensor_name, tensor):
    '''
    Print the name of the tensor, its shape and its content. (Use for DEBUG)
    
    Argument:
    tensor_name     -- String containing the name of the tensor
    tensor          -- Tensor to display the information from
    
    '''
    print(tensor_name + ' ' + str(tensor.shape) + ':')
    print(tensor.eval())

def compute_YOLOv1_loss(predictions, ground_truth, S, B, C, LAMBDA_COORD = 5, LAMBDA_NOOBJ = 0.5):
    '''
    Compute the loss for each images using the loss function defined in the YOLOv1 paper.
    
    In the YOLOv1 version, every grid cell can only detect one object per cell
    independently from the number of boxes that can be produce per cell.
    Therefore, we only take into account the first object of the ground_truth.
    
    Argument:
    predictions       -- [BATCH_SIZE, S*S*(5*B+C)] Predictions
    ground_truth      -- [BATCH_SIZE, S*S*(5*B+C)] Ground Truth 
    S                 -- Number of grid cell per dimension (total number of grid cells = S*S)
    B                 -- Number of Boxes per cell grid
    C                 -- Number of Classes
    LAMBDA_COORD      -- Parameter to apply to the loss computed on the position and size of the boxe
    LAMBDA_NOOBJ      -- Parameter to apply to the loss computed on the confidence of the wrongly created boxes
    
    Returns:
    total_loss        -- [BATCH_SIZE,]
    
    ''' 
    # Make sure the inputs are cast to the proper type
    predictions = tf.cast(predictions, dtype=tf.float64)
    ground_truth = tf.cast(ground_truth, dtype=tf.float64)
    
    # === EXTRACT INFO ===
    # --- Ground Truth ---
    true_reshape_ = tf.reshape(ground_truth, [-1, S, S, 5*B+C])                 # -> [BATCH_SIZE, S, S, 5*B + C]
    
    # Extract the one-hot vector of the classes
    true_class_proba = tf.reshape(true_reshape_[:, :, :, -C :], [-1, S*S, C])     # -> [BATCH_SIZE, S*S, C]
    
    # Get the 1st box (biggest) from the ground_truth (=> YOLOv1)
    true_boxes_info = tf.reshape(true_reshape_[:, :, :, : 5], [-1, S*S, 1, 5])    # -> [BATCH_SIZE, S*S, 1, 5]
    
    # Extract the information in different tensors
    true_center_coord = tf.tile(true_boxes_info[:, :, :, : 2], [1, 1, B, 1])      # -> [BATCH_SIZE, S*S, B, 2]
    true_box_size = tf.tile(true_boxes_info[:, :, :, 2: 4], [1, 1, B, 1])         # -> [BATCH_SIZE, S*S, B, 2]
    true_confidence = tf.tile(tf.reshape(true_boxes_info[:, :, :, 4], 
                                         [-1, S*S, 1, 1]), [1, 1, B, 1])          # -> [BATCH_SIZE, S*S, B, 1]
    
    batch_size = int(true_reshape_.shape[0])
    
    # ___DEBUG___
    #print('batch_size: ' + str(batch_size))
    #print_tensor('true_reshape_', true_reshape_)
    #print_tensor('true_class_proba', true_class_proba)
    #print_tensor('true_boxes_info', true_boxes_info)
    #print_tensor('true_center_coord', true_center_coord)
    #print_tensor('true_box_size', true_box_size)
    #print_tensor('true_confidence', true_confidence)
    #print_tensor('true_confidence', tf.reshape(true_confidence, [-1, S*S, B,]))
    # ___DEBUG___
    
    
    # --- Predictions ---
    pred_reshape_ = tf.reshape(predictions, [-1, S, S, 5*B+C])                # -> [BATCH_SIZE, S, S, 5*B + C]
    
    # Extract the class probabilities
    pred_class_proba = tf.reshape(pred_reshape_[:, :, :, -C :], [-1, S*S, C])   # -> [BATCH_SIZE, S*S, C]
    
    # Get the boxes info
    pred_boxes_info = tf.reshape(pred_reshape_[:, :, :, : -C], [-1, S*S, B, 5]) # -> [BATCH_SIZE, S*S, B, 5]
    
    # Extract the information in different tensors
    pred_center_coord = pred_boxes_info[:, :, :, : 2]                           # -> [BATCH_SIZE, S*S, B, 2]
    pred_box_size = pred_boxes_info[:, :, :, 2: 4]                              # -> [BATCH_SIZE, S*S, B, 2]
    pred_confidence = tf.reshape(pred_boxes_info[:, :, :, 4], [-1, S*S, B, 1])  # -> [BATCH_SIZE, S*S, B, 1]
    
    # ___DEBUG___
    #print_tensor('pred_reshape_', pred_reshape_)
    #print_tensor('pred_class_proba', pred_class_proba)
    #print_tensor('pred_boxes_info', pred_boxes_info)
    #print_tensor('pred_center_coord', pred_center_coord)
    #print_tensor('pred_box_size', pred_box_size)
    #print_tensor('pred_confidence', pred_confidence)
    #print_tensor('pred_confidence', tf.reshape(pred_confidence, [-1, S*S, B,]))
    # ___DEBUG___
    
    
    # === BOX LOSS === 
    # Zeroes out the predictions in empty cells
    pred_box_size = tf.multiply(pred_box_size, true_confidence)
    pred_center_coord = tf.multiply(pred_center_coord, true_confidence)
    
    # -- Compute IoU ---
    # N.B.1: The coordinates (x, y) of the center of the boxes have been normalized to the grid cell there are in.
    # N.B.2: The width and height of the boxes have been normalized to the full size of the image.
    # |-> In order to correct this when computing the Intersection Over Union we are going to multiply the 
    #     width and height by S has the image is just S times bigger than a grid cell.
    
    width_intersections = tf.minimum(pred_center_coord[:, :, :, 0] + pred_box_size[:, :, :, 0] * S/2,   \
                                     true_center_coord[:, :, :, 0] + true_box_size[:, :, :, 0] * S/2) - \
                          tf.maximum(pred_center_coord[:, :, :, 0] - pred_box_size[:, :, :, 0] * S/2,   \
                                     true_center_coord[:, :, :, 0] - true_box_size[:, :, :, 0] * S/2)
    
    height_intersections = tf.minimum(pred_center_coord[:, :, :, 1] + pred_box_size[:, :, :, 1] * S/2,   \
                                      true_center_coord[:, :, :, 1] + true_box_size[:, :, :, 1] * S/2) - \
                           tf.maximum(pred_center_coord[:, :, :, 1] - pred_box_size[:, :, :, 1] * S/2,   \
                                      true_center_coord[:, :, :, 1] - true_box_size[:, :, :, 1] * S/2)   
    
    # If the width or height is <0, it means their intersection is empty
    width_intersections = tf.maximum(width_intersections, 0)
    height_intersections = tf.maximum(height_intersections, 0)
    
    # Compute the are of the intersections
    intersections = tf.multiply(width_intersections, height_intersections)
    
    # Compute the area of the unions
    unions = tf.subtract(tf.multiply(pred_box_size[:, :, :, 0] * S, pred_box_size[:, :, :, 1] * S) +  \
                          tf.multiply(true_box_size[:, :, :, 0] * S, true_box_size[:, :, :, 1] * S), intersections)
    
    # Compute the Intersection Over Union (IoU)
    iou = tf.divide(intersections, unions)
    
    # Replace the Nan by 0
    # iou = tf.where(tf.is_nan(iou), tf.ones_like(iou) * 0, iou); #if iou is nan use 0 else use element in iou
    
    # Mask indicating where is the biggest IoU in each cell grid
    mask_bigger_iou = tf.cast(tf.greater_equal(iou, tf.tile(tf.reduce_max(iou, axis=2, keep_dims=True), [1, 1, B])), dtype=tf.float64)
    
    # ___DEBUG___
    #print_tensor('intersections', intersections)
    #print_tensor('unions', unions)
    #print_tensor('iou', iou)
    #print_tensor('mask_bigger_iou', mask_bigger_iou)
    # ___DEBUG___
    
    # --- Box Center ---    
    center_coord_loss = tf.reduce_sum(tf.square(pred_center_coord - true_center_coord), 3)

    # Only select the box which is predicting the object (biggest IoU)
    center_coord_loss = mask_bigger_iou * center_coord_loss

    # --- Box Size ---    
    sqrt_true_box_size = tf.sqrt(true_box_size)
    sqrt_pred_box_size = tf.sqrt(pred_box_size)
    
    box_size_loss = tf.reduce_sum(tf.square(sqrt_pred_box_size - sqrt_true_box_size), 3)

    # Only select the box which is predicting the object (biggest IoU)
    box_size_loss = mask_bigger_iou * box_size_loss
    
    # --- Box Loss ---
    box_loss = LAMBDA_COORD * (tf.reduce_sum(center_coord_loss, [1, 2]) + tf.reduce_sum(box_size_loss, [1, 2]))
    
    # ___DEBUG___
    #print_tensor('center_coord_loss', center_coord_loss)
    #print_tensor('box_size_loss', box_size_loss)
    #print_tensor('box_loss', box_loss)
    # ___DEBUG___

    
    # === OBJECT LOSS ===
    # Zero out the confidence for the boxes that are not responsible for the object in this cell
    true_confidence = mask_bigger_iou * tf.reshape(true_confidence, [-1, S*S, B])
    
    squared_confidence_pred_minus_true = tf.square(tf.reshape(pred_confidence, [-1, S*S, B]) - true_confidence)
    
    object_loss = tf.reduce_sum(mask_bigger_iou * squared_confidence_pred_minus_true, [1, 2])
    
    # ___DEBUG___
    #print_tensor('pred_confidence', pred_confidence)
    #print_tensor('object_loss', object_loss)
    # ___DEBUG___
    
    
    # === NO OBJECT LOSS ===
    # Inverse the IoU mask
    inverse_mask_bigger_iou = tf.square(mask_bigger_iou - 1)
    
    no_object_loss = LAMBDA_NOOBJ * tf.reduce_sum(inverse_mask_bigger_iou * squared_confidence_pred_minus_true, [1, 2])
    
    # ___DEBUG___
    #print_tensor('squared_confidence_pred_minus_true', squared_confidence_pred_minus_true)
    #print_tensor('inverse_mask_bigger_iou', inverse_mask_bigger_iou)
    #print_tensor('no_object_loss', no_object_loss)
    # ___DEBUG___
    
    
    # === CLASS LOSS ===
    # Use the fact that confidence is 1/0 if there is an object in a cell
    mask_cell_has_object = tf.reshape(true_boxes_info[:, :, :, 4], [-1, S*S, 1])
    
    class_loss = tf.reduce_sum(mask_cell_has_object * tf.square(pred_class_proba - true_class_proba), [1, 2])
    
     # ___DEBUG___
    #print_tensor('mask_cell_has_object', mask_cell_has_object)
    #print_tensor('class_loss', class_loss)
    # ___DEBUG___
    
    # === TOTAL LOSS ===
    total_loss = box_loss + object_loss + no_object_loss + class_loss
    
    # ___DEBUG___
    #print_tensor('total_loss', total_loss)
    # ___DEBUG___
    
    return total_loss