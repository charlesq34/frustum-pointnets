import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from roi_seg_box3d_dataset import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, compute_box3d_iou, class2type, type_mean_size
mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3))
for i in range(NUM_SIZE_CLUSTER):
    mean_size_arr[i,:] = type_mean_size[class2type[i]]

def huber_loss(error, delta):
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic**2 + delta * linear
    return tf.reduce_mean(losses)


def get_box3d_corners_helper(centers, headings, sizes):
    """ TF layer. Input: (N,3), (N,), (N,3), Output: (N,8,3) """
    print '-----', centers
    N = centers.get_shape()[0].value
    l = tf.slice(sizes, [0,0], [-1,1]) # (N,1)
    w = tf.slice(sizes, [0,1], [-1,1]) # (N,1)
    h = tf.slice(sizes, [0,2], [-1,1]) # (N,1)
    print l,w,h
    x_corners = tf.concat([l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2], axis=1) # (N,8)
    y_corners = tf.concat([h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2], axis=1) # (N,8)
    z_corners = tf.concat([w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2], axis=1) # (N,8)
    corners = tf.concat([tf.expand_dims(x_corners,1), tf.expand_dims(y_corners,1), tf.expand_dims(z_corners,1)], axis=1) # (N,3,8)
    print x_corners, y_corners, z_corners
    c = tf.cos(headings)
    s = tf.sin(headings)
    ones = tf.ones([N], dtype=tf.float32)
    zeros = tf.zeros([N], dtype=tf.float32)
    row1 = tf.stack([c,zeros,s], axis=1) # (N,3)
    row2 = tf.stack([zeros,ones,zeros], axis=1)
    row3 = tf.stack([-s,zeros,c], axis=1)
    R = tf.concat([tf.expand_dims(row1,1), tf.expand_dims(row2,1), tf.expand_dims(row3,1)], axis=1) # (N,3,3)
    print row1, row2, row3, R, N
    corners_3d = tf.matmul(R, corners) # (N,3,8)
    corners_3d += tf.tile(tf.expand_dims(centers,2), [1,1,8]) # (N,3,8)
    corners_3d = tf.transpose(corners_3d, perm=[0,2,1]) # (N,8,3)
    return corners_3d

def get_box3d_corners(center, heading_residuals, size_residuals):
    """ TF layer.
    Inputs:
        center: (B,3)
        heading_residuals: (B,NH)
        size_residuals: (B,NS,3)
    Outputs:
        box3d_corners: (B,NH,NS,8,3) tensor
    """
    batch_size = center.get_shape()[0].value
    heading_bin_centers = tf.constant(np.arange(0,2*np.pi,2*np.pi/NUM_HEADING_BIN), dtype=tf.float32) # (NH,)
    headings = heading_residuals + tf.expand_dims(heading_bin_centers, 0) # (B,NH)
    
    mean_sizes = tf.expand_dims(tf.constant(mean_size_arr, dtype=tf.float32), 0) + size_residuals # (B,NS,1)
    sizes = mean_sizes + size_residuals # (B,NS,3)
    sizes = tf.tile(tf.expand_dims(sizes,1), [1,NUM_HEADING_BIN,1,1]) # (B,NH,NS,3)
    headings = tf.tile(tf.expand_dims(headings,-1), [1,1,NUM_SIZE_CLUSTER]) # (B,NH,NS)
    centers = tf.tile(tf.expand_dims(tf.expand_dims(center,1),1), [1,NUM_HEADING_BIN, NUM_SIZE_CLUSTER,1]) # (B,NH,NS,3)

    N = batch_size*NUM_HEADING_BIN*NUM_SIZE_CLUSTER
    corners_3d = get_box3d_corners_helper(tf.reshape(centers, [N,3]), tf.reshape(headings, [N]), tf.reshape(sizes, [N,3]))

    return tf.reshape(corners_3d, [batch_size, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 8, 3])
