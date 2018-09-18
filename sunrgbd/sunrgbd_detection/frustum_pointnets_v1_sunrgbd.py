''' Compared with model_v1 use deeper network for 3D box regression. use BN for all FC layers
compared with model_v1_deeper, added more layers for 3D regression. added dropout for segmentation
compared with model_v1_deeper_0913, added visu for mean iou3d, added balanced loss.
compared with model_v1_deeper_0914, use two stage center regression..
'''
import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../../models'))
import tf_util
from roi_seg_box3d_dataset import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_CLASS, compute_box3d_iou, class2type, type_mean_size
from model_util_sunrgbd import get_box3d_corners, get_box3d_corners_helper
mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3))
for i in range(NUM_SIZE_CLUSTER):
    mean_size_arr[i,:] = type_mean_size[class2type[i]]

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32,
                                     shape=(batch_size, num_point, 6))
    one_hot_vec_pl = tf.placeholder(tf.float32, shape=(batch_size, NUM_CLASS))
    labels_pl = tf.placeholder(tf.int32,
                                shape=(batch_size, num_point))
    centers_pl = tf.placeholder(tf.float32,
                                shape=(batch_size, 3))
    heading_class_label_pl = tf.placeholder(tf.int32, shape=(batch_size,))
    heading_residual_label_pl = tf.placeholder(tf.float32, shape=(batch_size,))
    size_class_label_pl = tf.placeholder(tf.int32, shape=(batch_size,))
    size_residual_label_pl = tf.placeholder(tf.float32, shape=(batch_size,3))
    return pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, heading_class_label_pl, heading_residual_label_pl, size_class_label_pl, size_residual_label_pl


def get_model(point_cloud, one_hot_vec, is_training, bn_decay=None):
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    input_image = tf.expand_dims(point_cloud, -1)

    net = tf_util.conv2d(input_image, 64, [1,6],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    point_feat = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(point_feat, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    global_feat = tf_util.max_pool2d(net, [num_point,1],
                                     padding='VALID', scope='maxpool')
    print global_feat

    global_feat = tf.concat([global_feat, tf.expand_dims(tf.expand_dims(one_hot_vec, 1), 1)], axis=3)
    print 'Global Feat: ', global_feat
    global_feat_expand = tf.tile(global_feat, [1, num_point, 1, 1])
    print point_feat, global_feat_expand
    concat_feat = tf.concat(axis=3, values=[point_feat, global_feat_expand])
    print concat_feat

    net = tf_util.conv2d(concat_feat, 512, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv6', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv7', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv8', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv9', bn_decay=bn_decay)
    net = tf_util.dropout(net, is_training, 'dp1', keep_prob=0.5)

    logits = tf_util.conv2d(net, 2, [1,1],
                         padding='VALID', stride=[1,1], activation_fn=None,
                         scope='conv10')
    logits = tf.squeeze(logits, [2]) # BxNxC
    print logits
    
    print '-----------'
    #net = tf.concat(axis=3, values=[net, tf.expand_dims(tf.slice(point_cloud, [0,0,0], [-1,-1,3]), 2)])
    mask = tf.slice(logits,[0,0,0],[-1,-1,1]) < tf.slice(logits,[0,0,1],[-1,-1,1])
    mask = tf.to_float(mask) # BxNx1
    mask_count = tf.tile(tf.reduce_sum(mask,axis=1,keep_dims=True), [1,1,3]) # Bx1x3
    print mask
    point_cloud_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3]) # BxNx3

    # ---- Subtract points mean ----
    mask_xyz_mean = tf.reduce_sum(tf.tile(mask, [1,1,3])*point_cloud_xyz, axis=1, keep_dims=True) # Bx1x3
    mask_xyz_mean = mask_xyz_mean/tf.maximum(mask_count,1) # Bx1x3
    point_cloud_xyz_stage1 = point_cloud_xyz - tf.tile(mask_xyz_mean, [1,num_point,1])
    print 'Point cloud xyz stage1: ', point_cloud_xyz_stage1

    # ---- Regress 1st stage center ----
    net = tf.expand_dims(point_cloud_xyz_stage1, 2)
    print net
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg1-stage1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg2-stage1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg3-stage1', bn_decay=bn_decay)
    mask_expand = tf.tile(tf.expand_dims(mask,-1), [1,1,1,256])
    masked_net = net*mask_expand
    print masked_net
    net = tf_util.max_pool2d(masked_net, [num_point,1], padding='VALID', scope='maxpool-stage1')
    net = tf.squeeze(net, axis=[1,2])
    print net
    net = tf.concat([net, one_hot_vec], axis=1)
    net = tf_util.fully_connected(net, 256, scope='fc1-stage1', bn=True, is_training=is_training, bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 128, scope='fc2-stage1', bn=True, is_training=is_training, bn_decay=bn_decay)
    stage1_center = tf_util.fully_connected(net, 3, activation_fn=None, scope='fc3-stage1')
    stage1_center = stage1_center + tf.squeeze(mask_xyz_mean, axis=1) # Bx3
    end_points['stage1_center'] = stage1_center

    # ---- Subtract stage1 center ----
    point_cloud_xyz_submean = point_cloud_xyz - tf.expand_dims(stage1_center, 1)
    print 'Point cloud xyz submean: ', point_cloud_xyz_submean

    net = tf.expand_dims(point_cloud_xyz_submean, 2)
    print net
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 512, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg4', bn_decay=bn_decay)
    mask_expand = tf.tile(tf.expand_dims(mask,-1), [1,1,1,512])
    masked_net = net*mask_expand
    print masked_net
    net = tf_util.max_pool2d(masked_net, [num_point,1], padding='VALID', scope='maxpool2')
    net = tf.squeeze(net, axis=[1,2])
    print net
    net = tf.concat([net, one_hot_vec], axis=1)
    net = tf_util.fully_connected(net, 512, scope='fc1', bn=True, is_training=is_training, bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, scope='fc2', bn=True, is_training=is_training, bn_decay=bn_decay)

    # First 3 are cx,cy,cz, next NUM_HEADING_BIN*2 are for heading
    # next NUM_SIZE_CLUSTER*4 are for dimension
    output = tf_util.fully_connected(net, 3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4, activation_fn=None, scope='fc3')
    print output

    center = tf.slice(output, [0,0], [-1,3])
    center = center + stage1_center # Bx3
    end_points['center'] = center

    heading_scores = tf.slice(output, [0,3], [-1,NUM_HEADING_BIN])
    heading_residuals_normalized = tf.slice(output, [0,3+NUM_HEADING_BIN], [-1,NUM_HEADING_BIN])
    end_points['heading_scores'] = heading_scores # BxNUM_HEADING_BIN
    end_points['heading_residuals_normalized'] = heading_residuals_normalized # BxNUM_HEADING_BIN (should be -1 to 1)
    end_points['heading_residuals'] = heading_residuals_normalized * (np.pi/NUM_HEADING_BIN) # BxNUM_HEADING_BIN
    
    size_scores = tf.slice(output, [0,3+NUM_HEADING_BIN*2], [-1,NUM_SIZE_CLUSTER]) # BxNUM_SIZE_CLUSTER
    size_residuals_normalized = tf.slice(output, [0,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER], [-1,NUM_SIZE_CLUSTER*3])
    size_residuals_normalized = tf.reshape(size_residuals_normalized, [batch_size, NUM_SIZE_CLUSTER, 3]) # BxNUM_SIZE_CLUSTERx3
    end_points['size_scores'] = size_scores
    end_points['size_residuals_normalized'] = size_residuals_normalized
    end_points['size_residuals'] = size_residuals_normalized * tf.expand_dims(tf.constant(mean_size_arr, dtype=tf.float32), 0)

    return logits, end_points


def huber_loss(error, delta):
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic**2 + delta * linear
    return tf.reduce_mean(losses)

# TODO: Test correctness of the loss...
def get_loss(logits, \
             mask_label, center_label, \
             heading_class_label, heading_residual_label, \
             size_class_label, size_residual_label, \
             end_points, reg_weight=0.001):
    """ logits: BxNxC,
        mask_label: BxN, """
    batch_size = logits.get_shape()[0].value
    mask_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=mask_label))
    tf.summary.scalar('3d mask loss', mask_loss)

    center_dist = tf.norm(center_label - end_points['center'], axis=-1)
    center_loss = huber_loss(center_dist, delta=2.0)
    tf.summary.scalar('center loss', center_loss)

    stage1_center_dist = tf.norm(center_label - end_points['stage1_center'], axis=-1)
    stage1_center_loss = huber_loss(stage1_center_dist, delta=1.0)
    tf.summary.scalar('stage1 center loss', stage1_center_loss)

    heading_class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=end_points['heading_scores'], labels=heading_class_label))
    tf.summary.scalar('heading class loss', heading_class_loss)

    tmp = tf.one_hot(heading_class_label, depth=NUM_HEADING_BIN, on_value=1, off_value=0, axis=-1) # BxNUM_HEADING_BIN
    print tmp
    heading_residual_normalized_label = heading_residual_label / (np.pi/NUM_HEADING_BIN)
    heading_residual_normalized_loss = huber_loss(tf.reduce_sum(end_points['heading_residuals_normalized']*tf.to_float(tmp), axis=1) - heading_residual_normalized_label, delta=1.0)
    print heading_residual_normalized_loss
    tf.summary.scalar('heading residual normalized loss', heading_residual_normalized_loss)

    size_class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=end_points['size_scores'], labels=size_class_label))
    tf.summary.scalar('size class loss', size_class_loss)

    tmp2 = tf.one_hot(size_class_label, depth=NUM_SIZE_CLUSTER, on_value=1, off_value=0, axis=-1) # BxNUM_SIZE_CLUSTER
    tmp2_tiled = tf.tile(tf.expand_dims(tf.to_float(tmp2), -1), [1,1,3]) # BxNUM_SIZE_CLUSTERx3
    predicted_size_residual_normalized = tf.reduce_sum(end_points['size_residuals_normalized']*tmp2_tiled, axis=[1]) # Bx3

    tmp3 = tf.expand_dims(tf.constant(mean_size_arr, dtype=tf.float32),0) # 1xNUM_SIZE_CLUSTERx3
    mean_size_label = tf.reduce_sum(tmp2_tiled * tmp3, axis=[1]) # Bx3
    size_residual_label_normalized = size_residual_label / mean_size_label
 
    size_normalized_dist = tf.norm(size_residual_label_normalized - predicted_size_residual_normalized, axis=-1)
    size_residual_normalized_loss = huber_loss(size_normalized_dist, delta=1.0)
    print size_residual_normalized_loss
    tf.summary.scalar('size residual normalized loss', size_residual_normalized_loss)

    # Compute IOU 3D
    iou2ds, iou3ds = tf.py_func(compute_box3d_iou, [end_points['center'], end_points['heading_scores'], end_points['heading_residuals'], end_points['size_scores'], end_points['size_residuals'], center_label, heading_class_label, heading_residual_label, size_class_label, size_residual_label], [tf.float32, tf.float32])
    tf.summary.scalar('iou_2d', tf.reduce_mean(iou2ds))
    tf.summary.scalar('iou_3d', tf.reduce_mean(iou3ds))
 
    end_points['iou2ds'] = iou2ds 
    end_points['iou3ds'] = iou3ds 

    # Compute BOX3D corners
    corners_3d = get_box3d_corners(end_points['center'], end_points['heading_residuals'], end_points['size_residuals']) # (B,NH,NS,8,3)
    gt_mask = tf.tile(tf.expand_dims(tmp, 2), [1,1,NUM_SIZE_CLUSTER]) * tf.tile(tf.expand_dims(tmp2,1), [1,NUM_HEADING_BIN,1]) # (B,NH,NS)
    corners_3d_pred = tf.reduce_sum(tf.to_float(tf.expand_dims(tf.expand_dims(gt_mask,-1),-1))*corners_3d, axis=[1,2]) # (B,8,3)

    heading_bin_centers = tf.constant(np.arange(0,2*np.pi,2*np.pi/NUM_HEADING_BIN), dtype=tf.float32) # (NH,)
    heading_label = tf.expand_dims(heading_residual_label,1) + tf.expand_dims(heading_bin_centers, 0) # (B,NH)
    heading_label = tf.reduce_sum(tf.to_float(tmp)*heading_label, 1)
    mean_sizes = tf.expand_dims(tf.constant(mean_size_arr, dtype=tf.float32), 0) # (1,NS,3)
    size_label = mean_sizes + tf.expand_dims(size_residual_label, 1) # (1,NS,3) + (B,1,3) = (B,NS,3)
    size_label = tf.reduce_sum(tf.expand_dims(tf.to_float(tmp2),-1)*size_label, axis=[1]) # (B,3)
    corners_3d_gt = get_box3d_corners_helper(center_label, heading_label, size_label) # (B,8,3)
    corners_3d_gt_flip = get_box3d_corners_helper(center_label, heading_label+np.pi, size_label) # (B,8,3)

    corners_dist = tf.minimum(tf.norm(corners_3d_pred - corners_3d_gt, axis=-1), tf.norm(corners_3d_pred - corners_3d_gt_flip, axis=-1))
    print "Corners dist: ", corners_dist
    corners_loss = huber_loss(corners_dist, delta=1.0) 
    tf.summary.scalar('corners loss', corners_loss)

    return mask_loss + (center_loss + heading_class_loss + size_class_loss + heading_residual_normalized_loss*20 + size_residual_normalized_loss*20 + stage1_center_loss)*0.1 + corners_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,6))
        outputs = get_model(inputs, tf.ones((32,3)), tf.constant(True))
        print outputs
        loss = get_loss(outputs[0], tf.zeros((32,1024),dtype=tf.int32), tf.zeros((32,3)), tf.zeros((32,),dtype=tf.int32), tf.zeros((32,)), tf.zeros((32,),dtype=tf.int32), tf.zeros((32,3)), outputs[1])
        print loss
