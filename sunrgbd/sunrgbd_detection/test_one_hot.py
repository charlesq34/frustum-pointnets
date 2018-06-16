''' Testing Frustum PointNets on SUN-RGBD dataset.

Author: Charles R. Qi
Date: October 2017
'''

import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, '../sunrgbd_data'))
from utils import save_zipped_pickle
import roi_seg_box3d_dataset
from roi_seg_box3d_dataset import NUM_CLASS, NUM_SIZE_CLUSTER, NUM_HEADING_BIN
import cPickle as pickle
from train_util import get_batch

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--model', default='model', help='Model name [default: model]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--output', default='test_results', help='output filename [default: test_results]')
parser.add_argument('--data_path', default=None, help='data path [default: None]')
parser.add_argument('--from_rgb_detection', action='store_true', help='test from data file from rgb detection.')
parser.add_argument('--idx_path', default=None, help='filename of txt where each line is a data idx, used for rgb detection -- write <id>.txt for all frames. [default: None]')
parser.add_argument('--dump_result', action='store_true', help='dump result to .pickle file')
FLAGS = parser.parse_args()


MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
NUM_POINT = FLAGS.num_point
MODEL = importlib.import_module(FLAGS.model) # import network module
NUM_CHANNEL = 6
TEST_DATASET = roi_seg_box3d_dataset.ROISegBoxDataset(npoints=NUM_POINT, split='val', rotate_to_center=True, overwritten_data_path=FLAGS.data_path, from_rgb_detection=FLAGS.from_rgb_detection, one_hot=True)

def get_model(batch_size, num_point):
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, heading_class_label_pl, heading_residual_label_pl, size_class_label_pl, size_residual_label_pl = MODEL.placeholder_inputs(batch_size, num_point)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            pred, end_points = MODEL.get_model(pointclouds_pl, one_hot_vec_pl, is_training_pl)
            loss = MODEL.get_loss(pred, labels_pl, centers_pl, heading_class_label_pl, heading_residual_label_pl, size_class_label_pl, size_residual_label_pl, end_points)
            saver = tf.train.Saver()
        #for v in tf.global_variables():
        #    print(v.name)
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        ops = {'pointclouds_pl': pointclouds_pl,
               'one_hot_vec_pl': one_hot_vec_pl,
               'labels_pl': labels_pl,
               'centers_pl': centers_pl,
               'heading_class_label_pl': heading_class_label_pl,
               'heading_residual_label_pl': heading_residual_label_pl,
               'size_class_label_pl': size_class_label_pl,
               'size_residual_label_pl': size_residual_label_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'center': end_points['center'],
               'end_points': end_points,
               'loss': loss}
        return sess, ops

def softmax(x):
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

def inference(sess, ops, pc, one_hot_vec, batch_size):
    ''' pc: BxNx3 array, Bx3 array, return BxN pred and Bx3 centers '''
    assert pc.shape[0]%batch_size == 0
    num_batches = pc.shape[0]/batch_size
    logits = np.zeros((pc.shape[0], pc.shape[1], 2))
    centers = np.zeros((pc.shape[0], 3))
    heading_logits = np.zeros((pc.shape[0],NUM_HEADING_BIN))
    heading_residuals = np.zeros((pc.shape[0],NUM_HEADING_BIN))
    size_logits = np.zeros((pc.shape[0],NUM_SIZE_CLUSTER))
    size_residuals = np.zeros((pc.shape[0],NUM_SIZE_CLUSTER,3))
    scores = np.zeros((pc.shape[0],)) # score that indicates confidence in 3d box prediction (mask logits+heading+size); no confidence for the center...
   
    ep = ops['end_points'] 
    for i in range(num_batches):
        feed_dict = {ops['pointclouds_pl']: pc[i*batch_size:(i+1)*batch_size,...],
                     ops['one_hot_vec_pl']: one_hot_vec[i*batch_size:(i+1)*batch_size,:],
                     ops['is_training_pl']: False}
        batch_logits, batch_centers, batch_heading_scores, batch_heading_residuals, batch_size_scores, batch_size_residuals = sess.run([ops['pred'], ops['center'], ep['heading_scores'], ep['heading_residuals'], ep['size_scores'], ep['size_residuals']], feed_dict=feed_dict)
        logits[i*batch_size:(i+1)*batch_size,...] = batch_logits
        centers[i*batch_size:(i+1)*batch_size,...] = batch_centers
        heading_logits[i*batch_size:(i+1)*batch_size,...] = batch_heading_scores
        heading_residuals[i*batch_size:(i+1)*batch_size,...] = batch_heading_residuals
        size_logits[i*batch_size:(i+1)*batch_size,...] = batch_size_scores
        size_residuals[i*batch_size:(i+1)*batch_size,...] = batch_size_residuals

        # Compute scores
        batch_seg_prob = softmax(batch_logits)[:,:,1] # BxN
        batch_seg_mask = np.argmax(batch_logits, 2) # BxN
        mask_mean_prob = np.sum(batch_seg_prob * batch_seg_mask, 1) # B,
        mask_mean_prob = mask_mean_prob / np.sum(batch_seg_mask,1) # B,
        heading_prob = np.max(softmax(batch_heading_scores),1) # B
        size_prob = np.max(softmax(batch_size_scores),1) # B,
        batch_scores = np.log(mask_mean_prob) + np.log(heading_prob) + np.log(size_prob)
        scores[i*batch_size:(i+1)*batch_size] = batch_scores 
        # Finished computing scores

    heading_cls = np.argmax(heading_logits, 1) # B
    size_cls = np.argmax(size_logits, 1) # B
    return np.argmax(logits, 2), centers, heading_cls, np.array([heading_residuals[i,heading_cls[i]] for i in range(pc.shape[0])]), size_cls, np.vstack([size_residuals[i,size_cls[i],:] for i in range(pc.shape[0])]), scores

def write_detection_results(result_dir, id_list, type_list, box2d_list, center_list, heading_cls_list, heading_res_list, size_cls_list, size_res_list, rot_angle_list, score_list):
    if result_dir is None: return
    results = {} # map from idx to list of strings, each string is a line (without \n) 
    for i in range(len(center_list)):
        idx = id_list[i]
        output_str = type_list[i] + " -1 -1 -10 "
        box2d = box2d_list[i]
        output_str += "%f %f %f %f " % (box2d[0],box2d[1],box2d[2],box2d[3])
        h,w,l,tx,ty,tz,ry = roi_seg_box3d_dataset.from_prediction_to_label_format(center_list[i], heading_cls_list[i], heading_res_list[i], size_cls_list[i], size_res_list[i], rot_angle_list[i])
        score = score_list[i]
        output_str += "%f %f %f %f %f %f %f %f" % (h,w,l,tx,ty,tz,ry,score)
        if idx not in results: results[idx] = []
        results[idx].append(output_str)
    # Write TXT files
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    output_dir = os.path.join(result_dir, 'data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    for idx in results:
        pred_filename = os.path.join(output_dir, '%06d.txt'%(idx))
        fout = open(pred_filename, 'w')
        for line in results[idx]:
            fout.write(line+'\n')
        fout.close() 

def fill_files(output_dir, to_fill_filename_list):
    for filename in to_fill_filename_list:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            fout = open(filepath, 'w')
            fout.close()

# TODO: support variable length input..
def main_batch_from_rgb_detection(output_filename, result_dir=None):
    ps_list = []
    segp_list = []
    center_list = []
    heading_cls_list = []
    heading_res_list = []
    size_cls_list = []
    size_res_list = []
    rot_angle_list = []
    score_list = []

    test_idxs = np.arange(0, len(TEST_DATASET))
    print(len(TEST_DATASET))
    raw_input()
    batch_size = 32
    num_batches = int((len(TEST_DATASET)+batch_size-1)/batch_size)
    
    batch_data_to_feed = np.zeros((batch_size, NUM_POINT, NUM_CHANNEL))
    batch_one_hot_to_feed = np.zeros((batch_size, NUM_CLASS))
    sess, ops = get_model(batch_size=batch_size, num_point=NUM_POINT)
    for batch_idx in range(num_batches):
        print(batch_idx)
        start_idx = batch_idx * batch_size
        end_idx = min(len(TEST_DATASET), (batch_idx+1) * batch_size)
        cur_batch_size = end_idx - start_idx

        batch_data, batch_rot_angle, batch_rgb_prob, batch_one_hot_vec = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx, NUM_POINT, NUM_CHANNEL, from_rgb_detection=True)
        batch_data_to_feed[0:cur_batch_size,...] = batch_data
        batch_one_hot_to_feed[0:cur_batch_size,:] = batch_one_hot_vec
	batch_output, batch_center_pred, batch_hclass_pred, batch_hres_pred, batch_sclass_pred, batch_sres_pred, batch_scores = inference(sess, ops, batch_data_to_feed, batch_one_hot_to_feed, batch_size=batch_size)
        print(batch_hclass_pred.shape, batch_hres_pred.shape)
        print(batch_sclass_pred.shape, batch_sres_pred.shape)
	
        for i in range(cur_batch_size):
            ps_list.append(batch_data[i,...])
            segp_list.append(batch_output[i,...])
            center_list.append(batch_center_pred[i,:])
            heading_cls_list.append(batch_hclass_pred[i])
            heading_res_list.append(batch_hres_pred[i])
            size_cls_list.append(batch_sclass_pred[i])
            size_res_list.append(batch_sres_pred[i,:])
            rot_angle_list.append(batch_rot_angle[i])
            #score_list.append(batch_scores[i] + np.log(batch_rgb_prob[i])) # Combine 3D BOX score and 2D RGB detection score
            score_list.append(batch_rgb_prob[i]) # 2D RGB detection score

    if FLAGS.dump_result:
        save_zipped_pickle([ps_list, segp_list, center_list, heading_cls_list, heading_res_list, size_cls_list, size_res_list, rot_angle_list, score_list], output_filename) 

    # Write detection results for KITTI evaluation
    print(len(ps_list))
    raw_input()
    write_detection_results(result_dir, TEST_DATASET.id_list, TEST_DATASET.type_list, TEST_DATASET.box2d_list, center_list, heading_cls_list, heading_res_list, size_cls_list, size_res_list, rot_angle_list, score_list)
    # Make sure for each frame (no matter if we have measurment for that frame), there is a TXT file
    output_dir = os.path.join(result_dir, 'data')
    to_fill_filename_list = [line.rstrip()+'.txt' for line in open(FLAGS.idx_path)]
    fill_files(output_dir, to_fill_filename_list)

# TODO: support variable length input..
def main_batch(output_filename, result_dir=None):
    ps_list = []
    seg_list = []
    segp_list = []
    center_list = []
    heading_cls_list = []
    heading_res_list = []
    size_cls_list = []
    size_res_list = []
    rot_angle_list = []
    score_list = []

    test_idxs = np.arange(0, len(TEST_DATASET))
    batch_size = 32
    num_batches = int((len(TEST_DATASET)+batch_size-1)/batch_size)

    batch_data_to_feed = np.zeros((batch_size, NUM_POINT, NUM_CHANNEL))
    batch_one_hot_to_feed = np.zeros((batch_size, NUM_CLASS))
    sess, ops = get_model(batch_size=batch_size, num_point=NUM_POINT)
    correct_cnt = 0
    for batch_idx in range(num_batches):
        #if batch_idx == 50: break #TODO: remove this line!!
        print(batch_idx)
        start_idx = batch_idx * batch_size
        end_idx = min(len(TEST_DATASET), (batch_idx+1) * batch_size)
        cur_batch_size = end_idx - start_idx

        batch_data, batch_label, batch_center, batch_hclass, batch_hres, batch_sclass, batch_sres, batch_rot_angle, batch_one_hot_vec = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx, NUM_POINT, NUM_CHANNEL)
        batch_data_to_feed[0:cur_batch_size,...] = batch_data
        batch_one_hot_to_feed[0:cur_batch_size,:] = batch_one_hot_vec
	batch_output, batch_center_pred, batch_hclass_pred, batch_hres_pred, batch_sclass_pred, batch_sres_pred, batch_scores = inference(sess, ops, batch_data_to_feed, batch_one_hot_to_feed, batch_size=batch_size)
        print(batch_hclass_pred.shape, batch_hres_pred.shape)
        print(batch_sclass_pred.shape, batch_sres_pred.shape)
        #raw_input()
        correct_cnt += np.sum(batch_output[0:cur_batch_size,...]==batch_label[0:cur_batch_size,...])
	
        for i in range(cur_batch_size):
            ps_list.append(batch_data[i,...])
            seg_list.append(batch_label[i,...])
            segp_list.append(batch_output[i,...])
            center_list.append(batch_center_pred[i,:])
            heading_cls_list.append(batch_hclass_pred[i])
            heading_res_list.append(batch_hres_pred[i])
            size_cls_list.append(batch_sclass_pred[i])
            size_res_list.append(batch_sres_pred[i,:])
            rot_angle_list.append(batch_rot_angle[i])
            score_list.append(batch_scores[i])
    print("Accuracy: ", correct_cnt / float(len(TEST_DATASET)*NUM_POINT))

    save_zipped_pickle([ps_list, segp_list, center_list, heading_cls_list, heading_res_list, size_cls_list, size_res_list, rot_angle_list, score_list], output_filename) 

    # Write detection results for KITTI evaluation
    write_detection_results(result_dir, TEST_DATASET.id_list, TEST_DATASET.type_list, TEST_DATASET.box2d_list, center_list, heading_cls_list, heading_res_list, size_cls_list, size_res_list, rot_angle_list, score_list)

if __name__=='__main__':
    if FLAGS.from_rgb_detection:
        main_batch_from_rgb_detection(FLAGS.output+'.pickle', FLAGS.output)
    else:
        main_batch(FLAGS.output+'.pickle', FLAGS.output)
