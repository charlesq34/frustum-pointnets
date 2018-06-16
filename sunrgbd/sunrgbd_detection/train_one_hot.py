''' Training Frustum PointNets on SUN-RGBD dataset.

Author: Charles R. Qi
Date: October 2017
'''

import argparse
import math
from datetime import datetime
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR) # model
import roi_seg_box3d_dataset
from train_util import get_batch

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model', help='Model name [default: model]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--no_rgb', action='store_true', help='Only use XYZ for training')
parser.add_argument('--restore_model_path', default=None, help='Restore model path e.g. log/model.ckpt [default: None]')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
NUM_CHANNEL = 3 if FLAGS.no_rgb else 6

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

NUM_CLASSES = 2

TRAIN_DATASET = roi_seg_box3d_dataset.ROISegBoxDataset(npoints=NUM_POINT, split='train', rotate_to_center=True, random_flip=True, random_shift=True, overwritten_data_path='train_1002_aug5x.zip.pickle', one_hot=True)
TEST_DATASET = roi_seg_box3d_dataset.ROISegBoxDataset(npoints=NUM_POINT, split='val', rotate_to_center=True, overwritten_data_path='val_1002.zip.pickle', one_hot=True)
print(len(TRAIN_DATASET))
print(len(TEST_DATASET))

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, heading_class_label_pl, heading_residual_label_pl, size_class_label_pl, size_residual_label_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get model and loss")
            # Get model and loss 
            logits, end_points = MODEL.get_model(pointclouds_pl, one_hot_vec_pl, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(logits, labels_pl, centers_pl, heading_class_label_pl, heading_residual_label_pl, size_class_label_pl, size_residual_label_pl, end_points)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(logits, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        if FLAGS.restore_model_path is None:
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            saver.restore(sess, FLAGS.restore_model_path)

        ops = {'pointclouds_pl': pointclouds_pl,
               'one_hot_vec_pl': one_hot_vec_pl,
               'labels_pl': labels_pl,
               'centers_pl': centers_pl,
               'heading_class_label_pl': heading_class_label_pl,
               'heading_residual_label_pl': heading_residual_label_pl,
               'size_class_label_pl': size_class_label_pl,
               'size_residual_label_pl': size_residual_label_pl,
               'is_training_pl': is_training_pl,
               'logits': logits,
               'centers_pred': end_points['center'],
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        best_loss = 1e10
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            epoch_loss = eval_one_epoch(sess, ops, test_writer)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_%03d.ckpt"%(epoch)))
                log_string("Model saved in file: %s" % save_path)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET)/BATCH_SIZE
    
    log_string(str(datetime.now()))

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    iou2ds_sum = 0
    iou3ds_sum = 0
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_data, batch_label, batch_center, batch_hclass, batch_hres, batch_sclass, batch_sres, batch_rot_angle, batch_one_hot_vec = get_batch(TRAIN_DATASET, train_idxs, start_idx, end_idx, NUM_POINT, NUM_CHANNEL)
        #print 'Ground truth centers: ', batch_center
        # Augment batched point clouds by rotation and jittering
        aug_data = batch_data
        #aug_data = provider.random_scale_point_cloud(batch_data)
        #aug_data = provider.jitter_point_cloud(aug_data)
        feed_dict = {ops['pointclouds_pl']: aug_data,
                     ops['one_hot_vec_pl']: batch_one_hot_vec,
                     ops['labels_pl']: batch_label,
                     ops['centers_pl']: batch_center,
                     ops['heading_class_label_pl']: batch_hclass,
                     ops['heading_residual_label_pl']: batch_hres,
                     ops['size_class_label_pl']: batch_sclass,
                     ops['size_residual_label_pl']: batch_sres,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, logits_val, centers_pred_val, iou2ds, iou3ds = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['logits'], ops['centers_pred'],
            ops['end_points']['iou2ds'], ops['end_points']['iou3ds']], feed_dict=feed_dict)
        #print 'Predicted centers: ', centers_pred_val
        train_writer.add_summary(summary, step)
        preds_val = np.argmax(logits_val, 2)
        correct = np.sum(preds_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
        iou2ds_sum += np.sum(iou2ds)
        iou3ds_sum += np.sum(iou3ds)

        if (batch_idx+1)%10 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
            log_string('mean loss: %f' % (loss_sum / 10))
            log_string('accuracy: %f' % (total_correct / float(total_seen)))
            log_string('Box IoU (ground/3D): %f / %f' % (iou2ds_sum / float(BATCH_SIZE*10), iou3ds_sum / float(BATCH_SIZE*10)))
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            iou2ds_sum = 0
            iou3ds_sum = 0
        
        #if batch_idx == 200:
        #    break
        

        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET))
    num_batches = len(TEST_DATASET)/BATCH_SIZE

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    shape_ious = []
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    iou2ds_sum = 0
    iou3ds_sum = 0
    
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_data, batch_label, batch_center, batch_hclass, batch_hres, batch_sclass, batch_sres, batch_rot_angle, batch_one_hot_vec = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx, NUM_POINT, NUM_CHANNEL)

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['one_hot_vec_pl']: batch_one_hot_vec,
                     ops['labels_pl']: batch_label,
                     ops['centers_pl']: batch_center,
                     ops['heading_class_label_pl']: batch_hclass,
                     ops['heading_residual_label_pl']: batch_hres,
                     ops['size_class_label_pl']: batch_sclass,
                     ops['size_residual_label_pl']: batch_sres,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, logits_val, iou2ds, iou3ds = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['logits'], 
            ops['end_points']['iou2ds'], ops['end_points']['iou3ds']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        preds_val = np.argmax(logits_val, 2)
        correct = np.sum(preds_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum(batch_label==l)
            total_correct_class[l] += (np.sum((preds_val==l) & (batch_label==l)))
        iou2ds_sum += np.sum(iou2ds)
        iou3ds_sum += np.sum(iou3ds)

        for i in range(BATCH_SIZE):
            segp = preds_val[i,:]
            segl = batch_label[i,:] 
            part_ious = [0.0 for _ in range(NUM_CLASSES)]
            for l in range(NUM_CLASSES):
                if (np.sum(segl==l) == 0) and (np.sum(segp==l) == 0): # part is not present, no logitsiction as well
                    part_ious[l] = 1.0
                else:
                    part_ious[l] = np.sum((segl==l) & (segp==l)) / float(np.sum((segl==l) | (segp==l)))
            shape_ious.append(np.mean(part_ious))
        
        #if batch_idx == 100:
        #    break

    print(len(shape_ious))
    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    log_string('eval mIoU: %f' % (np.mean(shape_ious)))
    log_string('eval box IoU (ground/3D): %f / %f' % (iou2ds_sum / float(num_batches*BATCH_SIZE), iou3ds_sum / float(num_batches*BATCH_SIZE)))
         
    EPOCH_CNT += 1
    return loss_sum/float(num_batches)


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
