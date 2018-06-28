import cPickle as pickle
import numpy as np
import argparse
from PIL import Image
import cv2
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../sunrgbd_data'))
from sunrgbd_data import sunrgbd_object
from utils import rotz, compute_box_3d, load_zipped_pickle
sys.path.append(os.path.join(BASE_DIR, '../../train'))
from box_util import box3d_iou
import roi_seg_box3d_dataset
from roi_seg_box3d_dataset import rotate_pc_along_y, NUM_HEADING_BIN
from eval_det import eval_det
from compare_matlab_and_python_eval import get_gt_cls

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default=None, help='data path for .pickle file, the one used for val in train.py [default: None]')
parser.add_argument('--result_path', default=None, help='result path for .pickle file from test.py [default: None]')
parser.add_argument('--from_rgb_detection', action='store_true', help='test from data file from rgb detection.')
FLAGS = parser.parse_args()


IMG_DIR = '/home/rqi/Data/mysunrgbd/training/image'
TEST_DATASET = roi_seg_box3d_dataset.ROISegBoxDataset(npoints=2048, split='val', rotate_to_center=True, overwritten_data_path=FLAGS.data_path, from_rgb_detection=FLAGS.from_rgb_detection)
dataset = sunrgbd_object('/home/rqi/Data/mysunrgbd', 'training')

ps_list, segp_list, center_list, heading_cls_list, heading_res_list, size_cls_list, size_res_list, rot_angle_list, score_list = load_zipped_pickle(FLAGS.result_path)

# For detection evaluation
pred_all = {}
gt_all = {}
ovthresh = 0.25

print len(segp_list), len(TEST_DATASET)
raw_input()

# Get GT boxes
print 'Construct GT boxes...'
classname_list = ['bed','table','sofa','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub']
"""
for i in range(len(TEST_DATASET)):
    img_id = TEST_DATASET.id_list[i]
    if img_id in gt_all: continue # All ready counted..
    gt_all[img_id] = []

    objects = dataset.get_label_objects(img_id)
    calib = dataset.get_calibration(img_id)
    for obj in objects:
        if obj.classname not in classname_list: continue
        box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj, calib)
        box3d_pts_3d = calib.project_upright_depth_to_upright_camera(box3d_pts_3d)
        box3d_pts_3d_flipped = np.copy(box3d_pts_3d)
        box3d_pts_3d_flipped[0:4,:] = box3d_pts_3d[4:,:]
        box3d_pts_3d_flipped[4:,:] = box3d_pts_3d[0:4,:]
        gt_all[img_id].append((obj.classname, box3d_pts_3d_flipped))
"""

#gt_all2 = {}
gt_cls = {}
for classname in classname_list:
    gt_cls[classname] = get_gt_cls(classname)
    for img_id in gt_cls[classname]:
        if img_id not in gt_all:
            gt_all[img_id] = []
        for box in gt_cls[classname][img_id]:
            gt_all[img_id].append((classname, box))
#print gt_all[1]
#print gt_all2[1]
raw_input()

# Get PRED boxes
print 'Construct PRED boxes...'
for i in range(len(TEST_DATASET)):
    img_id = TEST_DATASET.id_list[i] 
    classname = TEST_DATASET.type_list[i]

    center = center_list[i].squeeze()
    ret = TEST_DATASET[i]
    if FLAGS.from_rgb_detection:
        rot_angle = ret[1]
    else:
        rot_angle = ret[7]

    # Get heading angle and size
    #print heading_cls_list[i], heading_res_list[i], size_cls_list[i], size_res_list[i]
    heading_angle = roi_seg_box3d_dataset.class2angle(heading_cls_list[i], heading_res_list[i], NUM_HEADING_BIN)
    box_size = roi_seg_box3d_dataset.class2size(size_cls_list[i], size_res_list[i]) 
    corners_3d_pred = roi_seg_box3d_dataset.get_3d_box(box_size, heading_angle, center)
    corners_3d_pred = rotate_pc_along_y(corners_3d_pred, -rot_angle)

    if img_id not in pred_all:
        pred_all[img_id] = []
    pred_all[img_id].append((classname, corners_3d_pred, score_list[i]))
print pred_all[1]
raw_input()

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('axes', linewidth=2)
print 'Computing AP...'
rec, prec, ap = eval_det(pred_all, gt_all, ovthresh)
for classname in ap.keys():
    print '%015s: %f' % (classname, ap[classname])
    plt.plot(rec[classname], prec[classname], lw=3)
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=24)
    plt.ylabel('Precision', fontsize=24)
    plt.title(classname, fontsize=24)
    plt.show()
    raw_input()
print 'mean AP: ', np.mean([ap[classname] for classname in ap])
