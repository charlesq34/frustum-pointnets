""" Compare MATLAB and Python eval code on AP computation """
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
from box_util import box3d_iou, is_clockwise
import roi_seg_box3d_dataset
from roi_seg_box3d_dataset import rotate_pc_along_y, NUM_HEADING_BIN
from eval_det import eval_det_cls

root_dir = '/home/rqi/Data/detection'
gt_boxes_dir = '/home/rqi/Projects/kitti-challenge/sunrgbd_detection/gt_boxes'

def flip_axis_to_camera(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
        Input and output are both (N,3) array
    '''
    pc2 = np.copy(pc)
    pc2[:,[0,1,2]] = pc2[:,[0,2,1]] # cam X,Y,Z = depth X,-Z,Y
    pc2[:,1] *= -1
    return pc2

def box_conversion(bbox):
    """ In upright depth camera coord """
    bbox3d = np.zeros((8,3))
    # Make clockwise
    # NOTE: in box3d IoU evaluation we require the polygon vertices in
    # counter clockwise order. However, from dumped data in MATLAB
    # some of the polygons are in clockwise, some others are counter clockwise
    # so we need to inspect each box and make them consistent..
    xy = np.reshape(bbox[0:8], (4,2))
    if is_clockwise(xy):    
        bbox3d[0:4,0:2] = xy
        bbox3d[4:,0:2] = xy
    else:
        bbox3d[0:4,0:2] = xy[::-1,:]
        bbox3d[4:,0:2] = xy[::-1,:]
    bbox3d[0:4,2] = bbox[9] # zmax
    bbox3d[4:,2] = bbox[8] # zmin
    return bbox3d

def wrapper(bbox):
    bbox3d = box_conversion(bbox)
    bbox3d = flip_axis_to_camera(bbox3d)
    bbox3d_flipped = np.copy(bbox3d)
    bbox3d_flipped[0:4,:] = bbox3d[4:,:]
    bbox3d_flipped[4:,:] = bbox3d[0:4,:]
    return bbox3d_flipped

def get_gt_cls(classname):
    gt = {}
    gt_boxes = np.loadtxt(os.path.join(gt_boxes_dir, '%s_gt_boxes.dat'%(classname)))
    gt_imgids = np.loadtxt(os.path.join(gt_boxes_dir, '%s_gt_imgids.txt'%(classname)))
    print gt_boxes.shape
    print gt_imgids.shape
    for i in range(len(gt_imgids)):
        imgid = gt_imgids[i]
        bbox = gt_boxes[i]
        bbox3d = wrapper(bbox)
    
        if imgid not in gt:
            gt[imgid] = []
        gt[imgid].append(bbox3d)
    return gt

if __name__=='__main__':
    #gt_boxes = np.loadtxt(os.path.join(gt_boxes_dir, 'chair_gt_boxes.dat'))
    #gt_imgids = np.loadtxt(os.path.join(gt_boxes_dir, 'chair_gt_imgids.txt'))
    pred_boxes = np.transpose(np.loadtxt(os.path.join(root_dir, 'chair_pred_boxes.dat')))
    pred_imgids = np.loadtxt(os.path.join(root_dir, 'chair_pred_imgids.txt'))
    pred_confidence = np.loadtxt(os.path.join(root_dir, 'chair_pred_confidence.txt'))
    
    pred = {}
    ovthresh = 0.25
    
    print pred_boxes.shape
     
    for i in range(0,10000):
        imgid = pred_imgids[i]
        score = pred_confidence[i]
        bbox = pred_boxes[i]
        bbox3d = wrapper(bbox)
    
        if imgid not in pred:
            pred[imgid] = []
        pred[imgid].append((bbox3d, score))
    
    gt = get_gt_cls('chair')
    
    # =================================================================================
    """
    import cPickle as pickle
    from PIL import Image
    import cv2
    import roi_seg_box3d_dataset
    sys.path.append('../sunrgbd_data')
    from sunrgbd_data import sunrgbd_object
    from utils import rotz, compute_box_3d, load_zipped_pickle
    
    IMG_DIR = '/home/rqi/Data/mysunrgbd/training/image'
    TEST_DATASET = roi_seg_box3d_dataset.ROISegBoxDataset(npoints=2048, split='val', rotate_to_center=True, overwritten_data_path='val_1002.zip.pickle', from_rgb_detection=False)
    dataset = sunrgbd_object('/home/rqi/Data/mysunrgbd', 'training')
    
    # For detection evaluation
    gt = {}
    
    # Get GT boxes
    print 'Construct GT boxes...'
    for i in range(len(TEST_DATASET)):
        img_id = TEST_DATASET.id_list[i]
        if img_id in gt: continue # All ready counted..
        gt[img_id] = []
    
        objects = dataset.get_label_objects(img_id)
        calib = dataset.get_calibration(img_id)
        for obj in objects:
            if obj.classname != 'chair': continue
            box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj, calib)
            box3d_pts_3d = calib.project_upright_depth_to_upright_camera(box3d_pts_3d)
            box3d_pts_3d_flipped = np.copy(box3d_pts_3d)
            box3d_pts_3d_flipped[0:4,:] = box3d_pts_3d[4:,:]
            box3d_pts_3d_flipped[4:,:] = box3d_pts_3d[0:4,:]
            gt[img_id].append(box3d_pts_3d_flipped)
    """
    # ====================================================================================
    
    
    import matplotlib.pyplot as plt
    rec, prec, ap = eval_det_cls(pred, gt, ovthresh)
    print prec[0:100]
    print rec[0:100]
    
    plt.plot(rec, prec, lw=2)
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 0.16])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()
    
    print ap
