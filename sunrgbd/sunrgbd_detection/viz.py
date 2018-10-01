''' Example usage:
    python viz.py --data_path roi_seg_box3d_caronly_val_0911.pickle --result_path test_results_caronly_aug5x.pickle

    End-to-end visualization of RGB detector and 3D segmentation and box regression.
'''
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
sys.path.append(os.path.join(BASE_DIR, '../../mayavi'))
from box_util import box3d_iou
import roi_seg_box3d_dataset
from roi_seg_box3d_dataset import rotate_pc_along_y, NUM_HEADING_BIN

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default=None, help='data path for .pickle file, the one used for val in train.py [default: None]')
parser.add_argument('--result_path', default=None, help='result path for .pickle file from test.py [default: None]')
parser.add_argument('--from_rgb_detection', action='store_true', help='test from data file from rgb detection.')
parser.add_argument('--viz', action='store_true', help='to visualize error result.')
FLAGS = parser.parse_args()


IMG_DIR = '/home/rqi/Data/mysunrgbd/training/image'
TEST_DATASET = roi_seg_box3d_dataset.ROISegBoxDataset(npoints=2048, split='val', rotate_to_center=True, overwritten_data_path=FLAGS.data_path, from_rgb_detection=FLAGS.from_rgb_detection)
dataset = sunrgbd_object('/home/rqi/Data/mysunrgbd', 'training')
VISU = FLAGS.viz
if VISU:
    import mayavi.mlab as mlab
    from viz_util import draw_lidar, draw_gt_boxes3d

#with open(FLAGS.result_path, 'rb') as fp:
#    ps_list = pickle.load(fp)
#    segp_list = pickle.load(fp)
#    center_list = pickle.load(fp)
#    heading_cls_list = pickle.load(fp)
#    heading_res_list = pickle.load(fp)
#    size_cls_list = pickle.load(fp)
#    size_res_list = pickle.load(fp)
#    rot_angle_list = pickle.load(fp)
#    score_list = pickle.load(fp)
ps_list, segp_list, center_list, heading_cls_list, heading_res_list, size_cls_list, size_res_list, rot_angle_list, score_list = load_zipped_pickle(FLAGS.result_path)

cnt = 0
correct_cnt = 0
box3d_map = {}
box2d_map = {}
pred_map = {}
type_map = {}
for i in range(len(segp_list)):
    print " ---- %d/%d"%(i,len(segp_list))
    if score_list[i]<0.5: continue
    img_id = TEST_DATASET.id_list[i] 
    box2d = TEST_DATASET.box2d_list[i]

    objects = dataset.get_label_objects(img_id)
    calib = dataset.get_calibration(img_id)
    if img_id not in box3d_map:
        box3d_map[img_id] = []
        box2d_map[img_id] = []
        type_map[img_id] = []
    for obj in objects:
        if obj.classname not in ['bed','table','sofa','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub']: continue
        box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj, calib)
        box3d_pts_3d = calib.project_upright_depth_to_upright_camera(box3d_pts_3d)
        box3d_map[img_id].append(box3d_pts_3d)
        box2d_map[img_id].append(obj.box2d)
        type_map[img_id].append(obj.classname)

    ps = ps_list[i]
    segp = segp_list[i].squeeze()
    center = center_list[i].squeeze()
    ret = TEST_DATASET[i]
    if FLAGS.from_rgb_detection:
        rot_angle = ret[1]
    else:
        rot_angle = ret[7]

    # Get heading angle and size
    print heading_cls_list[i], heading_res_list[i], size_cls_list[i], size_res_list[i]
    heading_angle = roi_seg_box3d_dataset.class2angle(heading_cls_list[i], heading_res_list[i], NUM_HEADING_BIN)
    box_size = roi_seg_box3d_dataset.class2size(size_cls_list[i], size_res_list[i]) 
    corners_3d_pred = roi_seg_box3d_dataset.get_3d_box(box_size, heading_angle, center)

    if img_id not in pred_map:
        pred_map[img_id] = []
    pred_map[img_id].append((box2d, segp, center, rot_angle, corners_3d_pred, roi_seg_box3d_dataset.class2type[size_cls_list[i]]))

shuffled_keys = pred_map.keys()
np.random.shuffle(shuffled_keys)
#for img_id in pred_map.keys():
#for img_id in [264,403]+shuffled_keys:
for img_id in shuffled_keys:
    calib = dataset.get_calibration(img_id)
    
    pred_list = pred_map[img_id]

    gt_box3d_list = box3d_map[img_id]
    gt_box2d_list = box2d_map[img_id]
    gt_cls_list = type_map[img_id]
    if 'sofa' not in gt_cls_list:
        continue
         
    img_filename = os.path.join(IMG_DIR, '%06d.jpg'%(img_id))
    img = cv2.imread(img_filename)
    print img_filename, img_id, img.shape
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    pc_upright_depth = dataset.get_depth(img_id) 
    pc_upright_camera = calib.project_upright_depth_to_upright_camera(pc_upright_depth[:,0:3])

    fig = mlab.figure(figure=None, bgcolor=(0.9,0.9,0.9), fgcolor=None, engine=None, size=(800, 800))
    mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.1, figure=fig)
    axes=np.array([
        [0.3,0.,0.,0.],
        [0.,0.3,0.,0.],
        [0.,0.,0.3,0.],
    ],dtype=np.float64)
    mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)


    mlab.points3d(pc_upright_camera[:,0], pc_upright_camera[:,1], pc_upright_camera[:,2], pc_upright_camera[:,2], mode='point', colormap='gnuplot', scale_factor=1, figure=fig)
    box3d_pred_list = []
    viz_gt_box3d_list = []
    color_list = []
    pred2gt_idx_map = [-1 for _ in range(len(pred_list))]
    for i,pred in enumerate(pred_list):
        box2d, segp, center, rot_angle, corners_3d_pred, classname = pred

        # -------- GREEDY WAY TO MAP PRED TO GT ------------
        predbox3d = rotate_pc_along_y(corners_3d_pred, -rot_angle)
        box3d_pred_list.append(predbox3d)
        iou3d_max = 0
        iou3d_idx = -1
        shift_arr = np.array([4,5,6,7,0,1,2,3])
        for j,gtbox3d in enumerate(gt_box3d_list):
            if j in pred2gt_idx_map: continue
            iou3d, _ = box3d_iou(gtbox3d[shift_arr,:], predbox3d)
            if iou3d>iou3d_max:
                iou3d_max = iou3d
                iou3d_idx = j
        if iou3d_idx>=0:
            pred2gt_idx_map[i] = j
        # ---------- END ----------------
                
        print i,iou3d_idx,iou3d_max
        #raw_input()
        if iou3d_max>=0.25:
            color_list.append((0,1,0))
        else:
            color_list.append((1,0,0))
            viz_gt_box3d_list.append(gt_box3d_list[i])
        if img_id==403:
            if i>=2: continue
        cv2.rectangle(img, (int(box2d[0]),int(box2d[1])), (int(box2d[2]),int(box2d[3])), (0,122,122), 3)
        cv2.putText(img,'%d-%s'%(i,classname),(int(box2d[0]),int(box2d[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,122,122),2,cv2.CV_AA) # For later OpenCV version use LINE_AA
    Image.fromarray(img).show()
    print pred2gt_idx_map

    #gt_box3d_list = [box3d for box3d in gt_box3d_list if np.linalg.norm(box3d[0,:])<=80]
    #draw_gt_boxes3d(box3d_pred_list, fig, color = (0,1,0), text_scale = (0.2,0.2,0.2), line_width=3)
    if img_id==403:
        box3d_pred_list = box3d_pred_list[0:2] 
    draw_gt_boxes3d(box3d_pred_list, fig, color = (0,1,0), text_scale = (0.1,0.1,0.1), line_width=3, color_list=color_list)
    #draw_gt_boxes3d(box3d_list, fig, color = (0,0,1), draw_text=False, line_width=3)
    #mlab.orientation_axes()
    #raw_input()
    #draw_gt_boxes3d(viz_gt_box3d_list, fig, color = (0,0,1), draw_text=False, line_width=3)

    #print np.max(pc_upright_camera[:,1])

    fig = mlab.figure(figure=None, bgcolor=(0.9,0.9,0.9), fgcolor=None, engine=None, size=(800, 800))
    mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.1, figure=fig)
    axes=np.array([
        [0.3,0.,0.,0.],
        [0.,0.3,0.,0.],
        [0.,0.,0.3,0.],
    ],dtype=np.float64)
    mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)



    mlab.points3d(pc_upright_camera[:,0], pc_upright_camera[:,1], pc_upright_camera[:,2], pc_upright_camera[:,2], mode='point', colormap='gnuplot', scale_factor=1, figure=fig)
    draw_gt_boxes3d(gt_box3d_list, fig, color = (0,0,1), draw_text=False, line_width=3, text_scale=(0.1,0.1,0.1))
    #mlab.orientation_axes()
    input_char = raw_input()
    if input_char=='s':
        np.savetxt('pc_%d.txt'%(img_id), pc_upright_camera, fmt='%.4f')
        output_arr = np.zeros((len(box3d_pred_list), 8, 3))
        for i in range(len(box3d_pred_list)):
            output_arr[i,:,:] = box3d_pred_list[i]
        print output_arr.shape
        print output_arr
        output_arr = output_arr.reshape((output_arr.shape[0], 24))
        np.savetxt('box3d_%d.txt'%(img_id), output_arr, fmt='%.4f')
