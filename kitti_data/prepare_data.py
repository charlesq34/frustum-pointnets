'''Provides 'object', which loads and parses raw KITTI data.'''

import os
import sys
import numpy as np
import cv2
from PIL import Image
import mayavi.mlab as mlab
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import kitti_util as utils
from viz_util import draw_lidar, draw_gt_boxes3d
import cPickle as pickle
from kitti_dataset import *

def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds

def extract_pc_in_box2d(pc, box2d):
    ''' pc: (N,2), box2d: (xmin,ymin,xmax,ymax) '''
    box2d_corners = np.zeros((4,2))
    box2d_corners[0,:] = [box2d[0],box2d[1]] 
    box2d_corners[1,:] = [box2d[2],box2d[1]] 
    box2d_corners[2,:] = [box2d[2],box2d[3]] 
    box2d_corners[3,:] = [box2d[0],box2d[3]] 
    box2d_roi_inds = in_hull(pc[:,0:2], box2d_corners)
    return pc[box2d_roi_inds,:], box2d_roi_inds
     
def demo():
    dataset = kitti_object('/home/rqi/Data/KITTI/object')
    data_idx = 0

    # Load data from dataset
    objects = dataset.get_label_objects(data_idx)
    objects[0].print_object()
    img = dataset.get_image(data_idx)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img_height, img_width, img_channel = img.shape
    print 'Image shape: ', img.shape
    pc_velo = dataset.get_lidar(data_idx)[:,0:3]
    calib = dataset.get_calibration(data_idx)

    # Draw lidar in rect camera coord
    pc_rect = calib.project_velo_to_rect(pc_velo)
    fig = draw_lidar(pc_rect)
    raw_input()

    # Draw 2d and 3d boxes on image
    show_image_with_boxes(img, objects, calib)
    raw_input()

    # Show all LiDAR points. Draw 3d box in LiDAR point cloud
    show_lidar_with_boxes(pc_velo, objects, calib)
    raw_input()
    show_lidar_with_boxes(pc_velo, objects, calib, True, img_width, img_height)
    raw_input()

    # Visualize LiDAR points on images
    show_lidar_on_image(pc_velo, img, calib, img_width, img_height) 
    raw_input()
    
    # Show LiDAR points that are in the 3d box
    box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(objects[0], calib.P) 
    box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
    box3droi_pc_velo, _ = extract_pc_in_box3d(pc_velo, box3d_pts_3d_velo)
    print 'Number of points in 3d box: ', box3droi_pc_velo.shape[0]

    fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(box3droi_pc_velo, fig=fig)
    draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig)
    mlab.show(1)
    raw_input()
    
    # UVDepth Image
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo, calib, 0, 0, img_width, img_height, True)
    imgfov_pts_2d = pts_2d[fov_inds,:]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    cameraUVDepth = np.zeros_like(imgfov_pc_rect)
    cameraUVDepth[:,0:2] = imgfov_pts_2d
    cameraUVDepth[:,2] = imgfov_pc_rect[:,2]

    # Verify the points are exactly the same
    backprojected_pc_velo = calib.project_image_to_velo(cameraUVDepth)
    print imgfov_pc_velo[0:20]
    print backprojected_pc_velo[0:20]
    raw_input()

    fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(backprojected_pc_velo, fig=fig)
    raw_input()

    # Only display those points that fall into 2d box
    xmin,ymin,xmax,ymax = objects[0].xmin, objects[0].ymin, objects[0].xmax, objects[0].ymax
    boxfov_pc_velo = get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax)
    print '2d box FOV poitn num: ', boxfov_pc_velo.shape[0]

    fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(boxfov_pc_velo, fig=fig)
    mlab.show(1)
    raw_input()

def random_shift_box2d(box2d, shift_ratio=0.1):
    ''' Randomly shift box center, randomly scale width and height 
    '''
    r = shift_ratio
    xmin,ymin,xmax,ymax = box2d
    h = ymax-ymin
    w = xmax-xmin
    cx = (xmin+xmax)/2.0
    cy = (ymin+ymax)/2.0
    cx2 = cx + w*r*(np.random.random()*2-1)
    cy2 = cy + h*r*(np.random.random()*2-1)
    h2 = h*(1+np.random.random()*2*r-r) # 0.9 to 1.1
    w2 = w*(1+np.random.random()*2*r-r) # 0.9 to 1.1
    return np.array([cx2-w2/2.0, cy2-h2/2.0, cx2+w2/2.0, cy2+h2/2.0])
 
def extract_roi_seg(idx_filename, split, output_filename, viz, perturb_box2d=False, augmentX=1, type_whitelist=['Car']):
    ''' Extract training data pairs for RoI point set segmentation.
        Given a frustum of points corresponding to a detected object in 2D with 2D box,
        Predict points in the frustum that are associated with the detected object.
        Update: put lidar points and 3d box in *rect camera* coord system (as that in 3d box label files)
        
        Input:
            idx_filename: each line is a number as sample ID
            split: corresponding to official either trianing or testing
            output_filename: the name for output .pickle file
            viz: whether to visualize extracted data
            perturb_box2d: whether to perturb the box2d (used for data augmentation in train set)
        Output:
            None (will write a .pickle file to the disk)

        Usage: extract_roi_seg("val_idx.txt", "training", "roi_seg_dataset_val.pickle")

    '''
    dataset = kitti_object('/home/rqi/Data/KITTI/object', split)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    id_list = [] # int number
    box2d_list = [] # [xmin,ymin,xmax,ymax]
    box3d_list = [] # (8,3) array in rect camera coord
    input_list = [] # channel number = 4, xyz,intensity in rect camera coord
    label_list = [] # 1 for roi object, 0 for clutter
    type_list = [] # string e.g. Car
    heading_list = [] # ry (along y-axis in rect camera coord) radius of clockwise angle from positive x axis in velo coord.
    box3d_size_list = [] # array of l,w,h
    frustum_angle_list = [] # angle of 2d box center from pos x-axis

    pos_cnt = 0
    all_cnt = 0
    for data_idx in data_idx_list:
        print '------------- ', data_idx
        calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)
        pc_velo = dataset.get_lidar(data_idx)
        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:,0:3] = calib.project_velo_to_rect(pc_velo[:,0:3])
        pc_rect[:,3] = pc_velo[:,3]
        img = dataset.get_image(data_idx)
        img_height, img_width, img_channel = img.shape
        _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:,0:3], calib, 0, 0, img_width, img_height, True)

        for obj_idx in range(len(objects)):
            if objects[obj_idx].type not in type_whitelist :continue

            # 2D BOX: Get pts rect backprojected 
            box2d = objects[obj_idx].box2d
            for _ in range(augmentX):
                # Augment data by box2d perturbation
                if perturb_box2d:
                    xmin,ymin,xmax,ymax = random_shift_box2d(box2d)
                    print box2d
                    print xmin,ymin,xmax,ymax
                else:
                    xmin,ymin,xmax,ymax = box2d
                box_fov_inds = (pc_image_coord[:,0]<xmax) & (pc_image_coord[:,0]>=xmin) & (pc_image_coord[:,1]<ymax) & (pc_image_coord[:,1]>=ymin)
                box_fov_inds = box_fov_inds & img_fov_inds
                pc_in_box_fov = pc_rect[box_fov_inds,:]
                # Get frustum angle (according to center pixel in 2D BOX)
                box2d_center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0])
                uvdepth = np.zeros((1,3))
                uvdepth[0,0:2] = box2d_center
                uvdepth[0,2] = 20 # some random depth
                box2d_center_rect = calib.project_image_to_rect(uvdepth)
                frustum_angle = -1 * np.arctan2(box2d_center_rect[0,2], box2d_center_rect[0,0]) # angle as to positive x-axis as in the Zoox paper
                # 3D BOX: Get pts velo in 3d box
                obj = objects[obj_idx]
                box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P) 
                _,inds = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)
                label = np.zeros((pc_in_box_fov.shape[0]))
                label[inds] = 1
                # Get 3D BOX heading
                heading_angle = obj.ry
                # Get 3D BOX size
                box3d_size = np.array([obj.l, obj.w, obj.h])

                # Reject too far away object or object without points
                if ymax-ymin<25 or np.sum(label)==0:
                    continue

                id_list.append(data_idx)
                box2d_list.append(np.array([xmin,ymin,xmax,ymax]))
                box3d_list.append(box3d_pts_3d)
                input_list.append(pc_in_box_fov)
                label_list.append(label)
                type_list.append(objects[obj_idx].type)
                heading_list.append(heading_angle)
                box3d_size_list.append(box3d_size)
                frustum_angle_list.append(frustum_angle)
    
                # collect statistics
                pos_cnt += np.sum(label)
                all_cnt += pc_in_box_fov.shape[0]
        
    print 'Average pos ratio: ', pos_cnt/float(all_cnt)
    print 'Average npoints: ', float(all_cnt)/len(id_list)
    
    with open(output_filename,'wb') as fp:
        pickle.dump(id_list, fp)
        pickle.dump(box2d_list,fp)
        pickle.dump(box3d_list,fp)
        pickle.dump(input_list, fp)
        pickle.dump(label_list, fp)
        pickle.dump(type_list, fp)
        pickle.dump(heading_list, fp)
        pickle.dump(box3d_size_list, fp)
        pickle.dump(frustum_angle_list, fp)
    
    import sys
    sys.path.append('../')
    from view_pc import draw_lidar, draw_gt_boxes3d
    if viz:
        for i in range(10):
            p1 = input_list[i]
            seg = label_list[i] 
            #fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
            #draw_lidar(p1[:,0:3], fig=fig)
            fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4), fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:,0], p1[:,1], p1[:,2], seg, mode='point', colormap='gnuplot', scale_factor=1, figure=fig)
            fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4), fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:,2], -p1[:,0], -p1[:,1], seg, mode='point', colormap='gnuplot', scale_factor=1, figure=fig)
            raw_input()

def get_box3d_dim_statistics(idx_filename):
    dataset = kitti_object('/home/rqi/Data/KITTI/object')
    dimension_list = []
    type_list = []
    ry_list = []
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
    for data_idx in data_idx_list:
        print '------------- ', data_idx
        calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)
        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]
            if obj.type=='DontCare':continue
            dimension_list.append(np.array([obj.l,obj.w,obj.h])) 
            type_list.append(obj.type) 
            ry_list.append(obj.ry)

    with open('box3d_dimensions.pickle','wb') as fp:
        pickle.dump(type_list, fp)
        pickle.dump(dimension_list, fp)
        pickle.dump(ry_list, fp)

def read_det_file(det_filename):
    det_id2str = {1:'Pedestrian', 2:'Car', 3:'Cyclist'}
    id_list = []
    type_list = []
    prob_list = []
    box2d_list = []
    for line in open(det_filename, 'r'):
        t = line.rstrip().split(" ")
        id_list.append(int(os.path.basename(t[0]).rstrip('.png')))
        type_list.append(det_id2str[int(t[1])])
        prob_list.append(float(t[2]))
        box2d_list.append(np.array([float(t[i]) for i in range(3,7)]))
    return id_list, type_list, box2d_list, prob_list

def test_read_det_file():
    ids, types, box2ds, probs = read_det_file('/mnt/nurostorage0/user/w/temp/train_rqi_results.txt')
    for i in range(100):
       print ids[i], types[i], box2ds[i], probs[i]

 
def extract_roi_seg_from_rgb_detection(det_filename, split, output_filename, viz, valid_id_list=None, type_whitelist=['Car'], img_height_threshold=25, lidar_point_threshold=5):
    ''' Extract data pairs for RoI point set segmentation from RGB detector outputed 2D boxes.
        Update: put lidar points and 3d box in *rect camera* coord system (as that in 3d box label files)
        
        Input:
            det_filename: each line is img_path typeid confidence xmin ymin xmax ymax
            split: corresponding to official either trianing or testing
            output_filename: the name for output .pickle file
            valid_id_list: specify a list of valid image IDs
        Output:
            None (will write a .pickle file to the disk)

        Usage: extract_roi_seg_from_rgb_detection("val_rqi_results.txt", "training", "roi_seg_val_rgb_detector_0908.pickle")

    '''
    dataset = kitti_object('/home/rqi/Data/KITTI/object', split)
    det_id_list, det_type_list, det_box2d_list, det_prob_list = read_det_file(det_filename)
    cache_id = -1
    cache = None
    
    id_list = []
    type_list = []
    box2d_list = []
    prob_list = []
    input_list = [] # channel number = 4, xyz,intensity in rect camera coord
    frustum_angle_list = [] # angle of 2d box center from pos x-axis

    for det_idx in range(len(det_id_list)):
        data_idx = det_id_list[det_idx]
        if valid_id_list is not None and data_idx not in valid_id_list: continue
        print 'det idx: %d/%d, data idx: %d' % (det_idx, len(det_id_list), data_idx)
        if cache_id != data_idx:
            calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
            #objects = dataset.get_label_objects(data_idx)
            pc_velo = dataset.get_lidar(data_idx)
            pc_rect = np.zeros_like(pc_velo)
            pc_rect[:,0:3] = calib.project_velo_to_rect(pc_velo[:,0:3])
            pc_rect[:,3] = pc_velo[:,3]
            img = dataset.get_image(data_idx)
            img_height, img_width, img_channel = img.shape
            _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:,0:3], calib, 0, 0, img_width, img_height, True)
            cache = [calib,pc_rect,pc_image_coord,img_fov_inds]
            cache_id = data_idx
        else:
            calib,pc_rect,pc_image_coord,img_fov_inds = cache

       
        #if det_type_list[det_idx]!='Car': continue
        if det_type_list[det_idx] not in type_whitelist: continue

        # 2D BOX: Get pts rect backprojected 
        xmin,ymin,xmax,ymax = det_box2d_list[det_idx]
        box_fov_inds = (pc_image_coord[:,0]<xmax) & (pc_image_coord[:,0]>=xmin) & (pc_image_coord[:,1]<ymax) & (pc_image_coord[:,1]>=ymin)
        box_fov_inds = box_fov_inds & img_fov_inds
        pc_in_box_fov = pc_rect[box_fov_inds,:]
        # Get frustum angle (according to center pixel in 2D BOX)
        box2d_center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0])
        uvdepth = np.zeros((1,3))
        uvdepth[0,0:2] = box2d_center
        uvdepth[0,2] = 20 # some random depth
        box2d_center_rect = calib.project_image_to_rect(uvdepth)
        frustum_angle = -1 * np.arctan2(box2d_center_rect[0,2], box2d_center_rect[0,0]) # angle as to positive x-axis as in the Zoox paper
        
        # Pass objects that are too small
        if ymax-ymin<img_height_threshold or len(pc_in_box_fov)<lidar_point_threshold:
            continue
       
        id_list.append(data_idx)
        type_list.append(det_type_list[det_idx])
        box2d_list.append(det_box2d_list[det_idx])
        prob_list.append(det_prob_list[det_idx])
        input_list.append(pc_in_box_fov)
        frustum_angle_list.append(frustum_angle)
    
    print len(id_list)
    with open(output_filename,'wb') as fp:
        pickle.dump(id_list, fp)
        pickle.dump(box2d_list,fp)
        pickle.dump(input_list, fp)
        pickle.dump(type_list, fp)
        pickle.dump(frustum_angle_list, fp)
        pickle.dump(prob_list, fp)
    
    import sys
    sys.path.append('../')
    from view_pc import draw_lidar, draw_gt_boxes3d
    if viz:
        for i in range(10):
            p1 = input_list[i]
            fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4), fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:,0], p1[:,1], p1[:,2], p1[:,1], mode='point', colormap='gnuplot', scale_factor=1, figure=fig)
            fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4), fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:,2], -p1[:,0], -p1[:,1], seg, mode='point', colormap='gnuplot', scale_factor=1, figure=fig)
            raw_input()

def write_2d_rgb_detection(det_filename, split, result_dir):
    ''' Write 2D detection results for KITTI evaluation.
        Convert from Wei's format to KITTI format. 
        
        Input:
            det_filename: each line is img_path typeid confidence xmin ymin xmax ymax
            split: corresponding to official either trianing or testing
            result_dir: folder path for results dumping
        Output:
            None (will write <xxx>.txt files to disk)

        Usage: write_2d_rgb_detection("val_rqi_results.txt", "training", "results")

    '''
    dataset = kitti_object('/home/rqi/Data/KITTI/object', split)
    det_id_list, det_type_list, det_box2d_list, det_prob_list = read_det_file(det_filename)
    results = {} # map from idx to list of strings, each string is a line without \n
    for i in range(len(det_id_list)):
        idx = det_id_list[i]
        typename = det_type_list[i]
        box2d = det_box2d_list[i]
        prob = det_prob_list[i]
        output_str = typename + " -1 -1 -10 "
        output_str += "%f %f %f %f " % (box2d[0],box2d[1],box2d[2],box2d[3])
        output_str += "-1 -1 -1 -1000 -1000 -1000 -10 %f" % (prob)
        if idx not in results: results[idx] = []
        results[idx].append(output_str)
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    output_dir = os.path.join(result_dir, 'data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    for idx in results:
        pred_filename = os.path.join(output_dir, '%06d.txt'%(idx))
        fout = open(pred_filename, 'w')
        for line in results[idx]:
            fout.write(line+'\n')
        fout.close() 

if __name__=='__main__':
    #demo()
    #get_box3d_dim_statistics("ImageSets/train.txt")
    #write_2d_rgb_detection("../results/val_rqi_results3.txt", "training", "rgb_results3_val")
    #extract_roi_seg('ImageSets/train.txt', 'training', 'kitti3d_carpedcyclist_train.pickle', viz=False, perturb_box2d=True, augmentX=5, type_whitelist=['Car', 'Pedestrian', 'Cyclist'])
    #extract_roi_seg('ImageSets/val.txt', 'training', 'kitti3d_carpedcyclist_val.pickle', viz=False, perturb_box2d=False, augmentX=1, type_whitelist=['Car', 'Pedestrian', 'Cyclist'])
    #extract_roi_seg('ImageSets/trainval.txt', 'training', 'kitti3d_carpedcyclist_trainval.pickle', viz=False, perturb_box2d=True, augmentX=5, type_whitelist=['Car', 'Pedestrian', 'Cyclist'])
