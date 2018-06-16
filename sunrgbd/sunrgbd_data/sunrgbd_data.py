''' Helper class and functions for loading SUN RGB-D objects

Author: Charles R. Qi
Date: October 2017

TODO: code formatting and clean-up.
'''

import os
import sys
import numpy as np
from mayavi import mlab
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import utils
from utils import random_shift_box2d, extract_pc_in_box3d
sys.path.append(os.path.join(BASE_DIR, '../../mayavi'))
from viz_util import draw_gt_boxes3d, draw_lidar
import cv2
from PIL import Image

data_dir = BASE_DIR

class sunrgbd_object(object):
    ''' Load and parse object data '''
    def __init__(self, root_dir, split='training'):
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        if split == 'training':
            self.num_samples = 10335
        elif split == 'testing':
            self.num_samples = 2860
        else:
            print('Unknown split: %s' % (split))
            exit(-1)

        self.image_dir = os.path.join(self.split_dir, 'image')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.depth_dir = os.path.join(self.split_dir, 'depth')
        self.label_dir = os.path.join(self.split_dir, 'label_dimension')
        #self.label_dimension_dir = os.path.join(self.split_dir, 'label_dimension')

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        img_filename = os.path.join(self.image_dir, '%06d.jpg'%(idx))
        return utils.load_image(img_filename)

    def get_depth(self, idx): 
        depth_filename = os.path.join(self.depth_dir, '%06d.txt'%(idx))
        return utils.load_depth_points(depth_filename)

    def get_calibration(self, idx):
        calib_filename = os.path.join(self.calib_dir, '%06d.txt'%(idx))
        return utils.SUNRGBD_Calibration(calib_filename)

    def get_label_objects(self, idx):
        assert(self.split=='training') 
        label_filename = os.path.join(self.label_dir, '%06d.txt'%(idx))
        return utils.read_sunrgbd_label(label_filename)

def dataset_viz(show_frustum=False):  
    sunrgbd = sunrgbd_object(data_dir)
    idxs = np.array(range(1,len(sunrgbd)+1))
    np.random.shuffle(idxs)
    for idx in range(len(sunrgbd)):
        data_idx = idxs[idx]
        print('--------------------', data_idx)
        pc = sunrgbd.get_depth(data_idx)
        print(pc.shape)
        
        # Project points to image
        calib = sunrgbd.get_calibration(data_idx)
        uv,d = calib.project_upright_depth_to_image(pc[:,0:3])
        print(uv)
        print(d)
        raw_input()
        
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap('hsv', 256)
        cmap = np.array([cmap(i) for i in range(256)])[:,:3]*255
        
        img = sunrgbd.get_image(data_idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        for i in range(uv.shape[0]):
            depth = d[i]
            color = cmap[int(120.0/depth),:]
            cv2.circle(img, (int(np.round(uv[i,0])), int(np.round(uv[i,1]))), 2, color=tuple(color), thickness=-1)
        Image.fromarray(img).show() 
        raw_input()
        
        # Load box labels
        objects = sunrgbd.get_label_objects(data_idx)
        print(objects)
        raw_input()
        
        # Draw 2D boxes on image
        img = sunrgbd.get_image(data_idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        for i,obj in enumerate(objects):
            cv2.rectangle(img, (int(obj.xmin),int(obj.ymin)), (int(obj.xmax),int(obj.ymax)), (0,255,0), 2)
            cv2.putText(img, '%d %s'%(i,obj.classname), (max(int(obj.xmin),15), max(int(obj.ymin),15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        Image.fromarray(img).show()
        raw_input()
        
        # Draw 3D boxes on depth points
        box3d = []
        ori3d = []
        for obj in objects:
            corners_3d_image, corners_3d = utils.compute_box_3d(obj, calib)
            ori_3d_image, ori_3d = utils.compute_orientation_3d(obj, calib)
            print('Corners 3D: ', corners_3d)
            box3d.append(corners_3d)
            ori3d.append(ori_3d)
        raw_input()
        
        bgcolor=(0,0,0)
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000))
        mlab.points3d(pc[:,0], pc[:,1], pc[:,2], pc[:,2], mode='point', colormap='gnuplot', figure=fig)
        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)
        draw_gt_boxes3d(box3d, fig=fig)
        for i in range(len(ori3d)):
            ori_3d = ori3d[i]
            x1,y1,z1 = ori_3d[0,:]
            x2,y2,z2 = ori_3d[1,:]
            mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.orientation_axes()
        for i,obj in enumerate(objects):
            print('Orientation: ', i, np.arctan2(obj.orientation[1], obj.orientation[0]))
            print('Dimension: ', i, obj.l, obj.w, obj.h)
        raw_input()
       
        if show_frustum: 
            img = sunrgbd.get_image(data_idx)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            for i,obj in enumerate(objects):
                box2d_fov_inds = (uv[:,0]<obj.xmax) & (uv[:,0]>=obj.xmin) & (uv[:,1]<obj.ymax) & (uv[:,1]>=obj.ymin)
                box2d_fov_pc = pc[box2d_fov_inds, :]
                img2 = np.copy(img)
                cv2.rectangle(img2, (int(obj.xmin),int(obj.ymin)), (int(obj.xmax),int(obj.ymax)), (0,255,0), 2)
                cv2.putText(img2, '%d %s'%(i,obj.classname), (max(int(obj.xmin),15), max(int(obj.ymin),15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
                Image.fromarray(img2).show()
                
                fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1000, 1000))
                mlab.points3d(box2d_fov_pc[:,0], box2d_fov_pc[:,1], box2d_fov_pc[:,2], box2d_fov_pc[:,2], mode='point', colormap='gnuplot', figure=fig)
                raw_input()


def extract_roi_seg(idx_filename, split, output_filename, viz, perturb_box2d=False, augmentX=1, type_whitelist=['bed','table','sofa','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub']):
    dataset = sunrgbd_object('/home/rqi/Data/mysunrgbd', split)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    id_list = [] # int number
    box2d_list = [] # [xmin,ymin,xmax,ymax]
    box3d_list = [] # (8,3) array in upright depth coord
    input_list = [] # channel number = 6, xyz,rgb in upright depth coord
    label_list = [] # 1 for roi object, 0 for clutter
    type_list = [] # string e.g. bed
    heading_list = [] # face of object angle, radius of clockwise angle from positive x axis in upright camera coord
    box3d_size_list = [] # array of l,w,h
    frustum_angle_list = [] # angle of 2d box center from pos x-axis (clockwise)

    pos_cnt = 0
    all_cnt = 0
    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx)
        objects = dataset.get_label_objects(data_idx)
        pc_upright_depth = dataset.get_depth(data_idx)
        pc_upright_camera = np.zeros_like(pc_upright_depth)
        pc_upright_camera[:,0:3] = calib.project_upright_depth_to_upright_camera(pc_upright_depth[:,0:3])
        pc_upright_camera[:,3:] = pc_upright_depth[:,3:]
        if viz:
            mlab.points3d(pc_upright_camera[:,0], pc_upright_camera[:,1], pc_upright_camera[:,2], pc_upright_camera[:,1], mode='point')
            mlab.orientation_axes()
            raw_input()
        img = dataset.get_image(data_idx)
        img_height, img_width, img_channel = img.shape
        pc_image_coord,_ = calib.project_upright_depth_to_image(pc_upright_depth)
        #print('PC image coord: ', pc_image_coord)

        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]
            if obj.classname not in type_whitelist: continue

            # 2D BOX: Get pts rect backprojected 
            box2d = obj.box2d
            for _ in range(augmentX):
                try:
                    # Augment data by box2d perturbation
                    if perturb_box2d:
                        xmin,ymin,xmax,ymax = random_shift_box2d(box2d)
                        print(xmin,ymin,xmax,ymax)
                    else:
                        xmin,ymin,xmax,ymax = box2d
                    box_fov_inds = (pc_image_coord[:,0]<xmax) & (pc_image_coord[:,0]>=xmin) & (pc_image_coord[:,1]<ymax) & (pc_image_coord[:,1]>=ymin)
                    pc_in_box_fov = pc_upright_camera[box_fov_inds,:]
                    # Get frustum angle (according to center pixel in 2D BOX)
                    box2d_center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0])
                    uvdepth = np.zeros((1,3))
                    uvdepth[0,0:2] = box2d_center
                    uvdepth[0,2] = 20 # some random depth
                    box2d_center_upright_camera = calib.project_image_to_upright_camerea(uvdepth)
                    print('UVdepth, center in upright camera: ', uvdepth, box2d_center_upright_camera)
                    frustum_angle = -1 * np.arctan2(box2d_center_upright_camera[0,2], box2d_center_upright_camera[0,0]) # angle as to positive x-axis as in the Zoox paper
                    print('Frustum angle: ', frustum_angle)
                    # 3D BOX: Get pts velo in 3d box
                    box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib) 
                    box3d_pts_3d = calib.project_upright_depth_to_upright_camera(box3d_pts_3d)
                    _,inds = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)
                    print(len(inds))
                    label = np.zeros((pc_in_box_fov.shape[0]))
                    label[inds] = 1
                    # Get 3D BOX heading
                    print('Orientation: ', obj.orientation)
                    print('Heading angle: ', obj.heading_angle)
                    # Get 3D BOX size
                    box3d_size = np.array([2*obj.l,2*obj.w,2*obj.h])
                    print('Box3d size: ', box3d_size)
                    print('Type: ', obj.classname)
                    print('Num of point: ', pc_in_box_fov.shape[0])
                    
                    # Subsample points..
                    num_point = pc_in_box_fov.shape[0]
                    if num_point > 2048:
                        choice = np.random.choice(pc_in_box_fov.shape[0], 2048, replace=False)
                        pc_in_box_fov = pc_in_box_fov[choice,:]
                        label = label[choice]
                    # Reject object with too few points
                    if np.sum(label) < 5:
                        continue

                    id_list.append(data_idx)
                    box2d_list.append(np.array([xmin,ymin,xmax,ymax]))
                    box3d_list.append(box3d_pts_3d)
                    input_list.append(pc_in_box_fov)
                    label_list.append(label)
                    type_list.append(obj.classname)
                    heading_list.append(obj.heading_angle)
                    box3d_size_list.append(box3d_size)
                    frustum_angle_list.append(frustum_angle)
    
                    # collect statistics
                    pos_cnt += np.sum(label)
                    all_cnt += pc_in_box_fov.shape[0]
       
                    # VISUALIZATION
                    if viz:
                        img2 = np.copy(img)
                        cv2.rectangle(img2, (int(obj.xmin),int(obj.ymin)), (int(obj.xmax),int(obj.ymax)), (0,255,0), 2)
                        utils.draw_projected_box3d(img2, box3d_pts_2d)
                        Image.fromarray(img2).show()
                        p1 = input_list[-1]
                        seg = label_list[-1] 
                        fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4), fgcolor=None, engine=None, size=(500, 500))
                        mlab.points3d(p1[:,0], p1[:,1], p1[:,2], seg, mode='point', colormap='gnuplot', scale_factor=1, figure=fig)
                        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)
                        draw_gt_boxes3d([box3d_pts_3d], fig=fig)
                        mlab.orientation_axes()
                        raw_input()
                except:
                    pass

    print('Average pos ratio: ', pos_cnt/float(all_cnt))
    print('Average npoints: ', float(all_cnt)/len(id_list))

    utils.save_zipped_pickle([id_list,box2d_list,box3d_list,input_list,label_list,type_list,heading_list,box3d_size_list,frustum_angle_list],output_filename)

 
def get_box3d_dim_statistics(idx_filename, type_whitelist=['bed','table','sofa','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub']):
    dataset = sunrgbd_object('/home/rqi/Data/mysunrgbd')
    dimension_list = []
    type_list = []
    ry_list = []
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)
        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]
            if obj.classname not in type_whitelist: continue
            heading_angle = -1 * np.arctan2(obj.orientation[1], obj.orientation[0])
            dimension_list.append(np.array([obj.l,obj.w,obj.h])) 
            type_list.append(obj.classname) 
            ry_list.append(heading_angle)

    import cPickle as pickle
    with open('box3d_dimensions.pickle','wb') as fp:
        pickle.dump(type_list, fp)
        pickle.dump(dimension_list, fp)
        pickle.dump(ry_list, fp)

def read_det_folder(det_folder):
    filenames = os.listdir(det_folder)
    id_list = []
    type_list = []
    prob_list = []
    box2d_list = []
    for filename in filenames:
        img_id = int(filename[0:6])
        full_filename = os.path.join(det_folder, filename)
        for line in open(full_filename, 'r'):
            t = line.rstrip().split(" ")
            prob = float(t[-1])
            if prob < 0.05: continue
            id_list.append(img_id)
            type_list.append(t[0]) 
            prob_list.append(prob)
            box2d_list.append(np.array([float(t[i]) for i in range(4,8)]))
    return id_list, type_list, box2d_list, prob_list


def extract_roi_seg_from_rgb_detection(det_folder, split, output_filename, viz, valid_id_list=None, type_whitelist=['bed','table','sofa','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub']):
    ''' Extract data pairs for RoI point set segmentation from RGB detector outputed 2D boxes.
        
        Input:
            det_folder: contains files for each frame, lines in each file are type -1 -10 -10 xmin ymin xmax ymax ... prob
            split: corresponding to official either trianing or testing
            output_filename: the name for output .pickle file
            valid_id_list: specify a list of valid image IDs
        Output:
            None (will write a .pickle file to the disk)

        Usage: extract_roi_seg_from_rgb_detection("val_result_folder", "training", "roi_seg_val_rgb_detector_0908.pickle")

    '''
    dataset = sunrgbd_object('/home/rqi/Data/mysunrgbd', split)
    det_id_list, det_type_list, det_box2d_list, det_prob_list = read_det_folder(det_folder)
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
        print('det idx: %d/%d, data idx: %d' % (det_idx, len(det_id_list), data_idx))
        if cache_id != data_idx:
            calib = dataset.get_calibration(data_idx)
            pc_upright_depth = dataset.get_depth(data_idx)
            pc_upright_camera = np.zeros_like(pc_upright_depth)
            pc_upright_camera[:,0:3] = calib.project_upright_depth_to_upright_camera(pc_upright_depth[:,0:3])
            pc_upright_camera[:,3:] = pc_upright_depth[:,3:]

            img = dataset.get_image(data_idx)
            img_height, img_width, img_channel = img.shape
            pc_image_coord,_ = calib.project_upright_depth_to_image(pc_upright_depth)
            cache = [calib,pc_upright_camera,pc_image_coord]
            cache_id = data_idx
        else:
            calib,pc_upright_camera,pc_image_coord = cache

       
        #if det_type_list[det_idx]!='Car': continue
        if det_type_list[det_idx] not in type_whitelist: continue

        # 2D BOX: Get pts rect backprojected 
        xmin,ymin,xmax,ymax = det_box2d_list[det_idx]
        box_fov_inds = (pc_image_coord[:,0]<xmax) & (pc_image_coord[:,0]>=xmin) & (pc_image_coord[:,1]<ymax) & (pc_image_coord[:,1]>=ymin)
        pc_in_box_fov = pc_upright_camera[box_fov_inds,:]
        # Get frustum angle (according to center pixel in 2D BOX)
        box2d_center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0])
        uvdepth = np.zeros((1,3))
        uvdepth[0,0:2] = box2d_center
        uvdepth[0,2] = 20 # some random depth
        box2d_center_upright_camera = calib.project_image_to_upright_camerea(uvdepth)
        frustum_angle = -1 * np.arctan2(box2d_center_upright_camera[0,2], box2d_center_upright_camera[0,0]) # angle as to positive x-axis as in the Zoox paper
        # Subsample points..
        num_point = pc_in_box_fov.shape[0]
        if num_point > 2048:
            choice = np.random.choice(pc_in_box_fov.shape[0], 2048, replace=False)
            pc_in_box_fov = pc_in_box_fov[choice,:]
 
        # Pass objects that are too small
        if len(pc_in_box_fov)<5:
            continue
       
        id_list.append(data_idx)
        type_list.append(det_type_list[det_idx])
        box2d_list.append(det_box2d_list[det_idx])
        prob_list.append(det_prob_list[det_idx])
        input_list.append(pc_in_box_fov)
        frustum_angle_list.append(frustum_angle)
    
    utils.save_zipped_pickle([id_list, box2d_list, input_list, type_list, frustum_angle_list, prob_list], output_filename)
    
    if viz:
        for i in range(10):
            p1 = input_list[i]
            fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4), fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:,0], p1[:,1], p1[:,2], p1[:,1], mode='point', colormap='gnuplot', scale_factor=1, figure=fig)
            fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4), fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:,2], -p1[:,0], -p1[:,1], seg, mode='point', colormap='gnuplot', scale_factor=1, figure=fig)
            raw_input()


if __name__=='__main__':
    #dataset_viz()
    #get_box3d_dim_statistics('/home/rqi/Data/mysunrgbd/training/train_data_idx.txt')
    extract_roi_seg('/home/rqi/Data/mysunrgbd/training/val_data_idx.txt', 'training', output_filename='val_1002.zip.pickle', viz=False, augmentX=1) 
    extract_roi_seg('/home/rqi/Data/mysunrgbd/training/train_data_idx.txt', 'training', output_filename='train_1002_aug5x.zip.pickle', viz=False, augmentX=5) 
    #extract_roi_seg_from_rgb_detection('FPN_384x384', 'training', 'fcn_det_val.zip.pickle', valid_id_list=[int(line.rstrip()) for line in open('/home/rqi/Data/mysunrgbd/training/val_data_idx.txt')], viz=True)
