def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        cPickle.dump(obj, f, protocol)

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = cPickle.load(f)
        return loaded_object


def my_crop_img(img, xmin, ymin, xmax, ymax):
    img_height, img_width, img_channel = img.shape
    print img.shape
    xmin = max(0,np.round(xmin))
    ymin = max(0,np.round(ymin))
    xmax = min(img_width, np.round(xmax))
    ymax = min(img_height, np.round(ymax))
    print xmin,ymin,xmax,ymax
    if ymin==ymax or xmin==xmax:
        return None
    img_patch = img[int(ymin):int(ymax),int(xmin):int(xmax),:]
    print img_patch.shape
    return img_patch



def extract_lidar_depth_map(idx_filename, split, output_folder, viz, type_whitelist=['Car']):
    ''' Extract LiDAR points as sparse depth map images
        Update: put lidar points and 3d box in *rect camera* coord system (as that in 3d box label files)
        
        Input:
            idx_filename: each line is a number as sample ID
            split: corresponding to official either trianing or testing
            output_filename: the name for output .pickle file
            viz: whether to visualize extracted data
        Output:
            None (will write a .pickle file to the disk)

        Usage: extract_lidar_depth_map("val_idx.txt", "training", "roi_seg_dataset_val.pickle")

    '''
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    dataset = kitti_object('/home/rqi/Data/KITTI/object', split)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    pos_cnt = 0
    all_cnt = 0
    for data_idx in data_idx_list:
        print '------------- ', data_idx
        calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
        pc_velo = dataset.get_lidar(data_idx)
        #print pc_velo[0:20,:]
        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:,0:3] = calib.project_velo_to_rect(pc_velo[:,0:3])
        pc_rect[:,3] = pc_velo[:,3]
        img = dataset.get_image(data_idx)
        img_height, img_width, img_channel = img.shape
        _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:,0:3], calib, 0, 0, img_width, img_height, True)
        pc_in_img_fov = pc_rect[img_fov_inds]
        #print pc_in_img_fov.shape
        #raw_input()
        uvmap = calib.project_rect_to_image(pc_in_img_fov[:,0:3])
        output_array = np.concatenate((uvmap, pc_in_img_fov[:,2:4]), axis=1)
        #print output_array.shape
        np.savetxt(os.path.join(output_folder, '%06d.txt'%(data_idx)), output_array, delimiter=' ', fmt='%.6f')
   
        if viz:
            import matplotlib.pyplot as plt
            cmap = plt.cm.get_cmap('hsv', 256)
            cmap = np.array([cmap(i) for i in range(256)])[:,:3]*255

            for i in range(output_array.shape[0]):
                depth = output_array[i,2]
                color = cmap[int(360.0/depth),:]
                cv2.circle(img, (int(np.round(output_array[i,0])), int(np.round(output_array[i,1]))), 2, color=tuple(color), thickness=-1)
            Image.fromarray(img).show() 

def extract_roi_seg_2dmask(idx_filename, split, output_filename, viz, perturb_box2d=False, augmentX=1, type_whitelist=['Car'], use_tight_box=False):
    ''' Extract training data pairs for RoI point set segmentation.
        Given a frustum of points corresponding to a detected object in 2D with 2D box,
        Predict points in the frustum that are associated with the detected object.
        Update: put lidar points and 3d box in *rect camera* coord system (as that in 3d box label files)

        UPDATE: Using 2D mask to further constrain point cloud in frustum.
        
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
    import json
    import sys
    sys.path.append('/home/rqi/Data/COCO/PythonAPI/')
    from pycocotools import mask as maskUtils

    assert(split=='training') # only have 2D mask on trainval set.
    data_dir = '/home/rqi/Data/KITTI/object/training/'

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
        anno_file = "{}/label_mask/{}.json".format(data_dir, '%06d'%(data_idx))
        with open(anno_file, "r") as f:
            anno = json.load(f);

        calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)

        #print len(objects)
        #for obj in objects:
        #    print obj.type, obj.box2d
        #print len(anno['annotation'])
        #for ann in anno['annotation']:
        #    print ann['category_id']

        masks_2d = []
        boxes_2d = []
        # --- BEGIN --- Match objects and masks
        for ann_idx, ann in enumerate(anno['annotation']):
            assert(ann['category_id'] == objects[ann_idx].type)
            m = maskUtils.decode([ann['segmentation']])
            masks_2d.append(m)
            ann_bbox = np.copy(ann['bbox'])
            ann_bbox[2] = ann_bbox[0]+ann_bbox[2]
            ann_bbox[3] = ann_bbox[1]+ann_bbox[3]
            boxes_2d.append(ann_bbox)
        '''
        #print anno['annotation'][matched_ann_idx]['bbox'], obj.box2d, m.shape
        for obj in objects:
            #print obj.type
            matched_ann_idx = -1
            min_bbox_dist = 9999.0
            for ann_idx, ann in enumerate(anno['annotation']):
                ann_bbox = np.copy(ann['bbox'])
                ann_bbox[2] = ann_bbox[0]+ann_bbox[2]
                ann_bbox[3] = ann_bbox[1]+ann_bbox[3]
                bbox_dist = np.sum((np.array(obj.box2d) - np.array(ann_bbox))**2)
                print obj.box2d, ann_bbox, bbox_dist
                if ann['category_id']==obj.type and bbox_dist < min_bbox_dist:
                    matched_ann_idx = ann_idx
                    min_bbox_dist = bbox_dist
            assert(matched_ann_idx>=0) 
            m = maskUtils.decode([anno['annotation'][matched_ann_idx]['segmentation']])
            #print anno['annotation'][matched_ann_idx]['bbox'], obj.box2d, m.shape
            masks_2d.append(m)
            #raw_input()
        '''
        # --- END --- Match objects and masks
                
        pc_velo = dataset.get_lidar(data_idx)
        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:,0:3] = calib.project_velo_to_rect(pc_velo[:,0:3])
        pc_rect[:,3] = pc_velo[:,3]
        img = dataset.get_image(data_idx)
        img_height, img_width, img_channel = img.shape
        _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:,0:3], calib, 0, 0, img_width, img_height, True)

        for obj_idx in range(len(objects)):
            #if objects[obj_idx].type=='DontCare':continue
            #if objects[obj_idx].type!='Car':continue
            if objects[obj_idx].type not in type_whitelist :continue

            # 2D BOX: Get pts rect backprojected 
            if use_tight_box:
                box2d = boxes_2d[obj_idx]
                print objects[obj_idx].box2d, box2d
            else:
                box2d = objects[obj_idx].box2d
            mask2d = masks_2d[obj_idx] 
            for _ in range(augmentX):
                # Augment data by box2d perturbation
                if perturb_box2d:
                    xmin,ymin,xmax,ymax = random_shift_box2d(box2d)
                    print box2d
                    print xmin,ymin,xmax,ymax
                else:
                    xmin,ymin,xmax,ymax = box2d
                box_fov_inds = (pc_image_coord[:,0]<xmax) & (pc_image_coord[:,0]>=xmin) & (pc_image_coord[:,1]<ymax) & (pc_image_coord[:,1]>=ymin)
                pc_image_coord_x = np.maximum(np.minimum(np.round(pc_image_coord[:,0]), img_width-1), 0).astype(np.int32)
                pc_image_coord_y = np.maximum(np.minimum(np.round(pc_image_coord[:,1]), img_height-1), 0).astype(np.int32)

                #print data_idx, obj.type, obj.box2d, mask2d.shape, np.min(mask2d), np.max(mask2d), np.sum(mask2d)
                #if obj.type=='Car':
                #    Image.fromarray(np.tile(mask2d*255,(1,1,3)).astype(np.uint8)).show()

                mask_fov_inds = (mask2d[pc_image_coord_y, pc_image_coord_x, 0] > 0)
                #print 'mask_fov_inds shape: ', mask_fov_inds.shape
                #raw_input()
                print 'original num points: ', np.sum(box_fov_inds & img_fov_inds)
                box_fov_inds = mask_fov_inds & box_fov_inds & img_fov_inds
                print '+ 2dmask num points: ', np.sum(box_fov_inds)
                #raw_input()
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
    
    import cPickle as pickle
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

def extract_roi_rgbd_img(idx_filename, split, output_filename, viz, perturb_box2d=False, augmentX=1, type_whitelist=['Car'], use_tight_box=False):
    ''' Extract RGB-D images (R,G,B,depth,intensity)
        
        Input:
            idx_filename: each line is a number as sample ID
            split: corresponding to official either trianing or testing
            output_filename: output .pickle file
            viz: whether to visualize extracted data
            perturb_box2d: whether to perturb the box2d (used for data augmentation in train set)
        Output:
            None (will write indivisual rgbd images as npy file)

    '''
    import json
    import sys
    sys.path.append('/home/rqi/Data/COCO/PythonAPI/')
    from pycocotools import mask as maskUtils

    assert(split=='training') # only have 2D mask on trainval set.
    data_dir = '/home/rqi/Data/KITTI/object/training/'

    dataset = kitti_object('/home/rqi/Data/KITTI/object', split)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    id_list = [] # int number
    box2d_list = [] # [xmin,ymin,xmax,ymax]
    box3d_list = [] # (8,3) array in rect camera coord
    box3d_uv_list = [] # (2,) array in relative image uv coord (relative to 2d box)
    input_list = [] # rgb image, binary mask and lidar sparse image ((HxWx3 uint8), (HxWx1 binary float), (Nx4 u,v,depth,intensity))
    label_list = [] # 1 for roi object, 0 for clutter
    type_list = [] # string e.g. Car
    heading_list = [] # ry (along y-axis in rect camera coord) radius of clockwise angle from positive x axis in velo coord.
    box3d_size_list = [] # array of l,w,h
    frustum_angle_list = [] # angle of 2d box center from pos x-axis

    pos_cnt = 0
    all_cnt = 0
    for data_idx in data_idx_list:
        print '------------- ', data_idx
        anno_file = "{}/label_mask/{}.json".format(data_dir, '%06d'%(data_idx))
        with open(anno_file, "r") as f:
            anno = json.load(f);

        calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)

        masks_2d = []
        boxes_2d = [] # tighter 2d boxes
        # --- BEGIN --- Match objects and masks
        for ann_idx, ann in enumerate(anno['annotation']):
            assert(ann['category_id'] == objects[ann_idx].type)
            m = maskUtils.decode([ann['segmentation']])
            masks_2d.append(m)
            ann_bbox = np.copy(ann['bbox'])
            ann_bbox[2] = ann_bbox[0]+ann_bbox[2]
            ann_bbox[3] = ann_bbox[1]+ann_bbox[3]
            boxes_2d.append(ann_bbox)
        # --- END --- Match objects and masks
                
        pc_velo = dataset.get_lidar(data_idx)
        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:,0:3] = calib.project_velo_to_rect(pc_velo[:,0:3])
        pc_rect[:,3] = pc_velo[:,3]
        #print np.unique(pc_rect[:,2])
        img = dataset.get_image(data_idx)
        img_height, img_width, img_channel = img.shape
        _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:,0:3], calib, 0, 0, img_width, img_height, True)

        for obj_idx in range(len(objects)):
            if objects[obj_idx].type not in type_whitelist :continue

            # 2D BOX: Get pts rect backprojected 
            if use_tight_box:
                box2d = boxes_2d[obj_idx]
                print objects[obj_idx].box2d, box2d
            else:
                box2d = objects[obj_idx].box2d
            mask2d = masks_2d[obj_idx] 
            for _ in range(augmentX):
                # Augment data by box2d perturbation
                if perturb_box2d:
                    xmin,ymin,xmax,ymax = random_shift_box2d(box2d)
                    print box2d
                    print xmin,ymin,xmax,ymax
                else:
                    xmin,ymin,xmax,ymax = box2d

                # Get RGB image
                img_patch = my_crop_img(img, xmin, ymin, xmax, ymax)
                mask_patch = my_crop_img(mask2d, xmin, ymin, xmax, ymax)
                #print mask_patch.shape
                #raw_input()
                # Get depth image (sparse array of Nx4 u,v,depth,intensity)
                box_fov_inds = (pc_image_coord[:,0]<xmax) & (pc_image_coord[:,0]>=xmin) & (pc_image_coord[:,1]<ymax) & (pc_image_coord[:,1]>=ymin)
                box_fov_inds = box_fov_inds & img_fov_inds
                pc_in_box_fov = pc_rect[box_fov_inds,:]
                #print np.unique(pc_in_box_fov[:,2])
                uvmap = calib.project_rect_to_image(pc_in_box_fov[:,0:3])
                #print np.unique(pc_in_box_fov[:,2])
                uvmap[:,0] -= xmin # get relative uvmap
                uvmap[:,1] -= ymin
                output_array = np.concatenate((uvmap, pc_in_box_fov[:,2:4]), axis=1)
                #print np.unique(pc_in_box_fov[:,2])
                #print img_patch.shape, output_array.shape

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
                
                # Get 3d box center's relative uv coord
                box3d_center = (box3d_pts_3d[0,:]+box3d_pts_3d[6,:])/2.0
                box3d_center_uv = calib.project_rect_to_image(np.expand_dims(box3d_center,0)).squeeze()
                box3d_center_uv[0] -= xmin
                box3d_center_uv[1] -= ymin

                _,inds = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)
                label = np.zeros((pc_in_box_fov.shape[0]))
                label[inds] = 1
                # Get 3D BOX heading
                heading_angle = obj.ry
                # Get 3D BOX size
                box3d_size = np.array([obj.l, obj.w, obj.h])

                # Reject too far away object or object without points
                if xmax==xmin or ymax-ymin<25 or np.sum(label)==0:
                    continue

                if viz:

                    box2dUVDepth = np.array([[xmin,ymin,4], [xmax,ymin,4], [xmax,ymax,4], [xmin,ymax,4]])
                    pc1 = calib.project_image_to_rect(box2dUVDepth)
                    box2dUVDepth_far = np.array([[xmin,ymin,60], [xmax,ymin,60], [xmax,ymax,60], [xmin,ymax,60]])
                    pc2 = calib.project_image_to_rect(box2dUVDepth_far)

                    lidar_img = show_lidar_on_image2(pc_velo[:,0:3], np.zeros_like(img), calib, img_width, img_height)
                    lidar_img_patch = my_crop_img(lidar_img, xmin, ymin, xmax, ymax)
                    pc_image_coord_x = np.maximum(np.minimum(np.round(pc_image_coord[:,0]), img_width-1), 0).astype(np.int32)
                    pc_image_coord_y = np.maximum(np.minimum(np.round(pc_image_coord[:,1]), img_height-1), 0).astype(np.int32)
                    mask_fov_inds = (mask2d[pc_image_coord_y, pc_image_coord_x, 0] > 0)
                    mask_fov_inds = mask_fov_inds & box_fov_inds
                    pc_in_mask_fov = pc_rect[mask_fov_inds,:]
                    pc_in_box3d = pc_in_box_fov[inds,:]

                    Image.fromarray(np.tile(mask2d*255,(1,1,3)).astype(np.uint8)).show()
                    Image.fromarray(img_patch).show()
                    Image.fromarray(np.tile(mask_patch*255,(1,1,3)).astype(np.uint8)).show()
                    #print np.min(img_patch), np.max(img_patch)
                    depth_img = np.zeros((img_patch.shape[0],img_patch.shape[1],3))
                    for k in range(uvmap.shape[0]):
                        pc_image_coord_x = np.maximum(np.minimum(np.round(uvmap[k,0]), img_patch.shape[1]-1), 0).astype(np.int32)
                        pc_image_coord_y = np.maximum(np.minimum(np.round(uvmap[k,1]), img_patch.shape[0]-1), 0).astype(np.int32)
                        depth_img[pc_image_coord_y, pc_image_coord_x, :] = pc_in_box_fov[k,2]
                        #print pc_in_box_fov[k,2]
                    #print np.unique(pc_in_box_fov[:,2])
                    #print np.unique(depth_img)
                    depth_img -= np.min(depth_img)
                    depth_img /= np.max(depth_img)
                    Image.fromarray((depth_img*255).astype(np.uint8)).show()
                   
                    with open('tmp.pkl','wb') as fp:
                        cPickle.dump((img_patch, mask_patch, depth_img, pc_in_box_fov, pc_in_mask_fov, pc_in_box3d, lidar_img_patch, pc1, pc2), fp)
                    raw_input()
                    
                id_list.append(data_idx)
                box2d_list.append(np.array([xmin,ymin,xmax,ymax]))
                box3d_list.append(box3d_pts_3d)
                box3d_uv_list.append(box3d_center_uv)
                input_list.append((img_patch, mask_patch, output_array))
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
   
    save_zipped_pickle([id_list, box2d_list, box3d_list, box3d_uv_list, input_list, label_list, type_list, heading_list, box3d_size_list, frustum_angle_list], output_filename)

def extract_topdown_roi_seg(idx_filename, split, output_filename, viz, perturb_box2d=False, augmentX=1, type_whitelist=['Car']):
    ''' Extract training data pairs for (topdown proposal) RoI point set segmentation.
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
            #if objects[obj_idx].type=='DontCare':continue
            #if objects[obj_idx].type!='Car':continue
            if objects[obj_idx].type not in type_whitelist :continue
            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(objects[obj_idx], calib.P) 
            topdown_box2d = np.array([np.min(box3d_pts_3d[:,0]), np.min(box3d_pts_3d[:,2]), np.max(box3d_pts_3d[:,0]), np.max(box3d_pts_3d[:,2])]) # xmin,zmin,xmax,ymax

            # 2D BOX: Get pts rect backprojected 
            for _ in range(augmentX):
                # Augment data by box2d perturbation
                if perturb_box2d:
                    xmin,zmin,xmax,zmax = random_shift_box2d(topdown_box2d, 0.15)
                    print topdown_box2d
                    print xmin,zmin,xmax,zmax
                else:
                    xmin,zmin,xmax,zmax = topdown_box2d
                print xmin,zmin,xmax,zmax
                box_fov_inds = (pc_rect[:,0]<xmax) & (pc_rect[:,0]>=xmin) & (pc_rect[:,2]<zmax) & (pc_rect[:,2]>=zmin)
                box_fov_inds = box_fov_inds & img_fov_inds
                pc_in_box_fov = pc_rect[box_fov_inds,:]
                print pc_in_box_fov.shape
                # Get frustum angle (according to center pixel in 2D BOX)
                frustum_angle = 0
                # 3D BOX: Get pts velo in 3d box
                obj = objects[obj_idx]
                _,inds = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)
                label = np.zeros((pc_in_box_fov.shape[0]))
                label[inds] = 1
                # Get 3D BOX heading
                heading_angle = obj.ry
                # Get 3D BOX size
                box3d_size = np.array([obj.l, obj.w, obj.h])

                # Reject too far away object or object without points
                if np.sum(label)==0:
                    continue

                id_list.append(data_idx)
                box2d_list.append(np.array([xmin,zmin,xmax,zmax]))
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
    
    import cPickle as pickle
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
            print box2d_list[i]
            print box3d_list[i]
            print heading_list[i]
            print p1.shape
            #fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
            #draw_lidar(p1[:,0:3], fig=fig)
            fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4), fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:,0], p1[:,1], p1[:,2], seg, mode='point', colormap='gnuplot', scale_factor=1, figure=fig)
            fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4), fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:,2], -p1[:,0], -p1[:,1], seg, mode='point', colormap='gnuplot', scale_factor=1, figure=fig)
            raw_input()


def extract_topdown_detection(topdown_folder):
    dataset = kitti_object('/home/rqi/Data/KITTI/object')

    id_list = []
    type_list = []
    box2d_list = [] # top-down 2d box in rect camera coord
    prob_list = []
    for data_idx in range(len(dataset)):
        calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
        filename = os.path.join(topdown_folder, '%06d.txt'%(data_idx))
        if not os.path.exists(filename): continue
        topdown_proposals = [line.rstrip() for line in open(filename).readlines()]
        objects = []
        for line in topdown_proposals:
            obj = utils.Object3d(line)
            prob = float(line.split(' ')[-1])
            if prob<0.001: continue

            id_list.append(data_idx)
            type_list.append(obj.type)
            prob_list.append(prob)

            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P) 
            topdown_box2d = np.array([np.min(box3d_pts_3d[:,0]), np.min(box3d_pts_3d[:,2]), np.max(box3d_pts_3d[:,0]), np.max(box3d_pts_3d[:,2])]) # xmin,zmin,xmax,ymax

            box2d_list.append(topdown_box2d)
    return id_list, type_list, box2d_list, prob_list


def extract_topdown_roi_seg_from_topdown_detection(det_folder, split, output_filename, viz, valid_id_list=None, type_whitelist=['Car'], img_height_threshold=25, lidar_point_threshold=5):
    ''' Extract data pairs for RoI point set segmentation from topdown detector outputed 2D boxes (XZ in rect camera coord).
        Update: put lidar points and 3d box in *rect camera* coord system (as that in 3d box label files)
        
        Input:
            det_folder: prediction file in KITTI label format (data/000000.txt)
            split: corresponding to official either trianing or testing
            output_filename: the name for output .pickle file
            valid_id_list: specify a list of valid image IDs
        Output:
            None (will write a .pickle file to the disk)

        Usage: extract_roi_seg_from_rgb_detection("wcx_topdown/", "training", "roi_seg_val_topdown_detector_1118.pickle")

    '''
    dataset = kitti_object('/home/rqi/Data/KITTI/object', split)
    det_id_list, det_type_list, det_box2d_list, det_prob_list = extract_topdown_detection(det_folder)
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
        xmin,zmin,xmax,zmax = det_box2d_list[det_idx]
        box_fov_inds = (pc_rect[:,0]<xmax) & (pc_rect[:,0]>=xmin) & (pc_rect[:,2]<zmax) & (pc_rect[:,2]>=zmin)
        box_fov_inds = box_fov_inds & img_fov_inds
        pc_in_box_fov = pc_rect[box_fov_inds,:]
        # Get frustum angle (according to center pixel in 2D BOX)
        box2d_center = np.array([(xmin+xmax)/2.0, (zmin+zmax)/2.0])
        frustum_angle = 0

        #print '-------------------'
        #print data_idx
        #print det_type_list[det_idx]
        #print det_box2d_list[det_idx]
        #print det_prob_list[det_idx]
        #print np.min(pc_in_box_fov[:,0]), np.max(pc_in_box_fov[:,0]), np.min(pc_in_box_fov[:,2]), np.max(pc_in_box_fov[:,2])
        #raw_input()
       
        # Pass objects that are too small
        if len(pc_in_box_fov)<lidar_point_threshold:
            continue
        id_list.append(data_idx)
        type_list.append(det_type_list[det_idx])
        box2d_list.append(det_box2d_list[det_idx])
        prob_list.append(det_prob_list[det_idx])
        input_list.append(pc_in_box_fov)
        frustum_angle_list.append(frustum_angle)
    
    import cPickle as pickle
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


