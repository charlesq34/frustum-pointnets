''' Example usage:
    python viz.py --data_path roi_seg_box3d_caronly_val_0911.pickle --result_path test_results_caronly_aug5x.pickle

    Take GT box2d, eval 3D box estimation accuracy. Also able to visualize 3D predictions.
'''
import cPickle as pickle
import numpy as np
import argparse
from PIL import Image
import cv2
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import roi_seg_box3d_dataset
sys.path.append(os.path.join(BASE_DIR, '../sunrgbd_data'))
from sunrgbd_data import sunrgbd_object
from utils import load_zipped_pickle
sys.path.append(os.path.join(BASE_DIR, '../../train'))
sys.path.append(os.path.join(BASE_DIR, '../../mayavi'))
from box_util import box3d_iou

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default=None, help='data path for .pickle file, the one used for val in train.py [default: None]')
parser.add_argument('--result_path', default=None, help='result path for .pickle file from test.py [default: None]')
parser.add_argument('--viz', action='store_true', help='to visualize error result.')
parser.add_argument('--from_rgb_detection', action='store_true', help='test from data file from rgb detection.')
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

total_cnt = 0
correct_cnt = 0
type_whitelist=['bed','table','sofa','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub']
class_correct_cnt = {classname:0 for classname in type_whitelist}
class_total_cnt = {classname:0 for classname in type_whitelist}
for i in range(len(segp_list)):
    print " ---- %d/%d"%(i,len(segp_list))
    img_id = TEST_DATASET.id_list[i] 
    box2d = TEST_DATASET.box2d_list[i]
    classname = TEST_DATASET.type_list[i]

    objects = dataset.get_label_objects(img_id)
    target_obj = None
    for obj in objects: # **Assuming we use GT box2d for 3D box estimation evaluation**
        if np.sum(np.abs(obj.box2d-box2d))<1e-3:
            target_obj = obj
            break
    assert(target_obj is not None)

    box3d = TEST_DATASET.get_center_view_box3d(i)
    ps = ps_list[i]
    segp = segp_list[i].squeeze()
    center = center_list[i].squeeze()
    ret = TEST_DATASET[i]
    rot_angle = ret[7]

    # Get heading angle and size
    print heading_cls_list[i], heading_res_list[i], size_cls_list[i], size_res_list[i]
    heading_angle = roi_seg_box3d_dataset.class2angle(heading_cls_list[i], heading_res_list[i], 12)
    box_size = roi_seg_box3d_dataset.class2size(size_cls_list[i], size_res_list[i]) 
    corners_3d_pred = roi_seg_box3d_dataset.get_3d_box(box_size, heading_angle, center)

    # NOTE: fix this, box3d (projected from upright_depth coord) has flipped ymin,ymax as that in corners_3d_pred
    box3d_new = np.copy(box3d)
    box3d_new[0:4,:] = box3d[4:,:]
    box3d_new[4:,:] = box3d[0:4,:]
    iou_3d, iou_2d = box3d_iou(corners_3d_pred, box3d_new)
    print corners_3d_pred
    print box3d_new
    print 'Ground/3D IoU: ', iou_2d, iou_3d
    correct = int(iou_3d >= 0.25)
    total_cnt += 1
    correct_cnt += correct
    class_total_cnt[classname] += 1
    class_correct_cnt[classname] += correct

    if VISU: #and iou_3d<0.7:
        img_filename = os.path.join(IMG_DIR, '%06d.jpg'%(img_id))
        img = cv2.imread(img_filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        cv2.rectangle(img, (int(box2d[0]),int(box2d[1])), (int(box2d[2]),int(box2d[3])), (0,255,0), 3)
        Image.fromarray(img).show()

        # Draw figures
        fig = mlab.figure(figure=None, bgcolor=(0.6,0.6,0.6), fgcolor=None, engine=None, size=(1000, 500))
        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2, figure=fig)
        mlab.points3d(ps[:,0], ps[:,1], ps[:,2], segp, mode='point', colormap='gnuplot', scale_factor=1, figure=fig)
        draw_gt_boxes3d([box3d], fig, color = (0,0,1), draw_text=False)
        draw_gt_boxes3d([corners_3d_pred], fig, color = (0,1,0), draw_text=False)
        mlab.points3d(center[0], center[1], center[2], color=(0,1,0), mode='sphere', scale_factor=0.4, figure=fig)
        mlab.orientation_axes()
        raw_input()

print '-----------------------'
print 'Total cnt: %d, acuracy: %f' % (total_cnt, correct_cnt/float(total_cnt))
for classname in type_whitelist:
    print 'Class: %s\tcnt: %d\taccuracy: %f' % (classname.ljust(15), class_total_cnt[classname], class_correct_cnt[classname]/float(class_total_cnt[classname]))

