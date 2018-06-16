''' Cluster and visualize distribution of box3d '''

import cPickle as pickle
with open('box3d_dimensions.pickle','rb') as fp:
    type_list = pickle.load(fp)
    dimension_list = pickle.load(fp) # l,w,h
    ry_list = pickle.load(fp)

import numpy as np
box3d_pts = np.vstack(dimension_list)
print box3d_pts.shape

print set(type_list)
raw_input()


# Get average box size for different catgories
median_box3d_list = []
for class_type in sorted(set(type_list)):
    cnt = 0
    box3d_list = []
    for i in range(len(dimension_list)):
        if type_list[i]==class_type:
            cnt += 1
            box3d_list.append(dimension_list[i])
    #print class_type, cnt, box3d/float(cnt)
    median_box3d = np.median(box3d_list,0)
    print "\'%s\': np.array([%f,%f,%f])," % (class_type, median_box3d[0]*2, median_box3d[1]*2, median_box3d[2]*2)
    median_box3d_list.append(median_box3d)
raw_input()

import mayavi.mlab as mlab
fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
mlab.points3d(box3d_pts[:,0], box3d_pts[:,1], box3d_pts[:,2], mode='point', colormap='gnuplot', scale_factor=1, figure=fig)
##draw axis
mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

axes=np.array([
    [2.,0.,0.,0.],
    [0.,2.,0.,0.],
    [0.,0.,2.,0.],
],dtype=np.float64)
fov=np.array([  ##<todo> : now is 45 deg. use actual setting later ...
    [20., 20., 0.,0.],
    [20.,-20., 0.,0.],
],dtype=np.float64)

mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)
mlab.plot3d([0, fov[0,0]], [0, fov[0,1]], [0, fov[0,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)
mlab.plot3d([0, fov[1,0]], [0, fov[1,1]], [0, fov[1,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)
mlab.orientation_axes()

for box in median_box3d_list:
    mlab.points3d(box[0], box[1], box[2], color=(1,0,1), mode='sphere', scale_factor=0.4)
raw_input()
