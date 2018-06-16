### Data Preparation, Training and Evaluation of Frustum PointNets on SUN-RGBD data

CLAIM: This is still a beta release of the code, with lots of things to clarify -- but could be useful for some of you who would like to start earlier.

#### 1. Prepare SUN RGB-D data
Download <a href="http://rgbd.cs.princeton.edu">SUNRGBD V1 dataset</a> and toolkit

Run `extract_rgbd_data.m` in `sunrgbd_data/matlab/SUNRGBDtoolbox/`

The generated data should be organized a bit in the supposed mysunrgbd foder by moving all subfolders into /mysunrgbd/training/ and creating a train/val file list etc.

Prepare pickle files for TensorFlow training pipeline: 
run `sunrgbd_data/sunrgbd_data.py`

This will prepare frustum point clouds and labels and save them to zipped pickle files.

#### 2. Training

Run `train_one_hot.py` with the following parameters:

`batch_size=32, decay_rate=0.5, decay_step=800000, gpu=0, learning_rate=0.001, log_dir='log', max_epoch=151, model='frustum_pointnets_v1_sunrgbd', momentum=0.9, no_rgb=False, num_point=2048, optimizer='adam', restore_model_path=None`

#### 3. Testing and evaluation

To test the model on validation set you also need to prepare pickle files from detected 2D boxes in step 2 (last line in the main function of `sunrgbd_data.py`) -- the 2D detector should be trained to predict ``amodal'' 2D boxes.

You can run `test_one_hot.py` to test a trained frustum pointnet model with `--dump_result` flag, which will dump a pickle file for test results. And then run `evaluate.py` to evaluate the 3D AP with the dumped pickle file. We wrote our own 3D detection evaluation script because the original MATLAB one is too slow.

A typical evaluation script is like:
`python evaluate.py --data_path ../sunrgbd_data/fcn_det_val.zip.pickle --result_path test_results_v1_fcn_ft_val.pickle --from_rgb_detection`

