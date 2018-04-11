#/bin/bash
python train/test.py --gpu 0 --num_point 1024 --model frustum_pointnets_v2 --model_path train/log_v2/model.ckpt --output train/detection_results_v2 --data_path kitti/frustum_carpedcyc_val_rgb_detection.pickle --from_rgb_detection --idx_path kitti/image_sets/val.txt --from_rgb_detection
train/kitti_eval/evaluate_object_3d_offline dataset/KITTI/object/training/label_2/ train/detection_results_v2
