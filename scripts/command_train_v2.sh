#/bin/bash
python train/train.py --gpu 0 --model frustum_pointnets_v2 --log_dir train/log_v2 --num_point 1024 --max_epoch 201 --batch_size 24 --decay_step 800000 --decay_rate 0.5
