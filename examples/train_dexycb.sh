#!/bin/bash
# Training script for Orca Hand on DexYCB manipulation task

HAND=orca_hand
CLIP="20200709-subject-01/20200709_141754-0-63"

EXP_NAME=dexycb_hybrid_start_stable
python dexmachina/rl/train_rl_games.py -B 10000 -obf -obt --max_epochs 5000 \
    --data_source dexycb \
    --actuate_object --retarget_name para --horizon 16 -imw 0.5 --gain_mode all \
    --curr_schedule uniform --wait_epochs 100 --learning_rate 0.0003 \
    --upper_ratios 0.9 0.9 1 0.95 --lower_ratios 0.8 0.8 1 0.9 \
    --save_freq 5000 --group_collisions --fixed_mode uniform --uniform_mode slow \
    --action_penalty 0.01 --dialback_ep_len 80 --skip_grad --deque_len 30 \
    --task_rew_betas 10 1 5 --use_retarget_contact \
    --aux_reset_thres 0 0 0 --curr_rew_thres 0.5 0.01 0.01 0.01 \
    -ert 0.4 --contact_beta 10 \
    --hybrid_scales 0.1 1.0 --kp_init 80 --kv_init 5 \
    --thumb_weight 4.0 \
    --clip $CLIP -imi 0.3 -bc 0.3 -con 10 -exp $EXP_NAME --hand $HAND
