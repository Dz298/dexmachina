#!/bin/bash
# Training script for Orca Hand on box manipulation task

HAND=orca_hand
CLIP=box-30-230

# # Option 1: Hybrid mode + hand demo diff + trajectory lookahead (NEW)
# # Tests if observing hand deviation AND future trajectory improves learning
# # -ohdd: observe hand demo diff (deviation from demo)
# # -otl 5: observe trajectory lookahead (next 5 frames of demo as delta)
# EXP_NAME=hybrid_demo_diff_lookahead
# python dexmachina/rl/train_rl_games.py -B 6000 -obf -obt -ohdd -otl 5 --max_epochs 5000 \
#     --actuate_object --retarget_name para --horizon 16 -imw 0.5 --gain_mode all \
#     --curr_schedule uniform --wait_epochs 100 --learning_rate 0.0003 \
#     --contact_beta 10 --upper_ratios 0.9 0.9 1 --lower_ratios 0.8 0.8 1 \
#     --save_freq 5000 --group_collisions --fixed_mode uniform --uniform_mode slow \
#     --action_penalty 0.01 --dialback_ep_len 80 --skip_grad --deque_len 30 \
#     --task_rew_betas 10 1 5 --use_retarget_contact \
#     --aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0.01 0.01 0.01 \
#     -am hybrid --hybrid_scales 0.1 1.0 --kp_init 80 --kv_init 5 \
#     --clip $CLIP -imi 0.3 -bc 0.3 -con 10 -ert 0.6 -exp $EXP_NAME --hand $HAND

# # Option 3: Original hybrid mode (baseline, no new observations)
# EXP_NAME=hybrid_baseline
# python dexmachina/rl/train_rl_games.py -B 4096 -obf -obt --max_epochs 5000 \
#     --actuate_object --retarget_name para --horizon 16 -imw 0.5 --gain_mode all \
#     --curr_schedule uniform --wait_epochs 100 --learning_rate 0.0003 \
#     --contact_beta 10 --upper_ratios 0.9 0.9 1 --lower_ratios 0.8 0.8 1 \
#     --save_freq 5000 --group_collisions --fixed_mode uniform --uniform_mode slow \
#     --action_penalty 0.01 --dialback_ep_len 80 --skip_grad --deque_len 30 \
#     --task_rew_betas 10 1 5 --use_retarget_contact \
#     --aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0.01 0.01 0.01 \
#     -am hybrid --hybrid_scales 0.1 1.0 --kp_init 80 --kv_init 5 \
#     --clip $CLIP -imi 0.3 -bc 0.3 -con 10 -ert 0.6 -exp $EXP_NAME --hand $HAND

# # Option 6: Full run with thumb weighting (5000 epochs)
# EXP_NAME=hybrid_thumb_weight_4x_full
# python dexmachina/rl/train_rl_games.py -B 6000 -obf -obt --max_epochs 5000 \
#     --actuate_object --retarget_name para --horizon 16 -imw 0.5 --gain_mode all \
#     --curr_schedule uniform --wait_epochs 100 --learning_rate 0.0003 \
#     --contact_beta 10 --upper_ratios 0.9 0.9 1 --lower_ratios 0.8 0.8 1 \
#     --save_freq 5000 --group_collisions --fixed_mode uniform --uniform_mode slow \
#     --action_penalty 0.01 --dialback_ep_len 80 --skip_grad --deque_len 30 \
#     --task_rew_betas 10 1 5 --use_retarget_contact \
#     --aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0.01 0.01 0.01 \
#     -am hybrid --hybrid_scales 0.1 1.0 --kp_init 80 --kv_init 5 \
#     --thumb_weight 4.0 \
#     --clip $CLIP -imi 0.3 -bc 0.3 -con 10 -ert 0.6 -exp $EXP_NAME --hand $HAND

# Option 7: Residual mode + all features (thumb weight, lookahead, hand demo diff)
# -am residual: residual action mode (policy outputs delta from demo)
# -ohdd: observe hand demo diff (deviation from demo)
# -otl 5: observe trajectory lookahead (next 5 frames)
# --thumb_weight 4.0: 4x weight on thumb contacts
EXP_NAME=residual_full_features
python dexmachina/rl/train_rl_games.py -B 6000 -obf -obt -ohdd -otl 5 --max_epochs 5000 \
    --actuate_object --retarget_name para --horizon 16 -imw 0.5 --gain_mode all \
    --curr_schedule uniform --wait_epochs 100 --learning_rate 0.0003 \
    --contact_beta 10 --upper_ratios 0.9 0.9 1 --lower_ratios 0.8 0.8 1 \
    --save_freq 5000 --group_collisions --fixed_mode uniform --uniform_mode slow \
    --action_penalty 0.01 --dialback_ep_len 80 --skip_grad --deque_len 30 \
    --task_rew_betas 10 1 5 --use_retarget_contact \
    --aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0.01 0.01 0.01 \
    -am residual --res_cap --hybrid_scales 0.1 1.0 --kp_init 80 --kv_init 5 \
    --thumb_weight 4.0 \
    --clip $CLIP -imi 1.0 -bc 0.1 -con 15 -ert 0.6 -exp $EXP_NAME --hand $HAND
