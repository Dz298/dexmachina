#!/bin/bash
# Training script for Orca Hand on box manipulation task

HAND=orca_hand
CLIP=ketchup-30-130
RAND_INIT_RATIO=0.5
RAND_INIT_PIN_SECONDS=1.0

# Full run with thumb weighting (5000 epochs) + matched contact normal alignment (kappa=2)
EXP_NAME=hybrid_rand_init_pin_1s
python dexmachina/rl/train_rl_games.py -B 6000 -obf -obt --max_epochs 5000 \
    --actuate_object --retarget_name para --horizon 16 -imw 0.5 --gain_mode all \
    --curr_schedule uniform --wait_epochs 100 --learning_rate 0.0003 \
    --upper_ratios 0.9 0.9 1 0.95 0.95 --lower_ratios 0.8 0.8 1 0.9 0.9 \
    --save_freq 500 --group_collisions --fixed_mode uniform --uniform_mode slow \
    --action_penalty 0.01 --dialback_ep_len 80 --skip_grad --deque_len 30 \
    --task_rew_betas 10 1 5 --use_retarget_contact \
    --aux_reset_thres 0 0 0 --curr_rew_thres 0.5 0.01 0.01 0.01 \
    -randr $RAND_INIT_RATIO --rand_init_pin_seconds $RAND_INIT_PIN_SECONDS \
    -am hybrid \
    -ert 0.4 --contact_beta 10 \
    --hybrid_scales 0.1 1.0 --kp_init 80 --kv_init 5 \
    --use_virtual_force_assist --virtual_force_alpha_init 1.0 \
    --virtual_force_delta 0.001 --virtual_force_kp 40 --virtual_force_kd 4 \
    --virtual_force_sigma 0.03 --virtual_force_fmax 1.5 \
    --virtual_force_link_keywords thumb index middle ring pinky \
    --thumb_weight 4.0 \
    --reach_rew_weight 0.5 --reach_sigma 0.2 \
    --grasp_gate_weight 0.5 --grasp_gate_force_threshold 1.0 --grasp_gate_min_fingers 2 \
    --clip $CLIP -imi 0.3 -bc 0.3 -con 10 -exp $EXP_NAME --hand $HAND 
    


# # # ============================================================================
# EXP_NAME=residual_less_stiff_gains_start_in_contact
# python dexmachina/rl/train_rl_games.py -B 6000 -obf -obt --max_epochs 5000 \
#     --actuate_object --retarget_name para --horizon 16 -imw 0.5 --gain_mode all \
#     --curr_schedule uniform --wait_epochs 100 --learning_rate 0.0003 \
#     --contact_beta 10 --upper_ratios 0.9 0.9 1 0.95 --lower_ratios 0.8 0.8 1 0.9 \
#     --save_freq 5000 --group_collisions --fixed_mode uniform --uniform_mode slow \
#     --action_penalty 0.01 --dialback_ep_len 80 --skip_grad --deque_len 30 \
#     --task_rew_betas 10 1 5 --use_retarget_contact \
#     --aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0.01 0.01 0.01 \
#     -am residual --res_cap --hybrid_scales 0.1 1.0 --kp_init 80 --kv_init 5 \
#     --thumb_weight 4.0 \
#     --clip $CLIP -imi 0.3 -bc 0.3 -con 10 -ert 0.6 -exp $EXP_NAME --hand $HAND


# # ============================================================================
# Policy residual mode: Train a residual policy on top of a trained base policy
# This trains a policy that adds corrections to an already-trained policy
# instead of on top of blind retargeted motion
# 
# USAGE: First train a base policy (e.g., hybrid mode above), then use its
# checkpoint path with -bpp flag
# ============================================================================
# BASE_POLICY_PATH="logs/rl_games/orca_hand/orca-hybrid_less_stiff_gains_start_in_contact_box100-230-s01-u01_B6000_hybrid_thres0.6_ho16_imi0.3_con10.0_bc0.3/nn/orca_hand.pth"
# EXP_NAME=policy_residual_constant_margin
# python dexmachina/rl/train_rl_games.py -B 6000 -obf -obt --max_epochs 5000 \
#     --actuate_object --retarget_name para --horizon 16 -imw 0.5 --gain_mode all \
#     --curr_schedule uniform --wait_epochs 100 --learning_rate 0.0003 \
#     --contact_beta 10 --upper_ratios 0.9 0.9 1 0.95 --lower_ratios 0.8 0.8 1 0.9 \
#     --save_freq 5000 --group_collisions --fixed_mode uniform --uniform_mode slow \
#     --action_penalty 0.01 --dialback_ep_len 80 --skip_grad --deque_len 30 \
#     --task_rew_betas 10 1 5 --use_retarget_contact \
#     --aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0.01 0.01 0.01 \
#     -am policy_residual --res_cap --hybrid_scales 0.1 1.0 --kp_init 80 --kv_init 5 \
#     --base_policy_path $BASE_POLICY_PATH \
#     --thumb_weight 4.0 \
#     --clip $CLIP -imi 0.3 -bc 0.3 -con 10 -ert 0.6 -exp $EXP_NAME --hand $HAND
