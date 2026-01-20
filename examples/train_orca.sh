#!/bin/bash
# Training script for Orca Hand on box manipulation task

HAND=orca_hand
CLIP=box-30-230

# # Option 1: Original hybrid mode (wrist residual + finger absolute)
# EXP_NAME=hybrid_baseline
# python dexmachina/rl/train_rl_games.py -B 8192 -obf -obt --max_epochs 5000 \
#     --actuate_object --retarget_name para --horizon 16 -imw 0.5 --gain_mode all \
#     --curr_schedule uniform --wait_epochs 100 --learning_rate 0.0003 \
#     --contact_beta 10 --upper_ratios 0.9 0.9 1 --lower_ratios 0.8 0.8 1 \
#     --save_freq 5000 --group_collisions --fixed_mode uniform --uniform_mode slow \
#     --action_penalty 0.01 --dialback_ep_len 80 --skip_grad --deque_len 30 \
#     --task_rew_betas 10 1 5 --use_retarget_contact \
#     --aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0.01 0.01 0.01 \
#     -am hybrid --hybrid_scales 0.1 1.0 --kp_init 80 --kv_init 5 \
#     --clip $CLIP -imi 0.3 -bc 0.3 -con 10 -ert 0.6 -exp $EXP_NAME --hand $HAND

# Option 2: Residual mode with res_cap (bounded wrist residual, unbounded finger residual)
EXP_NAME=residual-tune-rewcoeff
python dexmachina/rl/train_rl_games.py -B 8192 -obf -obt --max_epochs 5000 \
    --actuate_object --retarget_name para --horizon 16 -imw 0.5 --gain_mode all \
    --curr_schedule uniform --wait_epochs 100 --learning_rate 0.0003 \
    --contact_beta 10 --upper_ratios 0.9 0.9 1 --lower_ratios 0.8 0.8 1 \
    --save_freq 5000 --group_collisions --fixed_mode uniform --uniform_mode slow \
    --action_penalty 0.01 --dialback_ep_len 80 --skip_grad --deque_len 30 \
    --task_rew_betas 10 1 5 --use_retarget_contact \
    --aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0.01 0.01 0.01 \
    -am residual --res_cap --hybrid_scales 0.1 1.0 --kp_init 80 --kv_init 5 \
    --clip $CLIP -imi 1.0 -bc 0.1 -con 20 -ert 0.6 -exp $EXP_NAME --hand $HAND

# # Option 3: Residual mode with MUCH SLOWER curriculum decay
# # Changes:
# #   - upper_ratios: 0.9 -> 0.97 (decay 3% per step instead of 10%)
# #   - curr_rew_thres: 0.6 0.01 0.01 0.01 -> 0.8 0.1 0.1 0.1 (higher threshold to trigger decay)
# #   - wait_epochs: 200 (start decay later)
# EXP_NAME=residual_very_slow_decay
# python dexmachina/rl/train_rl_games.py -B 8192 -obf -obt --max_epochs 5000 \
#     --actuate_object --retarget_name para --horizon 16 -imw 0.5 --gain_mode all \
#     --curr_schedule uniform --wait_epochs 200 --learning_rate 0.0003 \
#     --contact_beta 10 --upper_ratios 0.97 0.97 1 --lower_ratios 0.9 0.9 1 \
#     --save_freq 5000 --group_collisions --fixed_mode uniform --uniform_mode slow \
#     --action_penalty 0.01 --dialback_ep_len 80 --skip_grad --deque_len 30 \
#     --task_rew_betas 10 1 5 --use_retarget_contact \
#     --aux_reset_thres 0 0 0 --curr_rew_thres 0.8 0.1 0.1 0.1 \
#     -am residual --res_cap --hybrid_scales 0.1 1.0 --kp_init 80 --kv_init 5 \
#     --clip $CLIP -imi 0.3 -bc 0.3 -con 10 -ert 0.6 -exp $EXP_NAME --hand $HAND

# ============== EVALUATION COMMANDS ==============
# Eval residual-tune-rewcoeff (adjust epoch number as needed)
# CKPT=logs/rl_games/orca_hand/orca-residual-tune-rewcoeff_box30-230-s01-u01_B8192_residual_thres0.6_ho16_imi1.0_con20.0_bc0.1/nn/orca_hand.pth
# python dexmachina/rl/eval_rl_games.py -ck $CKPT -ne 1 --record_video --render_dir orca_hand_eval --video_fname video.mp4

# Eval at specific epoch (e.g., epoch 2000)
# CKPT=logs/rl_games/orca_hand/orca-residual-tune-rewcoeff_box30-230-s01-u01_B8192_residual_thres0.6_ho16_imi1.0_con20.0_bc0.1/nn/orca_hand_ep2000.pth
# python dexmachina/rl/eval_rl_games.py -ck $CKPT -ne 1 --record_video --render_dir orca_hand_eval --video_fname video_ep2000.mp4

# Eval with reference trajectory visualization (requires -B 2)
# python dexmachina/rl/eval_rl_games.py -ck $CKPT -ne 1 -B 2 --show_reference --record_video --render_dir orca_hand_eval --video_fname video_with_ref.mp4
