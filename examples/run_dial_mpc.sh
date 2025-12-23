#!/bin/bash
# Run DIAL MPC with policy-guided residual sampling
#
# This script runs MPC that samples RESIDUAL corrections on top of a trained 
# RL policy, with optional virtual contact constraints for better convergence.
#
# Usage: 
#   ./run_dial_mpc.sh [checkpoint_path] [options]
#
# If checkpoint_path is omitted, uses the default checkpoint.
#
# Examples:
#   ./run_dial_mpc.sh --vis                         # Use default checkpoint
#   ./run_dial_mpc.sh --record_video --num_samples 512
#   ./run_dial_mpc.sh $CK --vis                     # Use custom checkpoint
#   ./run_dial_mpc.sh $CK --kp_max 100 --kv_max 10  # Enable virtual contact constraint
#
# Real-time deployment options (for speed):
#   ./run_dial_mpc.sh --policy_only                 # Fastest: policy only, no MPC
#   ./run_dial_mpc.sh --fast_mode -N 32 -H 4 -I 1   # Fast MPC: 32 samples, horizon 4, 1 iter
#   ./run_dial_mpc.sh --fast_mode                   # Fast MPC with default params
#
# Speed guide (approximate on RTX 4090):
#   --policy_only                    : ~500Hz (2ms/step)
#   --fast_mode -N 32 -H 4 -I 1      : ~50Hz  (20ms/step)
#   --fast_mode -N 64 -H 6 -I 1      : ~20Hz  (50ms/step)
#   Default (N=1024, H=10, I=2)      : ~0.5Hz (2s/step)

set -e

# Default checkpoint (update this to your trained policy)
DEFAULT_CK="logs/rl_games/orca_hand/orca-default_retargeted_tuned_gains_box30-230-s01-u01_B8192_hybrid_thres0.6_ho16_imi0.3_con10.0_bc0.3/nn/orca_hand.pth"

# Check if first argument is a checkpoint file or a flag
# If it's a file that exists, use it as checkpoint; otherwise use default and keep all args
if [[ -n "$1" && -f "$1" ]]; then
    CK="$1"
    shift  # Remove checkpoint from args since we consumed it
else
    CK=$DEFAULT_CK
    # Don't shift - all args are flags to pass through
fi

echo "=============================================="
echo "Running DIAL MPC with Policy-Guided Residual Sampling"
echo "=============================================="
echo "Checkpoint: $CK"
echo "Additional args: $@"
echo ""

python examples/run_dial_mpc_policy_guided.py \
    --checkpoint "$CK" \
    --horizon 10 \
    --num_iterations 2 \
    --noise_scale 0.2 \
    --noise_decay 0.8 \
    --temperature 0.1 \
    --show_traces \
    --n_traces 50 \
    --kp_max 100 \
    --kv_max 10
    "$@"
