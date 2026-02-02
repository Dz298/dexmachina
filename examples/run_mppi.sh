#!/bin/bash
# Run MPPI residual controller with policy-guided sampling
#
# MPPI is much faster than DIAL-MPC while maintaining control quality.
# Default configuration targets ~20-50Hz real-time performance.
#
# Usage: 
#   ./run_mppi.sh [checkpoint_path] [options]
#
# If checkpoint_path is omitted, uses your specified checkpoint.
#
# Examples:
#   ./run_mppi.sh --vis                          # Use default checkpoint, show visualization
#   ./run_mppi.sh --record_video                 # Record video
#   ./run_mppi.sh $CK --vis -N 64 -H 8           # Custom checkpoint, 64 samples, horizon 8
#   ./run_mppi.sh --policy_only                  # Fastest: policy only, no MPPI
#
# Performance guide (approximate on RTX 4090):
#   --policy_only                : ~500Hz (2ms/step)
#   -N 32 -H 6 (default)         : ~20-50Hz (20-50ms/step)
#   -N 64 -H 8                   : ~10-20Hz (50-100ms/step)
#   -N 128 -H 10                 : ~5-10Hz (100-200ms/step)

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate dexmachina

# Your specified default checkpoint
DEFAULT_CK="logs/rl_games/orca_hand/orca-hybrid_thumb_weight_4x_full_box30-230-s01-u01_B6000_hybrid_thres0.6_ho16_imi0.3_con10.0_bc0.3/nn/orca_hand.pth"

# Check if first argument is a checkpoint file or a flag
if [[ -n "$1" && -f "$1" ]]; then
    CK="$1"
    shift
else
    CK=$DEFAULT_CK
fi

echo "=============================================="
echo "Running MPPI Residual Controller"
echo "=============================================="
echo "Checkpoint: $CK"
echo "Additional args: $@"
echo ""

python examples/run_mppi_policy_guided.py \
    --checkpoint "$CK" \
    --num_samples 32 \
    --horizon 16 \
    --lambda_temp 1.0 \
    --noise_scale 0.2 \
    --fast_mode \
    "$@"
