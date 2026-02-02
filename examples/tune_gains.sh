#!/bin/bash
# Auto-tune PD gains for replay
# Usage: ./tune_gains.sh [coarse|fine|test]

MODE=${1:-coarse}

HAND=orca_hand
OBJ=box
SUBJECT=s01
USE_CLIP=01

if [ "$MODE" = "coarse" ]; then
    echo "Running COARSE sweep - BALANCED (small to large, 5x5x5 = 125 tests, ~20 min)..."
    echo "Tests from soft (may not track well) to stiff (may drop object)"
    python tune_replay_gains.py \
        --hand $HAND --obj $OBJ --subject $SUBJECT --use_clip $USE_CLIP \
        --coarse --max_steps 100

elif [ "$MODE" = "ultra" ]; then
    echo "Running ULTRA sweep - HIGH GAINS ONLY (4x3x3 = 36 tests, ~6 min)..."
    echo "WARNING: Tests very stiff gains - may cause object to fall!"
    python tune_replay_gains.py \
        --hand $HAND --obj $OBJ --subject $SUBJECT --use_clip $USE_CLIP \
        --ultra --max_steps 120

elif [ "$MODE" = "fine" ]; then
    echo "Running FINE sweep around best region (8x6x6 = 288 tests, ~40-50 min)..."
    python tune_replay_gains.py \
        --hand $HAND --obj $OBJ --subject $SUBJECT --use_clip $USE_CLIP \
        --fine --max_steps 150 \
        --kp_range 100 200 \
        --wrist_rot_kp_range 100 200 \
        --wrist_trans_kp_range 400 800 \
        --force_range 150

elif [ "$MODE" = "aggressive" ]; then
    # Removed - use 'ultra' instead
    echo "Mode 'aggressive' removed. Use './tune_gains.sh ultra' instead."
    exit 1
elif [ "$MODE" = "test" ]; then
    echo "Testing medium-high configuration..."
    python tune_replay_gains.py \
        --hand $HAND --obj $OBJ --subject $SUBJECT --use_clip $USE_CLIP \
        --kp_range 120 120 \
        --wrist_rot_kp_range 140 140 \
        --wrist_trans_kp_range 500 500 \
        --force_range 100 --max_steps 200 -v

else
    echo "Usage: ./tune_gains.sh [coarse|ultra|fine|test]"
    echo ""
    echo "  coarse - Balanced sweep from soft to stiff (125 tests, ~20 min) <-- START HERE"
    echo "  ultra  - Very high gain sweep - tests if too stiff (36 tests, ~6 min)"
    echo "  fine   - Narrow sweep around best (288 tests, ~40-50 min)"
    echo "  test   - Test single configuration (1 test, ~15 sec)"
    echo ""
    echo "TIP: Start with 'coarse' to find the sweet spot between:"
    echo "     - Too soft: Poor tracking, task fails"
    echo "     - Too stiff: Good tracking BUT object falls!"
    exit 1
fi
