#!/bin/bash
# Visualize retargeted orca hand motion

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate dexmachina

OBJ=box
HAND=orca_hand
SUBJECT=s01
USE_CLIP=01

echo "=== Retargeted Hand Visualization ==="
echo ""
echo "Creating headless visualization (video output for SSH/remote environments)"
echo ""

# Headless rendering (RECOMMENDED FOR SSH/REMOTE)
python visualize_hand_headless.py \
    --obj $OBJ \
    --hand $HAND \
    --subject $SUBJECT \
    --use_clip $USE_CLIP \
    --num_frames 120 \
    --fps 30 \
    --output_dir visualization_output \
    --output_name ${HAND}_${OBJ}_retargeted.mp4 \
    --create_video

echo ""
echo "Other options (uncomment to use):"
echo ""
echo "# Option 2: Render all frames (full trajectory)"
echo "# python visualize_hand_headless.py --obj \$OBJ --hand \$HAND --num_frames 0 --fps 60 --output_name full_trajectory.mp4"
echo ""
echo "# Option 3: Save individual frames as images"
echo "# python visualize_hand_headless.py --obj \$OBJ --hand \$HAND --num_frames 120 --save_frames --output_dir frames_output"
echo ""
echo "# Option 4: Single hand only"
echo "# python visualize_hand_headless.py --obj \$OBJ --hand \$HAND --hand_side left --both_hands false"
echo ""
echo "# Option 5: Live viewer (only works with local display/X11 forwarding)"
echo "# python visualize_hand_simple.py --obj \$OBJ --hand \$HAND --subject \$SUBJECT --use_clip \$USE_CLIP --vis"
