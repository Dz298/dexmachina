#!/bin/bash
# Replay retargeted orca hand motion with video output

set -e

echo "==================================================================="
echo "Replaying Retargeted ORCA Hand Motion in Genesis"
echo "==================================================================="
echo ""
echo "This replays the retargeted joint positions to verify if the"
echo "retargeting result achieves the box manipulation task."
echo ""
echo "Mode: HEADLESS VIDEO RECORDING (for SSH)"
echo ""

# Default parameters
OBJ="box"
HAND="orca_hand"
MAX_STEPS=130  # Longer sequence for better visualization
OUTPUT_DIR="outputs/replay"
OUTPUT_NAME="orca_hand_replay_pd_control.mp4"

echo "Recording $MAX_STEPS steps to video..."
echo "Output: $OUTPUT_DIR/$OUTPUT_NAME"
echo ""

python examples/replay_retargeted_standalone.py \
    --obj "$OBJ" \
    --hand "$HAND" \
    --max_steps "$MAX_STEPS" \
    --record_video \
    --output_dir "$OUTPUT_DIR" \
    --output_name "$OUTPUT_NAME" \
    --fps 30 \
    --start_frame 100 \
    "$@"

echo ""
echo "==================================================================="
echo "✓ Replay complete!"
echo ""
echo "Video saved to: $OUTPUT_DIR/$OUTPUT_NAME"
echo ""
echo "Download the video to view:"
echo "  scp ROG2404:~/Projects/dexmachina/$OUTPUT_DIR/$OUTPUT_NAME ."
echo ""
echo "The video shows if retargeting achieves the task!"
echo "==================================================================="
