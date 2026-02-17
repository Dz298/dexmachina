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
OBJ="ketchup"
# OBJ="box"
HAND="orca_hand"
START_FRAME=30
# START_FRAME=130
MAX_STEPS=100  # Longer sequence for better visualization
OUTPUT_DIR="outputs/replay"
OUTPUT_NAME="${HAND}_${OBJ}_replay_${START_FRAME}_$((START_FRAME+MAX_STEPS)).mp4"


python examples/replay_retargeted_standalone.py \
    --obj "$OBJ" \
    --hand "$HAND" --start_frame "$START_FRAME" --max_steps "$MAX_STEPS" \
    --VOC --kp 80 --kv 5 \
    --record_video --output_dir "$OUTPUT_DIR" --output_name "$OUTPUT_NAME" --fps 30 "$@"

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
