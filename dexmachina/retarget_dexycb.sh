#!/bin/bash
# DexYCB retarget pipeline: process_dexycb -> parallel_retarget -> map_contacts
# Same layout as retarget.sh; includes commented viz commands.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HAND=orca_hand
SUBJECT=20200709-subject-01
SEQUENCE_ID=20200709_141754
CLIP="${SUBJECT}/${SEQUENCE_ID}-0-72"

# 1. Process DexYCB sequence to .npy (Genesis frame)
python retargeting/process_dexycb.py --sequence ${SUBJECT}/${SEQUENCE_ID} --save -ow

# 2. Retarget human hand to robot hand
python retargeting/parallel_retarget.py --data_source dexycb --clip "$CLIP" --hand $HAND \
    --control_steps 2000 --save_name para --save -ow

# 3. Map contacts (object surface -> hand links)
FNAME=assets/dexycb/processed/${SUBJECT}/${SEQUENCE_ID}.npy
python retargeting/map_contacts.py --hand $HAND --load_fname $FNAME

# # --- Visualize unretargeted (human hand joints + YCB object) ---
# DEXYCB_NPY="${SCRIPT_DIR}/assets/dexycb/processed/${SUBJECT}/${SEQUENCE_ID}.npy"
# python visualize_unretargeted.py --data_source dexycb --npy "$DEXYCB_NPY" \
#     --num_frames 0 --fps 30 --output_dir visualization_output \
#     --output_name dexycb_unretargeted.mp4 --create_video

# # --- Visualize retargeted (robot hand + YCB object) ---
DEXYCB_PT="${SCRIPT_DIR}/assets/retargeted/${HAND}/${SUBJECT}/${SEQUENCE_ID}_vector_para.pt"
DEXYCB_OBJ="002_master_chef_can"  # YCB object (from processed .npy; change if needed)
python visualize_hand_headless.py --load_file "$DEXYCB_PT" --obj "$DEXYCB_OBJ" --hand $HAND \
    --num_frames 0 --fps 30 --output_dir visualization_output \
    --output_name ${HAND}_dexycb_retargeted.mp4 --create_video
